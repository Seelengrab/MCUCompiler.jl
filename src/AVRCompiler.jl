module AVRCompiler

using GPUCompiler
using LLVM
using Pkg: Pkg

using Dates

using avr_binutils_jll
using avrdude_jll

#####
# Compiler Target
#####

struct Arduino <: GPUCompiler.AbstractCompilerTarget end

GPUCompiler.llvm_triple(::Arduino) = "avr-unknown-unkown"
GPUCompiler.runtime_slug(j::GPUCompiler.CompilerJob{Arduino}) = j.config.params.name

struct ArduinoParams <: GPUCompiler.AbstractCompilerParams
    name::String
end

module StaticRuntime
    # the runtime library
    signal_exception() = return
    malloc(sz) = C_NULL
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

GPUCompiler.runtime_module(::GPUCompiler.CompilerJob{<:Any,ArduinoParams}) = StaticRuntime
GPUCompiler.runtime_module(::GPUCompiler.CompilerJob{Arduino}) = StaticRuntime
GPUCompiler.runtime_module(::GPUCompiler.CompilerJob{Arduino,ArduinoParams}) = StaticRuntime

function native_job(@nospecialize(func), @nospecialize(types), params)
    @info "Creating compiler job for '$func$types'"
    source = GPUCompiler.methodinstance(
                typeof(func), # our function
                Base.to_tuple_type(types)) # its signature
    target = Arduino()
    config = GPUCompiler.CompilerConfig(target, params;
                                        kernel=false,
                                        name=String(nameof(func)),
                                        always_inline=false)
    job = GPUCompiler.CompilerJob(source, config)
end

function build_ir(job, @nospecialize(func), @nospecialize(types); optimize=true)
    @info "Bulding LLVM IR for '$func$types'"
    ir, ir_meta = GPUCompiler.emit_llvm(
                    job; # our job
                    libraries=false, # whether this code uses GPU libraries
                    deferred_codegen=false, # should we resolve codegen?
                    optimize=optimize, # do we want to optimize the llvm?
                    only_entry=false, # only keep the entry point/inline everything?
                    ctx=JuliaContext()) # the LLVM context to use
    return ir, ir_meta
end

function build_obj(@nospecialize(func), @nospecialize(types), params=ArduinoParams("unnamed")
                   ;strip=true, validate=true)
    job = native_job(func, types, params)
    ir, ir_meta = build_ir(job, func, types)
    @info "Compiling AVR ASM for '$func$types'"
    obj, _ = GPUCompiler.emit_asm(
                job, # our job
                ir; # the IR we got
                strip=strip, # should the binary be stripped of debug info?
                validate=validate, # should the LLVM IR be validated?
                format=LLVM.API.LLVMObjectFile) # What format would we like to create?
    return obj
end

function builddump(@nospecialize(func), @nospecialize(args))
   obj = build_obj(func, args)
   mktemp() do path, io
       write(io, obj)
       flush(io)
       str = avr_objdump() do bin
            read(`$bin -dr $path`, String)
       end
   end |> print
end

#####
# Documented Interface
#####

"""
    build(mod::Module[, outpath::String]; clear=true)

Compile the module `mod` and prepare it for flashing to a device, building an ELF.
This function expects a `main()` without arguments to exist, which will be used as the entry point.
`main` needs to be exported from `mod`.

The built product will be placed in the `out` directory of the current project,
clearing the directory before writing. This can be overwritten by specifying `outpath`.
If the directory does not exist, it will be created.
Specifying `clear=false` will instead place the output in a subdirectory of `outpath`, named after the date & build time.

Returns the path to the main executable ELF.
"""
build(mod::Module; clear=true) = build(mod, joinpath(dirname(Pkg.project().path), "out/"); clear)

function build(mod::Module, outpath; clear=true)
    !isdirpath(outpath) && throw(ArgumentError("Given path `$path` does not describe a directory (doesn't end with a path seperator like `/`)!"))
    !hasproperty(mod, :main) && throw(ArgumentError("Module `$mod` doesn't have a `main` entrypoint!"))
    any(m -> isone(m.nargs), methods(mod.main)) || throw(ArgumentError("`main` has no method taking zero arguments!"))

    @info "Building main object file"
    buildpath = mktempdir()
    params = ArduinoParams(String(nameof(mod)))
    obj = build_obj(mod.main, (), params)
    mainobj_name = string(nameof(mod), ".o")
    builtobj_path = joinpath(buildpath, mainobj_name)
    open(builtobj_path, "w") do io
        write(io, obj)
        flush(io)
    end

    @info "Linking object files to ELF"
    mainelf_name = string(nameof(mod), ".elf")
    builtelf_name = joinpath(buildpath, mainelf_name)
    # TODO: Link with vector table shenanigans, if needed
    avr_ld() do bin
        run(`$bin -v -o $builtelf_name $builtobj_path`)
    end

    mainhex_name = string(nameof(mod), ".hex")
    builthex_name = joinpath(buildpath, mainhex_name)
    avr_objcopy() do bin
        run(`$bin -O ihex $builtelf_name $builthex_name`)
    end

    @info "Moving files from temporary directory to output directory"
    clear || (outpath = joinpath(outpath, string(now())))
    mkpath(outpath)
    mv(buildpath, outpath; force=true)
    joinpath(outpath, mainhex_name)
end

"""
    list_mcus()

List the microcontrollers supported by `avrdude`.
"""
function list_mcus()
    avrdude() do bin
        run(Cmd(`$bin -p \?`; ignorestatus=true))
    end
    nothing
end

"""
    flash(path, bin, partno
          ; clear=true, verify=true, programmer=:arduino)

Flash the binary `bin` to the device connected at `path`.
`partno` specifies the microcontroller that will be flashed.

 * `clear` specifies whether to clear the flash ROM of the device
 * `verify` tells the programmer to verify the written data
 * `programmer` specifies the programmer to use for flashing

!!! warn "Defaults"
    This is intended as a convenient interface to `avrdude` from `avrdude_jll`.
    For more complex configurations, consider using the JLL directly.
    The defaults specified here are only tested for an Arduino Ethernet with an ATmega328p.

!!! warn "Warranty"
    Using this to flash your device is not guaranteed to succeed and no warranty of any kind
    is given. Use at your own risk.
"""
function flash(path, binpath, partno; clear=true, verify=true, programmer=:arduino)
    ispath(path) || throw(ArgumentError("`$path` is not a path."))
    isfile(binpath) || throw(ArgumentError("`$binpath` is not a file."))
    flasharg = ':' in binpath ? `flash:w:$binpath:a` : `$binpath`
    verifyarg = verify ? `` : `-V`
    cleararg = clear ? `` : `-D`
    avrdude() do bin
        run(`$bin $verifyarg -c $programmer -p $partno -P $path $cleararg -U $flasharg`)
    end
end

end # module AVRCompiler
