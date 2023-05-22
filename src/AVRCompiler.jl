module AVRCompiler

using GPUCompiler
using LLVM
using JET

using Pkg: Pkg
using Dates
using Logging

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

function avr_job(@nospecialize(func), @nospecialize(types), params=ArduinoParams("$(nameof(func))"))
    # @info "Creating compiler job for '$func$types'"
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
    build(mod::Module[, outpath::String]; clear=false, target=:build)

Compile the module `mod` and prepare it for flashing to a device, building an ELF.
This function expects a `main()` without arguments to exist, which will be used as the entry point.
`main` needs to be exported from `mod`.

The built product will be placed in `outpath` under a directory named after the build date & time.
The default for `outpath` is the `out` directory under the project root of the given module.
`clear` specifies whether `outpath` should be cleared before building.

`target` can be one of these:

    * `:build` produce a `.o` file as part of the compilation process
    * `:llvm` instead of building an object, output LLVM IR. It is usually preferrable to use `GPUCompiler.code_llvm` for this in combination with `AVRCompiler.avr_job`.
    * `:asm` instead of building an object, output ASM. It is usually preferrable to inspect the actual binary after compilation & linking.

A symlink `latest` pointing to the latest built binary will be created.
"""
build(mod::Module; clear=true, target=:build, strip=true) = build(mod, joinpath(dirname(Pkg.project().path), "out/"); clear, target, strip)

function build(mod::Module, outpath; clear=false, target=:build, strip=true)
    !isdirpath(outpath) && throw(ArgumentError("Given path `$path` does not describe a directory (doesn't end with a path seperator like `/`)!"))
    !hasproperty(mod, :main) && throw(ArgumentError("Module `$mod` doesn't have a `main` entrypoint!"))
    target in (:build, :llvm, :asm)  || throw(ArgumentError("Supplied unsupported target: `$target`"))
    any(m -> isone(m.nargs), methods(mod.main)) || throw(ArgumentError("`main` has no method taking zero arguments!"))
    buildpath = mktempdir()

    @info "Checking for statically known problems"
    static_errors = false
    callresults = JET.report_call(mod.main, ())
    if !isempty(JET.get_reports(callresults))
        display(callresults)
        static_errors = true
    end
    optresults = JET.report_opt(mod.main, ())
    if !isempty(JET.get_reports(optresults))
        display(optresults)
        static_errors = true
    end
    # Fix your errors before compilation :^)
    if static_errors
        @error "Static errors detected - aborting compilation"
        return
    end

    params = ArduinoParams(String(nameof(mod)))
    obj = GPUCompiler.compile(target, native_job(mod.main, (), params); ctx=GPUCompiler.JuliaContext(), strip)[1]
    if target != :build
        print(obj)
        return
    end
    @info "Building main object file"
    mainobj_name = string(nameof(mod), ".o")
    builtobj_path = joinpath(buildpath, mainobj_name)
    open(builtobj_path, "w") do io
        write(io, obj)
        flush(io)
    end

    @info "Building vectors"
    vectorasm_path = joinpath(buildpath, "vectors.asm")
    vectorobj_path = joinpath(buildpath, "vectors.o")
    open(vectorasm_path, "w") do io
        println(io, """
          .vectors:
                rjmp main
          """)
        # TODO: `println` additional calls for interrupt vectors
    end
    avr_as() do bin
        run(`$bin -o $vectorobj_path $vectorasm_path`)
    end

    @info "Linking object files to ELF"
    mainelf_name = string(nameof(mod), ".elf")
    builtelf_name = joinpath(buildpath, mainelf_name)
    avr_ld() do bin
        run(`$bin -v -o $builtelf_name $vectorobj_path $builtobj_path`)
    end

    mainhex_name = string(nameof(mod), ".hex")
    builthex_name = joinpath(buildpath, mainhex_name)
    avr_objcopy() do bin
        run(`$bin -O ihex $builtelf_name $builthex_name`)
    end

    @info "Moving files from temporary directory to output directory"
    if clear
        rm(outpath; force=true, recursive=true)
    end
    mkpath(outpath)
    latestpath = joinpath(outpath, "latest")
    outpath = joinpath(outpath, string(now()))
    mv(buildpath, outpath)
    if ispath(latestpath)
        rm(latestpath)
    end
    symlink(basename(outpath), latestpath; dir_target=true)
    nothing
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
