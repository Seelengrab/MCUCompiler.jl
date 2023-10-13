module MCUCompiler

using GPUCompiler
using LLVM
using JET

using Pkg: Pkg
using Dates
using Logging

const PlatformParams = GPUCompiler.AbstractCompilerParams

include("array.jl")

struct MCUTarget{T,P} <: GPUCompiler.AbstractCompilerTarget
    params::P
    MCUTarget{T}(params::P) where {T,P <: PlatformParams} = new{T, P}(params)
end

include("runtime.jl")

"""
    triple(::MCUTarget)

Returns the target triple string for this compilation target.
"""
function triple end

"""
    build_vectors(::MCUTarget, buildpath)

Assemble an object file containing a section for the vector jump table.

!!! note "Jumping to main"
    You may need to insert an explicit jump to `main` into your vector table, depending on how your MCU is initialized.
"""
function build_vectors end

"""
    link(::MCUTarget, elf_path, vectors, obj)

Link the vectors' object file and the other binary object file to an ELF.
"""
function link end

"""
    postprocess(::MCUTarget, buildpath)

Do any necessary postprocessing steps on the built ELF, such as converting to `ihex` for `avrdude`.
"""
function postprocess end

GPUCompiler.llvm_triple(mcut::MCUTarget) = triple(mcut)
GPUCompiler.runtime_slug(j::GPUCompiler.CompilerJob{<:MCUTarget}) = j.config.params.name

function mcu_job(@nospecialize(func), @nospecialize(types), platform)
    # @info "Creating compiler job for '$func$types'"
    source = GPUCompiler.methodinstance(
                typeof(func), # our function
                Base.to_tuple_type(types)) # its signature
    config = GPUCompiler.CompilerConfig(platform, platform.params;
                                        kernel=false,
                                        name=String(nameof(func)),
                                        always_inline=false)
    job = GPUCompiler.CompilerJob(source, config)
end

# target specific definitions
include("arduino.jl")

const jlfuncs = (cglobal(:jl_alloc_array_1d) => :jl_alloc_array_1d,
                 cglobal(:jl_alloc_array_2d) => :jl_alloc_array_2d,
                 cglobal(:jl_alloc_array_3d) => :jl_alloc_array_3d,
                 cglobal(:jl_new_array) => :jl_new_array,
                 cglobal(:jl_array_copy) => :jl_array_copy,
                 cglobal(:jl_alloc_string) => :jl_alloc_string,
                 cglobal(:jl_in_threaded_region) => :jl_in_threaded_region,
                 cglobal(:jl_enter_threaded_region) => :jl_enter_threaded_region,
                 cglobal(:jl_exit_threaded_region) => :jl_exit_threaded_region,
                 cglobal(:jl_set_task_tid) => :jl_set_task_tid,
                 cglobal(:jl_new_task) => :jl_new_task,
                 cglobal(:jl_array_grow_beg) => :jl_array_grow_beg,
                 cglobal(:jl_array_grow_end) => :jl_array_grow_end,
                 cglobal(:jl_array_grow_at) => :jl_array_grow_at,
                 cglobal(:jl_array_del_beg) => :jl_array_del_beg,
                 cglobal(:jl_array_del_end) => :jl_array_del_end,
                 cglobal(:jl_array_del_at) => :jl_array_del_at,
                 cglobal(:jl_array_ptr) => :jl_array_ptr,
                 cglobal(:jl_value_ptr) => :jl_value_ptr,
                 cglobal(:jl_get_ptls_states) => :jl_get_ptls_states,
                 cglobal(:jl_gc_add_finalizer_th) => :jl_gc_add_finalizer_th,
                 cglobal(:malloc) => :malloc,
                 cglobal(:memmove) => :memmove,
                 cglobal(:jl_symbol_n) => :jl_symbol_n)

"""
   is_object_moveable(obj)

Heuristic for whether the object could be moved to a statically compiled LLVM module.

This essentially checks whether the object is pointerfree, by checking whether all its fields are allocated inline.
This is more conservative than necessary, since e.g. the following `f` would also be moveable, due to the memory of `Foo`
always having a defined size. The wrapped `Bar` would also need to be moved to the module.

```
mutable struct Bar
   a::Int
end

mutable struct Foo
   b::Bar
end

const f = Foo(Bar(1))
```

Theoretically, the only objects that can't be moved like this are `Array`s and `String`s, due to their size not being a function of their type.
"""
function is_object_moveable(obj)
    mapreduce(Base.allocatedinline, &, fieldtypes(typeof(obj)); init=true)
end

const llmod = Ref{LLVM.Module}()
const globalNamesCount = Dict{DataType, Int}()

function GPUCompiler.process_module!(@nospecialize(job::GPUCompiler.CompilerJob{<:MCUTarget}), mod::LLVM.Module)
    return
    ctx = context(mod)
    llmod[] = mod
    empty!(globalNamesCount)
    interned_objects = Dict{UInt, Any}()

    for f in functions(mod), b in blocks(f), i in instructions(b), o in operands(i)
        # only look at `inttoptr` for now
        # and only if they're conversions from constants
        o isa LLVM.ConstantExpr || continue
        opcode(o) === LLVM.API.LLVMIntToPtr || continue
        ptr_from_const  = o

        # TODO: This really needs to be generalized a bit... or Julia doesn't emit them anymore later?
        # patch objects that we can actually intern here
        # For now, we just _assume_ that pointers larger than 0xFFFF are
        # valid julia pointers. The number is the largest SRAM of any ATmega.
        # This also ignores AVR32.

        # The general steps are as follows:
        #  * Get the object this is pointing to
        #     (unless we already did that - then skip to the last step)
        #  * Check if its internable
        #  * Intern the object by copying its plain bits to an LLVM global
        #  * patch the use by replacing the `inttoptr` with just a pointer to the global

        ptrconst = only(operands(ptr_from_const))
        ptrconst isa LLVM.ConstantInt || continue
        ptrval = convert(UInt, ptrconst)
        # TODO: sometimes, inttoptr constants in the IR are truncated to the target pointersize, so do a second pass later to check whether we're in the extent of a known object with those.
        if ptrval > 0xFFFF
            @warn """
                Encountered a pointer literal greater than the available memory.
                This likely means julia emitted some literal to either a global constant,
                or to some function global of the current session. This will lead to problems.
                """ Ptr=ptrval
            continue
        else
            continue # this should prevent "interning" of MMIO
        end
        # we better be sure this is actually a julia pointer here - it ought to be..
        # if it isn't we segfault here!
        ptr = Ptr{Any}(ptrval)
        @debug ptr
        jl_cfunc_idx = findfirst(jlfuncs) do arg
            symptr, _ = arg
            symptr == ptr
        end
        if !isnothing(jl_cfunc_idx)
            @warn """
                A direct call to the runtime was encountered.
                The resulting binary WILL NOT WORK as intended.
                """ Ptr=ptrval Func=jlfuncs[jl_cfunc_idx][end]
            continue
        end

        obj = unsafe_pointer_to_objref(ptr)
        @debug typeof(obj)
        if !is_object_moveable(obj)
            # ok, we have an object, but we can't move it - log the object & type
            # so the user knows that *something* is up and this won't work on a
            # real device (if we don't crash during compilation later).
            # The pointer will end up truncated in the IR, any may be eliminated
            # entirely in the resulting assembly..
            @warn """
                An object was encountered that could not be interned to the target platform.
                The resulting binary WILL NOT WORK as intended.
                """ Ptr=ptrval Type=typeof(obj) Object=obj
            continue
        end

        # we have an object, and it is internable! So let's intern it.
        @debug "Found internable object" Ptr=ptrval Type=typeof(obj) Object=obj
        g = build_global!(mod, obj)
        @debug "Created global variable in module" Global=g

        # Second, replace uses of the original pointer with a pointer to this global
        LLVM.replace_uses!(ptr_from_const, g)

        # We're done!
    end
end

function build_global!(mod, obj)
    ctx = context(mod)
    T = typeof(obj)
    llvm_types = LLVM.LLVMType[]
    llvm_objs = LLVM.Constant[]
    for (elt, elname) in zip(fieldtypes(T), fieldnames(T))
        llvm_t = llvm_type(ctx, elt)
        llvm_obj = llvm_constant(llvm_t, getfield(obj, elname))
        push!(llvm_types, llvm_t)
        push!(llvm_objs, llvm_obj)
    end

    structType = LLVM.StructType(llvm_types; ctx)
    get!(globalNamesCount, T, 0)
    num = (globalNamesCount[T] += 1)
    globalVar = LLVM.GlobalVariable(mod, structType, "$(T)_$num")
    constant!(globalVar, !ismutable(obj))
    initialValue = LLVM.ConstantStruct(structType, llvm_objs)
    initializer!(globalVar, initialValue)

    globalVar
end

function llvm_type(ctx, ::Type{T}) where T
    if isprimitivetype(T) || T <: Base.BitInteger
        LLVM.IntType(sizeof(T)*8; ctx)
    elseif T <: NTuple
        elt = llvm_type(first(t.parameters))
        len = length(t.parameters)
        LLVM.ArrayType(elt, len)
    elseif T <: Base.IEEEFloat
        if T === Float16
            LLVM.LLVMHalf
        elseif T === Float32
            LLVM.LLVMFloat
        elseif T === Float64
            LLVM.LLVMDouble
        end
    end
end

function LLVM.ConstantInt(typ::LLVM.IntegerType, val::T) where T
    isprimitivetype(T) || throw(ArgumentError("Cannot ConstantInt non-primitive type `$T`"))
    valbits = sizeof(T)*8
    if valbits >= 64
        numwords = cld(valbits, 64)
        words = Vector{Culonglong}(undef, numwords)
        for i in 1:numwords
            shint = 64*(i-1)
            sh = Core.Intrinsics.zext_int(T, shint)
            v = Core.Intrinsics.lshr_int(val, sh)
            words[i] = Core.Intrinsics.trunc_int(Culonglong, v)
        end
        return LLVM.ConstantInt(LLVM.API.LLVMConstIntOfArbitraryPrecision(typ, numwords, words))
    else
        v = Core.Intrinsics.zext_int(Int64, val)
        return LLVM.ConstantInt(typ, v)
    end
end

function llvm_constant(llvm_t, obj)
    T = typeof(obj)
    if isprimitivetype(T) || T <: Base.BitInteger
        LLVM.ConstantInt(llvm_t, obj)
    elseif T <: NTuple
        LLVM.ConstantArray(llvm_t, [ llvm_constant(llvm_type(first(t.parameters)),  o) for o in obj ])
    elseif T <: Base.IEEEFloat
        LLVM.ConstantFP(obj)
    else
        @warn "Unknown Type encountered" Type=T, LLVMType=llvm_t, Obj=obj
    end
end

#####
# Documented Interface
#####

"""
    build(mod::Module, platform::MCUTarget[, outpath::String]; clear=false, target=:build, optimize=true, validate=true)

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

`optimize` specifies whether the binary should be optimized. This will likely be required, to remove julia specific calls into the runtime that are unused.
`validate` specifies whether the code should be validated to work with some basic heuristics, as well as whether GPUCompiler.jl should run its internal validation logic.

A symlink `latest` pointing to the latest built binary will be created.
"""
build(mod::Module, platform::MCUTarget; clear=true, target=:build, strip=true, optimize=true, validate=true) = build(mod, joinpath(dirname(Pkg.project().path), "out/"), platform; clear, target, strip, optimize, validate)

function build(mod::Module, outpath, platform::MCUTarget; clear=false, target=:build, strip=true, optimize=true, validate=true)
    !isdirpath(outpath) && throw(ArgumentError("Given path `$path` does not describe a directory (doesn't end with a path seperator like `/`)!"))
    !hasproperty(mod, :main) && throw(ArgumentError("Module `$mod` doesn't have a `main` entrypoint!"))
    target in (:build, :llvm, :asm)  || throw(ArgumentError("Supplied unsupported target: `$target`"))
    any(m -> isone(m.nargs), methods(mod.main)) || throw(ArgumentError("`main` has no method taking zero arguments!"))
    buildpath = mktempdir()

    if validate
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
    end

    GPUCompiler.reset_runtime()

    compile_goal = target == :build ? :obj : target
    obj = GPUCompiler.JuliaContext() do ctx
        GPUCompiler.compile(compile_goal, mcu_job(mod.main, (), platform); strip, optimize, validate, libraries=true)[1]
    end
    if target != :build
        return obj
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
    build_vectors(platform, vectorasm_path, vectorobj_path)

    @info "Linking object files to ELF"
    mainelf_name = string(nameof(mod), ".elf")
    builtelf_name = joinpath(buildpath, mainelf_name)
    link(platform, builtelf_name, vectorobj_path, builtobj_path)

    @info "Postprocessing"
    postprocess(platform, buildpath)

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

end # module AVRCompiler
