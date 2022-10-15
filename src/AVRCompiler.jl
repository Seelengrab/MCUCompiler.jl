module AVRCompiler

using GPUCompiler
using LLVM

#####
# Compiler Target
#####

struct Arduino <: GPUCompiler.AbstractCompilerTarget end

GPUCompiler.llvm_triple(::Arduino) = "avr-unknown-unkown"
GPUCompiler.runtime_slug(j::GPUCompiler.CompilerJob{Arduino}) = j.params.name

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
    @info "Creating compiler job for '$func($types)'"
    source = GPUCompiler.FunctionSpec(
                func, # our function
                Base.to_tuple_type(types), # its signature
                false, # whether this is a GPU kernel
                GPUCompiler.safe_name(repr(func))) # the name to use in the asm
    target = Arduino()
    job = GPUCompiler.CompilerJob(target, source, params)
end

function build_ir(job, @nospecialize(func), @nospecialize(types))
    @info "Bulding LLVM IR for '$func($types)'"
    mi, _ = GPUCompiler.emit_julia(job)
    ir, ir_meta = GPUCompiler.emit_llvm(
                    job, # our job
                    mi; # the method instance to compile
                    libraries=false, # whether this code uses libraries
                    deferred_codegen=false, # is there runtime codegen?
                    optimize=true, # do we want to optimize the llvm?
                    only_entry=false, # is this an entry point?
                    ctx=JuliaContext()) # the LLVM context to use
    return ir, ir_meta
end

function build_obj(@nospecialize(func), @nospecialize(types), params=ArduinoParams("unnamed"); kwargs...)
    job = native_job(func, types, params)
    @info "Compiling AVR ASM for '$func($types)'"
    ir, ir_meta = build_ir(job, func, types)
    obj, _ = GPUCompiler.emit_asm(
                job, # our job
                ir; # the IR we got
                strip=true, # should the binary be stripped of debug info?
                validate=true, # should the LLVM IR be validated?
                format=LLVM.API.LLVMObjectFile) # What format would we like to create?
    return obj
end

end # module AVRCompiler
