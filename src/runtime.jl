"""
    baseaddress() -> UInt

The base pointer for `malloc`. 
"""
baseaddress

module MCURuntime
    # the runtime library
    signal_exception() = return
    function malloc(sz)
        stored_ptr = Ptr{Ptr{Nothing}}(baseaddress() |> Int)
        base = unsafe_load(stored_ptr)
        nbase = base - sz
        unsafe_store!(stored_ptr, nbase)
        return nbase
    end
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

GPUCompiler.runtime_module(::GPUCompiler.CompilerJob{<:MCUTarget}) = MCURuntime
GPUCompiler.uses_julia_runtime(::GPUCompiler.CompilerJob{<:MCUTarget}) = true
GPUCompiler.can_throw(::GPUCompiler.CompilerJob{<:MCUTarget}) = false
