function _new_array(atype, ndims::UInt32, dims::Ptr{Csize_t})
    eltype = atype.parameters[1]::Type
    elsz = Base.RefValue{Csize_t}(0)
    al = Base.RefValue{Csize_t}(0)
    isunboxed = islayout_inline(atype, elsz, al) != 0 # ccall(:jl_islayout_inline, Cint, (Any, Ptr{Csize_t}, Ptr{Csize_t}), atype, sz, algn) != 0
    isunion = eltype isa Union
    hasptr = isunboxed && 
             (eltype isa DataType) &&
             !Base.datatype_pointerfree(eltype)
    if !isunboxed
        elsz[] = sizeof(Ptr{Cvoid})
        al[] = elsz[]
    else
        elsz[] = Base.LLT_ALIGN(elsz[], al[])
    end
    zi = !isunboxed || hasptr || isunion || (eltype isa DataType) && is_zeroinit(eltype)
    _new_array_(atype, ndims, dims, isunboxed, hasptr, isunion, zi, elsz[])
end
is_zeroinit(T::DataType) = !iszero(T.flags & 0x10)

function islayout_inline(T::Type, sz::Ref{Csize_t}, algn::Ref{Csize_t})
    countbits = union_isinlinable(T, false, sz, algn, true)
    return 0x0 < countbits < 0x7f ? countbits : 0
end

function union_isinlinable(T::Type, pointerfree::Bool, nbytes::Ref{Csize_t}, align::Ref{Csize_t}, asfield::Bool)
    if T isa Union
        na = union_isinlinable(T.a, true, nbytes, align, asfield)
        iszero(na) && return 0
        nb = union_isinlinable(T.b, true, nbytes, align, asfield)
        iszero(nb) && return 0
        return na + nb
    end
    if T isa DataType && datatype_isinlinealloc(T, pointerfree)
        sz = Core.sizeof(T)
        al = Base.datatype_alignment(T)
        if asfield && isprimitivetype(T)
            sz = Base.LLT_ALIGN(sz, al)
        end
        if nbytes[] < sz
            nbytes[] = sz
        end
        if align[] < al
            align[] = al
        end
        return 1
    end
    return 0
end

mayinlinealloc(T::DataType) = !iszero(T.name.flags & 0x4) 
function struct_try_layout(T::DataType)
    if isdefined(T, :layout)
        return true
    elseif !has_fixed_layout(T)
        return false
    end
    # compute_field_offsets?
    return true
end
function has_fixed_layout(T::DataType)
    if isconcretetype(T)
        return true
    elseif isabstracttype(T)
        return false
    elseif T isa NamedTuple
        # TODO: replace with actual implementation on free typevars
        return false
    elseif T isa Tuple
        return false
    end
    # TODO: insert checking actual field types with free typevars
    return true
end
function fielddesc_type(T::DataType)
    Base.@_foldable_meta
    # just assume the type has a layout xD
    # T.layout == C_NULL && throw(UndefRefError())
    flags = unsafe_load(convert(Ptr{Base.DataTypeLayout}, T.layout)).flags
    return Int((flags >> 1) & 0x3)
end
function datatype_isinlinealloc(T::DataType, pointerfree::Bool)
    if mayinlinealloc(T) && struct_try_layout(T)
        if T.layout != C_NULL
            if pointerfree
                return false
            end
            if T.name.n_uninitialized != 0
                return false
            end
            if fielddesc_type(T) > 1
                return false
            end
        end
        return true
    end
    return false
end

using FieldFlags

@bitfield struct jl_array_flags_t
    how       : 2
    ndims     : 9
    pooled    : 1
    ptrarray  : 1
    hasptr    : 1
    isshared  : 1
    isaligned : 1
end

mutable struct jl_array_t
    data::Ptr{Cvoid}
    length::Csize_t
    flags::jl_array_flags_t
    elsize::UInt16
    offset::UInt32
    nrows::Csize_t
    maxsize::Csize_t
    # other dim sizes go here for ndims > 2
    # followed by alignment padding and inline data, or owner pointer
end

const ARRAY_INLINE_NBYTES = 2048*sizeof(Ptr{Cvoid})
const ARRAY_CACHE_ALIGN_THRESHOLD = 2048
const JL_SMALL_BYTE_ALIGNMENT = 16
const JL_CACHE_BYTE_ALIGNMENT = 64

const JL_ARRAY_IMPL_NUL = true
const GC_MAX_SZCLASS = 2032 - sizeof(Ptr{Cvoid})

function alloc(sz)
    Base.llvmcall(
        ("""declare i8* @gpu_gc_pool_alloc(i64)

        define i64 @entry(i64 %sz) #0 {
            %a = call i8* @gpu_gc_pool_alloc(i64 %sz)
            %r = ptrtoint i8* %a to i64
            ret i64 %r
        }

        attributes #0 = { alwaysinline }
        """, "entry"),
        Ptr{Cchar},
        Tuple{Csize_t},
        sz
    )    
end

function _new_array_(atype, ndims::UInt32, dims::Ptr{Csize_t},
        isunboxed::Bool, hasptr::Bool, isunion::Bool,
        zeroinit::Bool, elsz::Csize_t)
    # data::Ptr{Cvoid}
    # @assert isunboxed || elsz == sizeof(Ptr{Cvoid})
    # @assert atype == C_NULL || isunion == (atype.parameters[1] isa Union)
    tot = Ref{Csize_t}(0)
    nel = Ref{Csize_t}(0)
    validated = array_validate_dims(nel, tot, ndims, dims, elsz)
    if validated == 1
        # invalid array dimensions
    elseif validated == 2
        # invalid array size
    end
    if isunboxed
        if elsz == 1 && !isunion
            # extra byte for all julia allocated byte arrays
            tot[] += 1
        end
        if isunion
            # an extra byte for each isbits union array element, stored after a-> maxsize
            tot[] += nel[]
        end
    end

    ndimwords = array_ndimwords(ndims)
    tsz = (sizeof(jl_array_t) + ndimwords*sizeof(Csize_t)) % UInt64
    if tot[] <= ARRAY_INLINE_NBYTES
        if tot[] >= ARRAY_CACHE_ALIGN_THRESHOLD
            tsz = Base.LLT_ALIGN(tsz, JL_CACHE_BYTE_ALIGNMENT)
        elseif isunboxed && elsz >= 4
            tsz = Base.LLT_ALIGN(tsz, JL_SMALL_BYTE_ALIGNMENT)
        end
        doffs = tsz
        tsz += tot[]
        # jl_array_t is large enogh that objects will always be aligned 16
        # NOTE: this should in theory care about the PTLS, but in reality, *I* don't
        a_ptr = convert(Ptr{jl_array_t}, alloc(tsz % Csize_t))
        a = unsafe_pointer_to_objref(a_ptr)::jl_array_t
        flags_how = 0x0
        data = convert(Ptr{Cvoid}, a_ptr) + doffs
    else
        data = alloc(tot[] % Csize_t)
        a_ptr = convert(Ptr{jl_array_t}, alloc(tsz % Csize_t))
        a = unsafe_pointer_to_objref(a_ptr)::jl_array_t
        flags_how = 0x2
    end

    flags_pooled = tsz <= GC_MAX_SZCLASS

    if zeroinit
        Base.memset(data, 0, tot[])
    end

    a.data = data
    if JL_ARRAY_IMPL_NUL && elsz == 1
        unsafe_store!(convert(Ptr{Cchar}, data), tot[]-0x1, 0x0 % Cchar)
    end
    a.length = nel[]
    flags_ndims = ndims
    flags_ptrarray = !isunboxed
    flags_hasptr = hasptr
    a.elsize = elsz % UInt16
    flags_isshared = false
    flags_isaligned = true
    a.offset = 0
    if isone(ndims)
        a.nrows = nel[]
        a.maxsize = nel[]
    elseif flags_ndims != ndims
        # invalid array dimensions
    else
        adims = convert(Ptr{Csize_t}, a_ptr) + 4*sizeof(Csize_t)
        for i in 1:ndims
            dims_data = unsafe_load(dims, i)
            unsafe_store!(adims, dims_data, i)
        end
    end

    a.flags = jl_array_flags_t(flags_how, flags_ndims, flags_pooled, flags_ptrarray, flags_hasptr, flags_isshared, flags_isaligned)
    return a_ptr
end

function array_validate_dims(nel, tot, ndims::UInt32, dims::Ptr{Csize_t}, elsz::Csize_t)
    _nel = Int128(1)
    for i in 1:ndims
        di = unsafe_load(dims, i)
        prod = Int128(_nel)*Int128(di)
        if prod >= typemax(Csize_t) || di >= typemax(Csize_t)
            return 0x1
        end
        _nel = prod
    end
    prod = Int128(elsz)*Int128(_nel)
    if prod >= typemax(Csize_t)
        return 0x2
    end
    nel[] = unsafe_trunc(UInt64, _nel)
    tot[] = unsafe_trunc(UInt64, prod)
    return 0x0
end

function array_ndimwords(ndims::UInt32)
    ndims < 3 ? (0 % Cint) : ((ndims - 0x2) % Cint)
end