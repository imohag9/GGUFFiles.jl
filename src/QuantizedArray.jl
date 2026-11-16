# A lazily dequantized array: GGML-encoded bytes that behave as an
# `AbstractArray{T}` and are decoded when used.

"""
    QuantizedArray{T}(t::GGUFTensor)
    QuantizedArray{T}(bytes, type::GGMLType, dims)

An array of element type `T` stored as GGML-encoded `bytes` (any
`AbstractVector{UInt8}`: a view into a memory-mapped file, an `Array`, or a GPU
array). It is decoded with [`dequantize`](@ref) when used, so a model can keep
its weights packed in (GPU) memory and decode one layer at a time.

- `adapt(CuArray, q)` (Adapt.jl) moves the bytes to the GPU; decoding then runs
  on the GPU too. Flux's `gpu`/`cpu` and `Functors.fmap` go through `adapt`.
- `q * x` and `transpose(q) * x` decode `q` and multiply. A PyTorch linear
  weight of shape `(out, in)` is stored in GGUF as `(in, out)`, so
  `Flux.Dense(transpose(q), bias)` is that linear layer.
- [`dequantize_columns`](@ref) decodes only some columns (embedding lookup).
- `similar(q, ...)` gives a dense array on the device of the bytes.
- Scalar indexing works on the CPU (decoding one value), for display and tests.
"""
struct QuantizedArray{T,N,A<:AbstractVector{UInt8}} <: AbstractArray{T,N}
    data::A
    type::GGMLType
    dims::NTuple{N,Int}
    function QuantizedArray{T,N,A}(data::A, type::GGMLType, dims::NTuple{N,Int}) where {T,N,A<:AbstractVector{UInt8}}
        can_dequantize(type) || error("GGUFFiles: dequantization of $type is not supported")
        nb = tensor_nbytes(type, dims)
        length(data) == nb || error("GGUFFiles: $(length(data)) bytes for a $type array of size $dims, expected $nb")
        return new{T,N,A}(data, type, dims)
    end
end

QuantizedArray{T}(data::A, type::GGMLType, dims) where {T,A<:AbstractVector{UInt8}} =
    QuantizedArray{T,length(dims),A}(data, type, Tuple(Int.(dims)))
QuantizedArray{T}(t::GGUFTensor) where {T} = QuantizedArray{T}(t.data, t.type, size(t))
QuantizedArray(t::GGUFTensor) = QuantizedArray{Float32}(t)

Base.size(q::QuantizedArray) = q.dims
Base.IndexStyle(::Type{<:QuantizedArray}) = IndexLinear()

function Base.getindex(q::QuantizedArray{T}, i::Int) where {T}
    @boundscheck checkbounds(q, i)
    blk, j = divrem(i - 1, block_size(q.type))
    return T(_deq(Val(Int(q.type)), q.data, blk * type_size(q.type), j))
end

Base.similar(q::QuantizedArray, ::Type{S}, dims::Dims) where {S} = similar(q.data, S, dims)

Base.show(io::IO, q::QuantizedArray{T}) where {T} =
    print(io, "QuantizedArray{", T, "}(", q.type, ", ", join(q.dims, "×"), ", ", typeof(q.data).name.name, ")")
Base.show(io::IO, ::MIME"text/plain", q::QuantizedArray) = show(io, q)

# A view (into a memory-mapped file) is copied to its own bytes first: Adapt
# moves a SubArray by moving its whole parent. File bytes are read sequentially.
_own(x::SubArray{UInt8,1}) = copy(x)
_own(x::FileBytes) = copy(x)
_own(x) = x
Adapt.adapt_structure(to, q::QuantizedArray{T,N}) where {T,N} =
    QuantizedArray{T}(Adapt.adapt(to, _own(q.data)), q.type, q.dims)

"""
    dequantize(q::QuantizedArray) -> dense array

Decode `q` into a dense array of `eltype(q)` on the device of its bytes.
"""
dequantize(q::QuantizedArray{T}) where {T} = dequantize!(similar(q.data, T, q.dims), q.data, q.type)

Base.Array(q::QuantizedArray{T,N}) where {T,N} = Array{T,N}(dequantize(q))
Base.collect(q::QuantizedArray) = Array(q)

# Separate vector and matrix methods: `AbstractVecOrMat` would be ambiguous
# with LinearAlgebra's own `*` methods.
const _QMatrix = QuantizedArray{<:Any,2}
const _QTransposed = Union{LinearAlgebra.Transpose{<:Any,<:QuantizedArray{<:Any,2}},
                           LinearAlgebra.Adjoint{<:Real,<:QuantizedArray{<:Real,2}}}

Base.:*(q::_QMatrix, x::AbstractVector) = dequantize(q) * x
Base.:*(q::_QMatrix, x::AbstractMatrix) = dequantize(q) * x
Base.:*(q::_QTransposed, x::AbstractVector) = transpose(dequantize(parent(q))) * x
Base.:*(q::_QTransposed, x::AbstractMatrix) = transpose(dequantize(parent(q))) * x

"""
    dequantize_columns(q::QuantizedArray{T,2}, cols) -> dense matrix

Decode only the columns `cols` of `q` (e.g. the rows of an embedding table for
some token ids), on the device of its bytes. `cols` can be any integer vector
on the CPU. Each column must be a whole number of blocks.
"""
function dequantize_columns(q::QuantizedArray{T,2}, cols::AbstractVector{<:Integer}) where {T}
    nrow = size(q, 1)
    nbc = tensor_nbytes(q.type, (nrow,))
    all(c -> 1 <= c <= size(q, 2), cols) || throw(BoundsError(q, (:, cols)))
    idx = Vector{Int}(undef, nbc * length(cols))
    for (k, c) in enumerate(cols)
        idx[(k-1)*nbc+1:k*nbc] .= (c - 1) * nbc .+ (1:nbc)
    end
    bytes = q.data[copyto!(similar(q.data, Int, length(idx)), idx)]
    return dequantize!(similar(q.data, T, (nrow, length(cols))), bytes, q.type)
end
