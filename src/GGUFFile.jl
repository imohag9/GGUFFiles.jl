# A GGUF file opened for reading tensor data.

"""
    FileBytes <: AbstractVector{UInt8}

The bytes of one tensor of a memory-mapped [`GGUFFile`](@ref). Indexing goes
through the memory map; `copy`/`collect` (and so moving the tensor to a GPU
with Adapt) read the whole region with one sequential read instead, which is
much faster than page faults on network or WSL-mounted (drvfs) drives.
"""
struct FileBytes <: AbstractVector{UInt8}
    path::String
    offset::Int                    # 0-based position in the file
    view::SubArray{UInt8,1,Vector{UInt8},Tuple{UnitRange{Int}},true}
end
Base.size(b::FileBytes) = size(b.view)
Base.IndexStyle(::Type{FileBytes}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(b::FileBytes, i::Int) = b.view[i]
function Base.copy(b::FileBytes)
    out = Vector{UInt8}(undef, length(b))
    open(io -> (seek(io, b.offset); read!(io, out)), b.path, "r")
    return out
end
Base.collect(b::FileBytes) = copy(b)
Base.Vector(b::FileBytes) = copy(b)
Base.similar(b::FileBytes, ::Type{S}, dims::Dims) where {S} = Array{S}(undef, dims)

# The array a kernel reads the bytes from.
_kernel_bytes(b::FileBytes) = b.view
_kernel_bytes(b) = b

"""
    GGUFTensor

One tensor of a [`GGUFFile`](@ref): its name, GGML type, dimensions and raw
bytes (a view into the file's memory map).

`dims` is in GGUF order, which is also Julia's: the first dimension varies
fastest. A tensor written by PyTorch with shape `(out, in)` has
`dims == (in, out)`, so `dequantize(t)` returns the transpose of the PyTorch
matrix, which is the same memory.
"""
struct GGUFTensor{A<:AbstractVector{UInt8}}
    name::String
    type::GGMLType
    dims::Vector{Int}
    data::A
end

Base.size(t::GGUFTensor) = Tuple(t.dims)
Base.size(t::GGUFTensor, d::Integer) = d <= length(t.dims) ? t.dims[d] : 1
Base.length(t::GGUFTensor) = prod(t.dims; init=1)
Base.ndims(t::GGUFTensor) = length(t.dims)

Base.show(io::IO, t::GGUFTensor) =
    print(io, "GGUFTensor(\"", t.name, "\", ", t.type, ", ", join(t.dims, "×"), ")")

"""
    GGUFFile(path; mmap=true)

Open a GGUF file: reads its header, metadata and tensor table, and maps the
tensor data into memory (with `mmap=false`, reads the whole file instead).

- `f.metadata` is a `Dict{String,Any}`; arrays are typed vectors
  (`Vector{String}`, `Vector{Int32}`, ...).
- `keys(f)` lists the tensor names in file order; `f[name]` is a
  [`GGUFTensor`](@ref); `haskey(f, name)` tests for one.
- [`dequantize`](@ref) decodes a tensor to a dense array.
"""
struct GGUFFile
    path::String
    header::GGUFHeader
    metadata::Dict{String,Any}
    tensors::Vector{GGUFTensorInfo}
    index::Dict{String,Int}
    data_offset::Int
    bytes::Vector{UInt8}
    mmap::Bool
end

const GGUF_DEFAULT_ALIGNMENT = 32

function GGUFFile(path::AbstractString; mmap::Bool=true)
    header, kvs, infos, pos = open(path, "r") do io
        h, kv, ti = read_gguf(io)
        h, kv, ti, position(io)
    end
    header.magic == GGUF_MAGIC || error("GGUFFiles: $path is not a GGUF file")
    metadata = Dict{String,Any}(kv.key => kv.value for kv in kvs)
    align = Int(get(metadata, "general.alignment", GGUF_DEFAULT_ALIGNMENT))
    data_offset = cld(pos, align) * align
    bytes = mmap ? open(io -> Mmap.mmap(io, Vector{UInt8}), path, "r") : read(path)
    index = Dict(ti.name => i for (i, ti) in enumerate(infos))
    return GGUFFile(String(path), header, metadata, infos, index, data_offset, bytes, mmap)
end

Base.keys(f::GGUFFile) = [ti.name for ti in f.tensors]
Base.haskey(f::GGUFFile, name::AbstractString) = haskey(f.index, name)
Base.length(f::GGUFFile) = length(f.tensors)

function Base.getindex(f::GGUFFile, name::AbstractString)
    i = get(f.index, name, 0)
    i == 0 && throw(KeyError(name))
    ti = f.tensors[i]
    dims = Int.(ti.dimensions)
    start = f.data_offset + Int(ti.offset)
    nb = tensor_nbytes(ti.type, dims)
    start + nb <= length(f.bytes) || error("GGUFFiles: tensor $name runs past the end of the file")
    v = view(f.bytes, start+1:start+nb)
    return GGUFTensor(ti.name, ti.type, dims, f.mmap ? FileBytes(f.path, start, v) : v)
end

Base.iterate(f::GGUFFile, i=1) = i > length(f.tensors) ? nothing : (f[f.tensors[i].name], i + 1)

Base.show(io::IO, f::GGUFFile) =
    print(io, "GGUFFile(\"", basename(f.path), "\", ", length(f.tensors), " tensors, ",
          length(f.metadata), " metadata keys)")

"""
    dequantize(t::GGUFTensor, T=Float32) -> Array{T}

Decode a tensor to a dense `Array{T}` of size `size(t)`, on the CPU. To decode
on a GPU, copy `t.data` to the device and call [`dequantize!`](@ref) with a
device output array.
"""
dequantize(t::GGUFTensor, ::Type{T}=Float32) where {T} =
    dequantize!(Array{T}(undef, size(t)...), t.data, t.type)
