"""
    write_gguf_header(io::IO, header::GGUFHeader)

Writes the GGUF header to an IO stream.

# Arguments
- `io::IO`: The IO stream to write to.
- `header::GGUFHeader`: The GGUF header to write.
"""
function write_gguf_header(io::IO, header::GGUFHeader)
    write(io, header.magic)
    write(io, header.version)
    write(io, header.tensor_count)
    write(io, header.metadata_kv_count)
end

"""
    write_gguf_metadata_kv(io::IO, metadata_kv::GGUFMetadataKV)

Writes a metadata key-value pair to an IO stream.

# Arguments
- `io::IO`: The IO stream to write to.
- `metadata_kv::GGUFMetadataKV`: The metadata key-value pair to write.
"""
function write_gguf_metadata_kv(io::IO, metadata_kv::GGUFMetadataKV)
    write(io, UInt64(sizeof(metadata_kv.key)))
    write(io, metadata_kv.key)
    write(io, UInt32(metadata_kv.value_type))
    write_gguf_value(io, metadata_kv.value, metadata_kv.value_type)
end

"""
    write_gguf_value(io::IO, value::Any, value_type::GGUFValueType)

Writes a value to an IO stream based on its type.

# Arguments
- `io::IO`: The IO stream to write to.
- `value::Any`: The value to write.
- `value_type::GGUFValueType`: The type of the value.
"""
function write_gguf_value(io::IO, value::Any, value_type::GGUFValueType)
    if value_type == GGUF_TYPE_UINT8
        write(io, UInt8(value))
    elseif value_type == GGUF_TYPE_INT8
        write(io, Int8(value))
    elseif value_type == GGUF_TYPE_UINT16
        write(io, UInt16(value))
    elseif value_type == GGUF_TYPE_INT16
        write(io, Int16(value))
    elseif value_type == GGUF_TYPE_UINT32
        write(io, UInt32(value))
    elseif value_type == GGUF_TYPE_INT32
        write(io, Int32(value))
    elseif value_type == GGUF_TYPE_FLOAT32
        write(io, Float32(value))
    elseif value_type == GGUF_TYPE_BOOL
        write(io, Bool(value))
    elseif value_type == GGUF_TYPE_STRING
        write(io, UInt64(sizeof(value)))
        write(io, value)
    elseif value_type == GGUF_TYPE_ARRAY
        et = eltype(value) !== Any ? gguf_value_type(eltype(value)) :
             isempty(value) ? GGUF_TYPE_UINT8 : gguf_value_type(typeof(first(value)))
        write(io, UInt32(et))
        write(io, UInt64(length(value)))
        for v in value
            write_gguf_value(io, v, et)
        end
    elseif value_type == GGUF_TYPE_UINT64
        write(io, UInt64(value))
    elseif value_type == GGUF_TYPE_INT64
        write(io, Int64(value))
    elseif value_type == GGUF_TYPE_FLOAT64
        write(io, Float64(value))
    else
        error("Unknown GGUF value type: ", value_type)
    end
end

"""
    write_gguf_tensor_info(io::IO, tensor_info::GGUFTensorInfo)

Writes tensor information to an IO stream.

# Arguments
- `io::IO`: The IO stream to write to.
- `tensor_info::GGUFTensorInfo`: The tensor information to write.
"""
function write_gguf_tensor_info(io::IO, tensor_info::GGUFTensorInfo)
    write(io, UInt64(sizeof(tensor_info.name)))
    write(io, tensor_info.name)
    write(io, UInt32(tensor_info.n_dimensions))
    for dimension in tensor_info.dimensions
        write(io, dimension)
    end
    write(io, UInt32(tensor_info.type))
    write(io, tensor_info.offset)
end

# Writes the entire GGUF file to an IO stream.
function write_gguf(io::IO, header::GGUFHeader, metadata_kv::Vector{GGUFMetadataKV}, tensor_info::Vector{GGUFTensorInfo})
    write_gguf_header(io, header)
    for kv in metadata_kv
        write_gguf_metadata_kv(io, kv)
    end
    for ti in tensor_info
        write_gguf_tensor_info(io, ti)
    end
end

"""
    create_gguf(file_path::String, header::GGUFHeader, metadata_kv::Vector{GGUFMetadataKV}, tensor_info::Vector{GGUFTensorInfo})

Creates a GGUF file at the specified file path.

# Arguments
- `file_path::String`: The path to the GGUF file.
- `header::GGUFHeader`: The GGUF header.
- `metadata_kv::Vector{GGUFMetadataKV}`: The metadata key-value pairs.
- `tensor_info::Vector{GGUFTensorInfo}`: The tensor information.
"""
function create_gguf(file_path::String, header::GGUFHeader, metadata_kv::Vector{GGUFMetadataKV}, tensor_info::Vector{GGUFTensorInfo})
    io = open(file_path, "w")
    write_gguf(io, header, metadata_kv, tensor_info)
    close(io)
end
export create_gguf, write_gguf_header, write_gguf_metadata_kv, write_gguf_value, write_gguf_tensor_info

"""
    gguf_value_type(T::Type) -> GGUFValueType

GGUF metadata value type used to store a Julia value of type `T`.
"""
gguf_value_type(::Type{UInt8}) = GGUF_TYPE_UINT8
gguf_value_type(::Type{Int8}) = GGUF_TYPE_INT8
gguf_value_type(::Type{UInt16}) = GGUF_TYPE_UINT16
gguf_value_type(::Type{Int16}) = GGUF_TYPE_INT16
gguf_value_type(::Type{UInt32}) = GGUF_TYPE_UINT32
gguf_value_type(::Type{Int32}) = GGUF_TYPE_INT32
gguf_value_type(::Type{Float32}) = GGUF_TYPE_FLOAT32
gguf_value_type(::Type{Bool}) = GGUF_TYPE_BOOL
gguf_value_type(::Type{<:AbstractString}) = GGUF_TYPE_STRING
gguf_value_type(::Type{UInt64}) = GGUF_TYPE_UINT64
gguf_value_type(::Type{Int64}) = GGUF_TYPE_INT64
gguf_value_type(::Type{Float64}) = GGUF_TYPE_FLOAT64
gguf_value_type(::Type{<:AbstractVector}) = GGUF_TYPE_ARRAY

"""
    write_gguf(path, metadata::AbstractDict, tensors; alignment=32)

Write a complete GGUF file, tensor data included. `metadata` maps keys to
values (numbers, strings, or vectors of them; the value type follows the
Julia type). `tensors` is a collection of [`GGUFTensor`](@ref)s, or of
`(name, type, dims, bytes)` tuples, whose bytes are already encoded in `type`.
"""
function write_gguf(path::AbstractString, metadata::AbstractDict, tensors; alignment::Integer=GGUF_DEFAULT_ALIGNMENT)
    ts = [t isa GGUFTensor ? (t.name, t.type, t.dims, t.data) : t for t in tensors]
    md = Dict{String,Any}(metadata)
    alignment != GGUF_DEFAULT_ALIGNMENT && (md["general.alignment"] = UInt32(alignment))
    infos = GGUFTensorInfo[]
    offset = 0
    for (name, type, dims, bytes) in ts
        nb = tensor_nbytes(type, dims)
        length(bytes) == nb || error("GGUFFiles: tensor $name has $(length(bytes)) bytes, expected $nb")
        push!(infos, GGUFTensorInfo(name, length(dims), UInt64.(collect(dims)), type, offset))
        offset = cld(offset + nb, alignment) * alignment
    end
    kvs = [GGUFMetadataKV(k, gguf_value_type(typeof(v)), v) for (k, v) in sort!(collect(md); by=first)]
    open(path, "w") do io
        write_gguf(io, GGUFHeader(GGUF_MAGIC, GGUF_VERSION, length(infos), length(kvs)), kvs, infos)
        _pad!(io, alignment)
        start = position(io)
        for ((_, _, _, bytes), info) in zip(ts, infos)
            write(io, zeros(UInt8, start + Int(info.offset) - position(io)))
            write(io, collect(UInt8, bytes))
        end
        _pad!(io, alignment)
    end
    return path
end

_pad!(io, alignment) = write(io, zeros(UInt8, cld(position(io), alignment) * alignment - position(io)))

export write_gguf, gguf_value_type
