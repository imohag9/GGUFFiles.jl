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
    write(io, UInt64(length(metadata_kv.key)))
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
        write(io, UInt64(length(value)))
        write(io, value)
    elseif value_type == GGUF_TYPE_ARRAY
        write(io, UInt32(typeof(value[1])))
        write(io, UInt64(length(value)))
        for v in value
            write_gguf_value(io, v, GGUFValueType(typeof(v)))
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
    write(io, UInt64(length(tensor_info.name)))
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