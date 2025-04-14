"""
    read_gguf_header(io::IO)

Reads the GGUF header from an IO stream.

# Arguments
- `io::IO`: The IO stream to read from.

# Returns
- `GGUFHeader`: The GGUF header.
"""
function read_gguf_header(io::IO)
    magic = read(io, UInt32)
    version = read(io, UInt32)
    tensor_count = read(io, UInt64)
    metadata_kv_count = read(io, UInt64)
    return GGUFHeader(magic, version, tensor_count, metadata_kv_count)
end

"""
    read_gguf_metadata_kv(io::IO)

Reads a metadata key-value pair from an IO stream.

# Arguments
- `io::IO`: The IO stream to read from.

# Returns
- `GGUFMetadataKV`: The metadata key-value pair.
"""
function read_gguf_metadata_kv(io::IO)
    key_length = read(io, UInt64)
    key = String(read(io, key_length))
    value_type = GGUFValueType(read(io, UInt32))
    value = read_gguf_value(io, value_type)
    return GGUFMetadataKV(key, value_type, value)
end

"""
    read_gguf_value(io::IO, value_type::GGUFValueType)

Reads a value from an IO stream based on its type.

# Arguments
- `io::IO`: The IO stream to read from.
- `value_type::GGUFValueType`: The type of the value to read.

# Returns
- `Any`: The value.
"""
function read_gguf_value(io::IO, value_type::GGUFValueType)
    data = nothing
    if value_type == GGUF_TYPE_UINT8
        data = read(io, UInt8)
    elseif value_type == GGUF_TYPE_INT8
        data = read(io, Int8)
    elseif value_type == GGUF_TYPE_UINT16
        data = read(io, UInt16)
    elseif value_type == GGUF_TYPE_INT16
        data = read(io, Int16)
    elseif value_type == GGUF_TYPE_UINT32
        data = read(io, UInt32)
    elseif value_type == GGUF_TYPE_INT32
        data = read(io, Int32)
    elseif value_type == GGUF_TYPE_FLOAT32
        data = read(io, Float32)
    elseif value_type == GGUF_TYPE_BOOL
        data = read(io, Bool)
    elseif value_type == GGUF_TYPE_STRING
        len = read(io, UInt64)
        data = String(read(io, len))
    elseif value_type == GGUF_TYPE_UINT64
        data = read(io, UInt64)
    elseif value_type == GGUF_TYPE_INT64
        data = read(io, Int64)
    elseif value_type == GGUF_TYPE_FLOAT64
        data = read(io, Float64)
    elseif value_type == GGUF_TYPE_ARRAY
        data = []
        begin
            arrtype = GGUFValueType(read(io, UInt32))

            len = Int(read(io, Int64))
            for i in 1:len
                push!(data,read_gguf_value(io, arrtype))
            end


        end
    else
        error("Unknown GGUF value type: ", value_type)
    end
    return data
end

"""
    read_gguf_tensor_info(io::IO)

Reads tensor information from an IO stream.

# Arguments
- `io::IO`: The IO stream to read from.

# Returns
- `GGUFTensorInfo`: The tensor information.
"""
function read_gguf_tensor_info(io::IO)
    name_length = read(io, UInt64)
    name = String(read(io, name_length))
    n_dimensions = read(io, UInt32)
    dimensions = Vector{UInt64}(undef, n_dimensions)
    for i in 1:n_dimensions
        dimensions[i] = read(io, UInt64)
    end
    type = GGMLType(read(io, UInt32))
    offset = read(io, UInt64)
    return GGUFTensorInfo(name, n_dimensions, dimensions, type, offset)
end


# Reads the entire GGUF file from an IO stream.

function read_gguf(io::IO)
    header = read_gguf_header(io)
    metadata_kv = Vector{GGUFMetadataKV}(undef, header.metadata_kv_count)
    for i in 1:header.metadata_kv_count
        metadata_kv[i] = read_gguf_metadata_kv(io)
    end
    tensor_info = Vector{GGUFTensorInfo}(undef, header.tensor_count)
    for i in 1:header.tensor_count
        tensor_info[i] = read_gguf_tensor_info(io)
    end
    return header, metadata_kv, tensor_info
end

"""
    parse_gguf(file_path::String)

Parses a GGUF file from a given file path.

# Arguments
- `file_path::String`: The path to the GGUF file.

# Returns
- `header::GGUFHeader`: The GGUF header.
- `metadata_kv::Vector{GGUFMetadataKV}`: The metadata key-value pairs.
- `tensor_info::Vector{GGUFTensorInfo}`: The tensor information.
"""
function parse_gguf(file_path::String)
    io = open(file_path, "r")
    header, metadata_kv, tensor_info = read_gguf(io)
    close(io)
    return header, metadata_kv, tensor_info
end

export parse_gguf