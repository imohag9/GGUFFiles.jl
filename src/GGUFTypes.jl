const GGUF_MAGIC = 0x47475546

# GGUF Version
const GGUF_VERSION = 3
# GGUF Metadata Value Types
@enum GGUFValueType::UInt32 begin
    GGUF_TYPE_UINT8   = 0
    GGUF_TYPE_INT8    = 1
    GGUF_TYPE_UINT16  = 2
    GGUF_TYPE_INT16   = 3
    GGUF_TYPE_UINT32  = 4
    GGUF_TYPE_INT32   = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL    = 7
    GGUF_TYPE_STRING  = 8
    GGUF_TYPE_ARRAY   = 9
    GGUF_TYPE_UINT64  = 10
    GGUF_TYPE_INT64   = 11
    GGUF_TYPE_FLOAT64 = 12
end

# GGML Types
@enum GGMLType::UInt32 begin
    GGML_TYPE_F32     = 0
    GGML_TYPE_F16     = 1
    GGML_TYPE_Q4_0    = 2
    GGML_TYPE_Q4_1    = 3
    GGML_TYPE_Q5_0    = 6
    GGML_TYPE_Q5_1    = 7
    GGML_TYPE_Q8_0    = 8
    GGML_TYPE_Q8_1    = 9
    GGML_TYPE_Q2_K    = 10
    GGML_TYPE_Q3_K    = 11
    GGML_TYPE_Q4_K    = 12
    GGML_TYPE_Q5_K    = 13
    GGML_TYPE_Q6_K    = 14
    GGML_TYPE_Q8_K    = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_IQ2_XS  = 17
    GGML_TYPE_IQ3_XXS = 18
    GGML_TYPE_IQ1_S   = 19
    GGML_TYPE_IQ4_NL  = 20
    GGML_TYPE_IQ3_S   = 21
    GGML_TYPE_IQ2_S   = 22
    GGML_TYPE_IQ4_XS  = 23
    GGML_TYPE_I8      = 24
    GGML_TYPE_I16     = 25
    GGML_TYPE_I32     = 26
    GGML_TYPE_I64     = 27
    GGML_TYPE_F64     = 28
    GGML_TYPE_IQ1_M   = 29
    GGML_TYPE_BF16    = 30
end

struct GGUFHeader
    magic::UInt32
    version::UInt32
    tensor_count::UInt64
    metadata_kv_count::UInt64
end

struct GGUFMetadataKV
    key::String
    value_type::GGUFValueType
    value::Any
end

struct GGUFTensorInfo
    name::String
    n_dimensions::UInt32
    dimensions::Vector{UInt64}
    type::GGMLType
    offset::UInt64
end

Base.show(io::IO, ::MIME"text/plain", tensor_info::GGUFTensorInfo) = begin
    print(io, "GGUFTensorInfo(")
    print(io, "name = \"", tensor_info.name, "\", ")
    print(io, "n_dimensions = ", tensor_info.n_dimensions, ", ")
    print(io, "dimensions = ", tensor_info.dimensions, ", ")
    print(io, "type = ", tensor_info.type, ", ")
    print(io, "offset = ", tensor_info.offset)
    print(io, ")")
end

Base.show(io::IO, ::MIME"text/plain", metadata_kv::GGUFMetadataKV) = begin
    print(io, "GGUFMetadataKV(")
    print(io, "key = \"", metadata_kv.key, "\", ")
    print(io, "value_type = ", metadata_kv.value_type, ", ")
    print(io, "value = ", metadata_kv.value)
    print(io, ")")
end
export GGUF_MAGIC,GGUF_VERSION
export GGUFHeader,GGUFMetadataKV,GGUFTensorInfo,GGUFValueType,GGMLType