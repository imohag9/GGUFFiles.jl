# Block sizes and dequantization of GGML tensor types.
#
# Every type is stored as a sequence of blocks; each block holds `block_size`
# values in `type_size` bytes. Dequantization runs one work item per output
# value through KernelAbstractions, so the same code runs on the CPU (on
# threads) and on GPU arrays (pass a `CuArray{UInt8}` of the tensor bytes and a
# `CuArray` output).
#
# The per-value decoders follow ggml's `dequantize_row_*` functions
# (ggml/src/ggml-quants.c).

"""
    block_size(type::GGMLType) -> Int

Number of values stored in one block of `type`.
"""
function block_size(t::GGMLType)
    t in (GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F64,
          GGML_TYPE_I8, GGML_TYPE_I16, GGML_TYPE_I32, GGML_TYPE_I64) && return 1
    t in (GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
          GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_IQ4_NL) && return 32
    return 256
end

const _TYPE_SIZES = Dict{GGMLType,Int}(
    GGML_TYPE_F32 => 4, GGML_TYPE_F16 => 2, GGML_TYPE_BF16 => 2, GGML_TYPE_F64 => 8,
    GGML_TYPE_I8 => 1, GGML_TYPE_I16 => 2, GGML_TYPE_I32 => 4, GGML_TYPE_I64 => 8,
    GGML_TYPE_Q4_0 => 18, GGML_TYPE_Q4_1 => 20, GGML_TYPE_Q5_0 => 22, GGML_TYPE_Q5_1 => 24,
    GGML_TYPE_Q8_0 => 34, GGML_TYPE_Q8_1 => 36,
    GGML_TYPE_Q2_K => 84, GGML_TYPE_Q3_K => 110, GGML_TYPE_Q4_K => 144,
    GGML_TYPE_Q5_K => 176, GGML_TYPE_Q6_K => 210, GGML_TYPE_Q8_K => 292,
    GGML_TYPE_IQ4_NL => 18, GGML_TYPE_IQ4_XS => 136,
)

"""
    type_size(type::GGMLType) -> Int

Number of bytes in one block of `type`.
"""
type_size(t::GGMLType) = get(() -> error("GGUFFiles: unknown size for $t"), _TYPE_SIZES, t)

"""
    tensor_nbytes(type::GGMLType, dims) -> Int

Number of bytes taken by a tensor of `type` with dimensions `dims`.
"""
function tensor_nbytes(t::GGMLType, dims)
    n = prod(Int, dims; init=1)
    bs = block_size(t)
    n % bs == 0 || error("GGUFFiles: $n values is not a whole number of $t blocks of $bs")
    return (n ÷ bs) * type_size(t)
end

"""
    is_quantized(type::GGMLType) -> Bool

Whether `type` is a block-quantized type (anything but plain floats and integers).
"""
is_quantized(t::GGMLType) = block_size(t) > 1

const DEQUANT_TYPES = (GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
                       GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
                       GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
                       GGML_TYPE_Q6_K)

"""
    can_dequantize(type::GGMLType) -> Bool

Whether [`dequantize!`](@ref) supports `type`.
"""
can_dequantize(t::GGMLType) = t in DEQUANT_TYPES

# --- reading little-endian scalars from a byte array (0-based offset `o`) ---

@inline _u8(b, o) = @inbounds b[o + 1]
@inline _u16(b, o) = UInt16(_u8(b, o)) | (UInt16(_u8(b, o + 1)) << 8)
@inline _u32(b, o) = UInt32(_u16(b, o)) | (UInt32(_u16(b, o + 2)) << 16)
@inline _f16(b, o) = Float32(reinterpret(Float16, _u16(b, o)))
@inline _bf16(b, o) = reinterpret(Float32, UInt32(_u16(b, o)) << 16)
@inline _f32(b, o) = reinterpret(Float32, _u32(b, o))
@inline _i8(b, o) = reinterpret(Int8, _u8(b, o))

# K-quant 6-bit scale and min `j` (0..7) from the 12 packed bytes at `o`.
@inline function _scale_min_k4(b, o, j)
    if j < 4
        sc = _u8(b, o + j) & 0x3f
        m = _u8(b, o + j + 4) & 0x3f
    else
        sc = (_u8(b, o + j + 4) & 0x0f) | ((_u8(b, o + j - 4) >> 6) << 4)
        m = (_u8(b, o + j + 4) >> 4) | ((_u8(b, o + j) >> 6) << 4)
    end
    return Float32(sc), Float32(m)
end

# Value `j` (0-based, within the block) of the block starting at byte `o`.
# Dispatch is on the integer code of the type, not the enum: kernel compilers
# print type parameters (GPUCompiler's name mangling), and printing an enum
# from a kernel compilation running in an older world fails.

@inline _deq(::Val{Int(GGML_TYPE_F32)}, b, o, j) = _f32(b, o)
@inline _deq(::Val{Int(GGML_TYPE_F16)}, b, o, j) = _f16(b, o)
@inline _deq(::Val{Int(GGML_TYPE_BF16)}, b, o, j) = _bf16(b, o)

# Q8_0: d::f16, qs::int8[32]
@inline _deq(::Val{Int(GGML_TYPE_Q8_0)}, b, o, j) = _f16(b, o) * Float32(_i8(b, o + 2 + j))

# Q4_0: d::f16, qs::u8[16]; values 0..15 low nibbles, 16..31 high nibbles
@inline function _deq(::Val{Int(GGML_TYPE_Q4_0)}, b, o, j)
    q = _u8(b, o + 2 + (j & 15))
    q = j < 16 ? q & 0x0f : q >> 4
    return _f16(b, o) * Float32(Int(q) - 8)
end

# Q4_1: d::f16, m::f16, qs::u8[16]
@inline function _deq(::Val{Int(GGML_TYPE_Q4_1)}, b, o, j)
    q = _u8(b, o + 4 + (j & 15))
    q = j < 16 ? q & 0x0f : q >> 4
    return _f16(b, o) * Float32(q) + _f16(b, o + 2)
end

# Q5_0: d::f16, qh::u32, qs::u8[16]
@inline function _deq(::Val{Int(GGML_TYPE_Q5_0)}, b, o, j)
    q = _u8(b, o + 6 + (j & 15))
    q = j < 16 ? q & 0x0f : q >> 4
    h = UInt8((_u32(b, o + 2) >> j) & 0x1)
    return _f16(b, o) * Float32(Int(q | (h << 4)) - 16)
end

# Q5_1: d::f16, m::f16, qh::u32, qs::u8[16]
@inline function _deq(::Val{Int(GGML_TYPE_Q5_1)}, b, o, j)
    q = _u8(b, o + 8 + (j & 15))
    q = j < 16 ? q & 0x0f : q >> 4
    h = UInt8((_u32(b, o + 4) >> j) & 0x1)
    return _f16(b, o) * Float32(q | (h << 4)) + _f16(b, o + 2)
end

# Q2_K: scales::u8[16], qs::u8[64], d::f16, dmin::f16
@inline function _deq(::Val{Int(GGML_TYPE_Q2_K)}, b, o, j)
    h, r = divrem(j, 128)
    s, r2 = divrem(r, 32)
    half, l = divrem(r2, 16)
    sc = _u8(b, o + 8h + 2s + half)
    q = (_u8(b, o + 16 + 32h + 16half + l) >> (2s)) & 0x03
    d = _f16(b, o + 80)
    dmin = _f16(b, o + 82)
    return d * Float32(sc & 0x0f) * Float32(q) - dmin * Float32(sc >> 4)
end

# Q4_K: d::f16, dmin::f16, scales::u8[12], qs::u8[128]
@inline function _deq(::Val{Int(GGML_TYPE_Q4_K)}, b, o, j)
    c, r = divrem(j, 64)
    hi, l = divrem(r, 32)
    sc, m = _scale_min_k4(b, o + 4, 2c + hi)
    q = _u8(b, o + 16 + 32c + l)
    q = hi == 0 ? q & 0x0f : q >> 4
    return _f16(b, o) * sc * Float32(q) - _f16(b, o + 2) * m
end

# Q5_K: d::f16, dmin::f16, scales::u8[12], qh::u8[32], qs::u8[128]
@inline function _deq(::Val{Int(GGML_TYPE_Q5_K)}, b, o, j)
    c, r = divrem(j, 64)
    hi, l = divrem(r, 32)
    sc, m = _scale_min_k4(b, o + 4, 2c + hi)
    q = _u8(b, o + 48 + 32c + l)
    q = hi == 0 ? q & 0x0f : q >> 4
    h = (_u8(b, o + 16 + l) >> (2c + hi)) & 0x01
    return _f16(b, o) * sc * Float32(q + (h << 4)) - _f16(b, o + 2) * m
end

# Q6_K: ql::u8[128], qh::u8[64], scales::i8[16], d::f16
@inline function _deq(::Val{Int(GGML_TYPE_Q6_K)}, b, o, j)
    h, r = divrem(j, 128)
    g, l = divrem(r, 32)
    ql = _u8(b, o + 64h + l + 32(g & 1))
    ql = g < 2 ? ql & 0x0f : ql >> 4
    qh = (_u8(b, o + 128 + 32h + l) >> (2g)) & 0x03
    q = Int(ql | (qh << 4)) - 32
    sc = _i8(b, o + 192 + 8h + (l >> 4) + 2g)
    return _f16(b, o + 208) * Float32(sc) * Float32(q)
end

# Block and byte sizes are type parameters, so that the division below is by a
# constant (a shift for the power-of-two block sizes) rather than a 64-bit
# integer division, which is slow on GPUs.
@kernel function _dequant_kernel!(y, @Const(b), qt::Val, ::Val{BS}, ::Val{TS}) where {BS,TS}
    i = @index(Global, Linear)
    blk, j = divrem(i - 1, BS)
    @inbounds y[i] = _deq(qt, b, blk * TS, j)
end

# Plain float types: reinterpret the bytes and convert (coalesced reads).
_bf16_to_f32(u::UInt16) = reinterpret(Float32, UInt32(u) << 16)
function _dequant_plain!(y, bytes, t::GGMLType)
    n = length(y)
    if t == GGML_TYPE_F32
        y .= reshape(reinterpret(Float32, bytes)[1:n], size(y))
    elseif t == GGML_TYPE_F16
        y .= reshape(reinterpret(Float16, bytes)[1:n], size(y))
    else # BF16
        y .= reshape(_bf16_to_f32.(reinterpret(UInt16, bytes)[1:n]), size(y))
    end
    return y
end

"""
    dequantize!(y::AbstractArray, bytes::AbstractVector{UInt8}, type::GGMLType) -> y

Decode the values of a tensor of `type` stored in `bytes` into `y`, in memory
order (the first GGUF dimension varies fastest, as the first dimension of a
Julia array). `y` and `bytes` must be on the same device: `Array`s run on CPU
threads, GPU arrays (e.g. `CuArray`) on the GPU. Values are decoded in
`Float32` and converted to `eltype(y)`.
"""
function dequantize!(y::AbstractArray, bytes::AbstractVector{UInt8}, t::GGMLType)
    can_dequantize(t) || error("GGUFFiles: dequantization of $t is not supported")
    bytes = _kernel_bytes(bytes)
    n = length(y)
    nb = tensor_nbytes(t, (n,))
    length(bytes) >= nb || error("GGUFFiles: $(length(bytes)) bytes is too few for $n values of $t ($nb bytes)")
    n == 0 && return y
    backend = KernelAbstractions.get_backend(y)
    KernelAbstractions.get_backend(bytes) == backend ||
        error("GGUFFiles: output and tensor bytes are on different devices")
    if t in (GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16)
        return _dequant_plain!(y, bytes, t)
    end
    _dequant_kernel!(backend)(y, bytes, Val(Int(t)), Val(block_size(t)), Val(type_size(t)); ndrange=n)
    return y
end
