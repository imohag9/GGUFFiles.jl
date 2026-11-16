# Reference dequantizers for the tests, written independently of the kernel in
# src/GGUFQuants.jl: these follow the whole-array reshape formulation of
# gguf-py / ComfyUI-GGUF (row-major), transcribed to Julia by reversing every
# shape. Blocks are the columns of a `(type_size, nblocks)` byte matrix.

_f16v(B, r) = Float32.(reinterpret.(Float16, UInt16.(B[r:r, :]) .| (UInt16.(B[r+1:r+1, :]) .<< 8)))
_dev(v, dims...) = reshape(UInt8.(collect(v)), dims...)

function ref_dequant(bytes::Vector{UInt8}, t::GGMLType)
    if t == GGUFFiles.GGML_TYPE_F32
        return Float32.(reinterpret(Float32, bytes))
    elseif t == GGUFFiles.GGML_TYPE_F16
        return Float32.(reinterpret(Float16, bytes))
    elseif t == GGUFFiles.GGML_TYPE_BF16
        return reinterpret(Float32, UInt32.(reinterpret(UInt16, bytes)) .<< 16)
    end
    ts = type_size(t)
    n = length(bytes) ÷ ts
    B = reshape(bytes, ts, n)
    nib(qs, k) = (reshape(qs, k, 1, :, n) .>> _dev((0, 4), 1, 2, 1, 1)) .& 0x0f   # (k, 2, :, n)
    if t == GGUFFiles.GGML_TYPE_Q8_0
        return vec(_f16v(B, 1) .* Float32.(reinterpret(Int8, B[3:34, :])))
    elseif t == GGUFFiles.GGML_TYPE_Q4_0
        qs = Float32.(Int8.(reshape(nib(B[3:18, :], 16), 32, n)) .- Int8(8))
        return vec(_f16v(B, 1) .* qs)
    elseif t == GGUFFiles.GGML_TYPE_Q4_1
        qs = Float32.(reshape(nib(B[5:20, :], 16), 32, n))
        return vec(_f16v(B, 1) .* qs .+ _f16v(B, 3))
    elseif t in (GGUFFiles.GGML_TYPE_Q5_0, GGUFFiles.GGML_TYPE_Q5_1)
        o = t == GGUFFiles.GGML_TYPE_Q5_0 ? 2 : 4
        qh32 = reshape(UInt32.(B[o+1, :]) .| (UInt32.(B[o+2, :]) .<< 8) .| (UInt32.(B[o+3, :]) .<< 16) .| (UInt32.(B[o+4, :]) .<< 24), 1, n)
        qh = UInt8.((qh32 .>> UInt32.(0:31)) .& 0x1)                                     # (32, n)
        ql = reshape(nib(B[o+5:o+20, :], 16), 32, n)
        q = ql .| (qh .<< 4)
        if t == GGUFFiles.GGML_TYPE_Q5_0
            return vec(_f16v(B, 1) .* Float32.(Int8.(q) .- Int8(16)))
        else
            return vec(_f16v(B, 1) .* Float32.(q) .+ _f16v(B, 3))
        end
    elseif t == GGUFFiles.GGML_TYPE_Q2_K
        sc = B[1:16, :]
        d, dmin = _f16v(B, 81), _f16v(B, 83)
        dl = reshape(d .* Float32.(sc .& 0x0f), 1, 16, n)
        ml = reshape(dmin .* Float32.(sc .>> 4), 1, 16, n)
        qs = (reshape(B[17:80, :], 32, 1, :, n) .>> _dev((0, 2, 4, 6), 1, 4, 1, 1)) .& 0x03
        qs = reshape(qs, 16, 16, n)
        return vec(dl .* Float32.(qs) .- ml)
    elseif t in (GGUFFiles.GGML_TYPE_Q4_K, GGUFFiles.GGML_TYPE_Q5_K)
        d, dmin = _f16v(B, 1), _f16v(B, 3)
        S = reshape(B[5:16, :], 4, 3, n)
        d_, m_, md = S[:, 1:1, :], S[:, 2:2, :], S[:, 3:3, :]
        sc = reshape(vcat(d_ .& 0x3f, (md .& 0x0f) .| ((d_ .>> 2) .& 0x30)), 8, n)
        mn = reshape(vcat(m_ .& 0x3f, (md .>> 4) .| ((m_ .>> 2) .& 0x30)), 8, n)
        dd = reshape(d .* Float32.(sc), 1, 8, n)
        dm = reshape(dmin .* Float32.(mn), 1, 8, n)
        if t == GGUFFiles.GGML_TYPE_Q4_K
            q = reshape(nib(B[17:144, :], 32), 32, 8, n)
        else
            ql = reshape(nib(B[49:176, :], 32), 32, 8, n)
            qh = (reshape(B[17:48, :], 32, 1, :, n) .>> _dev(0:7, 1, 8, 1, 1)) .& 0x01
            q = ql .| (reshape(qh, 32, 8, n) .<< 4)
        end
        return vec(dd .* Float32.(q) .- dm)
    elseif t == GGUFFiles.GGML_TYPE_Q6_K
        ql = reshape(nib(B[1:128, :], 64), 32, 8, n)
        qh = (reshape(B[129:192, :], 32, 1, :, n) .>> _dev((0, 2, 4, 6), 1, 4, 1, 1)) .& 0x03
        qh = reshape(qh, 32, 8, n)
        q = reshape(Int8.(ql .| (qh .<< 4)) .- Int8(32), 16, 16, n)
        sc = Float32.(reinterpret(Int8, B[193:208, :]))
        dd = reshape(_f16v(B, 209) .* sc, 1, 16, n)
        return vec(dd .* Float32.(q))
    end
    error("no reference for $t")
end

# Random blocks of `t` whose float fields (scales, mins) are finite.
function random_blocks(rng, t::GGMLType, nblocks)
    ts = type_size(t)
    if t == GGUFFiles.GGML_TYPE_F32
        return collect(reinterpret(UInt8, randn(rng, Float32, nblocks)))
    elseif t == GGUFFiles.GGML_TYPE_F16
        return collect(reinterpret(UInt8, Float16.(randn(rng, Float32, nblocks))))
    elseif t == GGUFFiles.GGML_TYPE_BF16
        return collect(reinterpret(UInt8, UInt16.(reinterpret.(UInt32, randn(rng, Float32, nblocks)) .>> 16)))
    end
    B = rand(rng, UInt8, ts, nblocks)
    fields = Dict(GGUFFiles.GGML_TYPE_Q8_0 => (0,), GGUFFiles.GGML_TYPE_Q4_0 => (0,),
                  GGUFFiles.GGML_TYPE_Q4_1 => (0, 2), GGUFFiles.GGML_TYPE_Q5_0 => (0,),
                  GGUFFiles.GGML_TYPE_Q5_1 => (0, 2), GGUFFiles.GGML_TYPE_Q2_K => (80, 82),
                  GGUFFiles.GGML_TYPE_Q4_K => (0, 2), GGUFFiles.GGML_TYPE_Q5_K => (0, 2),
                  GGUFFiles.GGML_TYPE_Q6_K => (208,))[t]
    for o in fields, k in 1:nblocks
        h = reinterpret(UInt16, Float16(2 * rand(rng) - 1))
        B[o+1, k] = h % UInt8
        B[o+2, k] = (h >> 8) % UInt8
    end
    return vec(B)
end
