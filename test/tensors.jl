using Random

@testset "block sizes" begin
    @test block_size(GGUFFiles.GGML_TYPE_Q6_K) == 256
    @test type_size(GGUFFiles.GGML_TYPE_Q6_K) == 210
    @test tensor_nbytes(GGUFFiles.GGML_TYPE_Q8_0, (64, 3)) == 6 * 34
    @test tensor_nbytes(GGUFFiles.GGML_TYPE_F16, (5, 7)) == 70
    @test_throws ErrorException tensor_nbytes(GGUFFiles.GGML_TYPE_Q4_K, (100,))
    @test is_quantized(GGUFFiles.GGML_TYPE_Q4_K)
    @test !is_quantized(GGUFFiles.GGML_TYPE_BF16)
end

@testset "dequantize matches reference: $t" for t in GGUFFiles.DEQUANT_TYPES
    rng = MersenneTwister(Int(t))
    nblocks = 7
    bytes = random_blocks(rng, t, nblocks)
    n = nblocks * block_size(t)
    y = dequantize!(Vector{Float32}(undef, n), bytes, t)
    r = ref_dequant(bytes, t)
    @test length(r) == n
    @test all(isfinite, y)
    @test y ≈ r rtol = 1e-6 atol = 1e-6
    # other output types and shapes
    y16 = dequantize!(Matrix{Float16}(undef, block_size(t), nblocks), bytes, t)
    @test vec(Float32.(y16)) ≈ Float32.(Float16.(r)) rtol = 1e-3
end

@testset "Q8_0 known values" begin
    d = Float16(0.5)
    qs = Int8.(-16:15)
    bytes = vcat(reinterpret(UInt8, [d]), reinterpret(UInt8, qs))
    @test dequantize!(zeros(Float32, 32), bytes, GGUFFiles.GGML_TYPE_Q8_0) == Float32.(0.5 .* (-16:15))
end

@testset "unsupported and mismatched" begin
    @test !can_dequantize(GGUFFiles.GGML_TYPE_IQ2_XS)
    @test_throws ErrorException dequantize!(zeros(Float32, 256), zeros(UInt8, 1000), GGUFFiles.GGML_TYPE_IQ2_XS)
    @test_throws ErrorException dequantize!(zeros(Float32, 256), zeros(UInt8, 100), GGUFFiles.GGML_TYPE_Q6_K)
end

@testset "write and read a file with tensors" begin
    rng = MersenneTwister(1)
    dir = mktempdir()
    path = joinpath(dir, "t.gguf")
    w = randn(rng, Float32, 4, 3)
    q = random_blocks(rng, GGUFFiles.GGML_TYPE_Q6_K, 2)
    h = Float16.(randn(rng, Float32, 5))
    md = Dict{String,Any}(
        "general.architecture" => "test",
        "general.name" => "Ünïcode name",
        "test.count" => UInt32(3),
        "tokenizer.ggml.tokens" => ["a", "bé", "<|im_start|>"],
        "tokenizer.ggml.token_type" => Int32[1, 1, 3],
        "test.scores" => Float32[0.5, -1.0],
        "test.empty" => String[],
    )
    write_gguf(path, md, [("w", GGUFFiles.GGML_TYPE_F32, (4, 3), reinterpret(UInt8, vec(w))),
                          ("blk.0.q", GGUFFiles.GGML_TYPE_Q6_K, (256, 2), q),
                          ("h", GGUFFiles.GGML_TYPE_F16, (5,), reinterpret(UInt8, h))])
    @test read(path, 4) == b"GGUF"        # the magic as found in real files
    for mmap in (true, false)
        f = GGUFFile(path; mmap)
        @test f.metadata["general.name"] == "Ünïcode name"
        @test f.metadata["test.count"] === UInt32(3)
        @test f.metadata["tokenizer.ggml.tokens"] isa Vector{String}
        @test f.metadata["tokenizer.ggml.tokens"] == ["a", "bé", "<|im_start|>"]
        @test f.metadata["tokenizer.ggml.token_type"] == Int32[1, 1, 3]
        @test f.metadata["test.scores"] isa Vector{Float32}
        @test f.metadata["test.empty"] == String[] && f.metadata["test.empty"] isa Vector{String}
        @test keys(f) == ["w", "blk.0.q", "h"]
        @test haskey(f, "h") && !haskey(f, "nope")
        @test_throws KeyError f["nope"]
        @test size(f["w"]) == (4, 3)
        @test dequantize(f["w"]) == w
        @test dequantize(f["h"], Float16) == h
        @test dequantize(f["blk.0.q"]) ≈ reshape(ref_dequant(q, GGUFFiles.GGML_TYPE_Q6_K), 256, 2)
        @test [t.name for t in f] == keys(f)
        # memory-mapped tensors read their bytes sequentially when copied
        tq = f["blk.0.q"]
        @test (tq.data isa GGUFFiles.FileBytes) == mmap
        @test copy(tq.data) == q && collect(tq.data) == q && tq.data[3] == q[3]
        @test QuantizedArray{Float32}(tq).data === tq.data
        @test GGUFFiles.Adapt.adapt(Array, QuantizedArray{Float32}(tq)).data == q
        # every tensor starts on an alignment boundary
        @test all(ti -> ti.offset % 32 == 0, f.tensors)
        @test f.data_offset % 32 == 0
    end
end

@testset "QuantizedArray" begin
    rng = MersenneTwister(2)
    t = GGUFFiles.GGML_TYPE_Q4_K
    bytes = random_blocks(rng, t, 6)                    # 1536 values
    ref = reshape(ref_dequant(bytes, t), 512, 3)        # (in, out) as stored
    q = QuantizedArray{Float32}(bytes, t, (512, 3))
    @test size(q) == (512, 3)
    @test eltype(q) == Float32
    @test q[5, 2] ≈ ref[5, 2]
    @test Array(q) ≈ ref
    @test dequantize(q) ≈ ref
    x = randn(rng, Float32, 512, 4)
    @test transpose(q) * x ≈ transpose(ref) * x rtol = 1e-5
    @test q' * x ≈ transpose(ref) * x rtol = 1e-5
    y = randn(rng, Float32, 3)
    @test q * y ≈ ref * y rtol = 1e-5
    @test dequantize_columns(q, [3, 1, 3]) ≈ ref[:, [3, 1, 3]]
    @test_throws BoundsError dequantize_columns(q, [4])
    @test similar(q, Float16, (2, 2)) isa Matrix{Float16}
    @test occursin("QuantizedArray{Float32}", sprint(show, MIME"text/plain"(), q))
    @test_throws ErrorException QuantizedArray{Float32}(bytes[1:10], t, (512, 3))
    # Float16 compute type
    q16 = QuantizedArray{Float16}(bytes, t, (512, 3))
    @test dequantize(q16) isa Matrix{Float16}
    @test Float32.(dequantize(q16)) ≈ ref rtol = 1e-3
end
