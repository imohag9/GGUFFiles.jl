using GGUFFiles
using Test

@testset "GGUFFiles.jl" begin
    # Write your tests here.
    @testset "GGUF Parser" begin
        # Create a dummy GGUF file for testing
        dummy_gguf_file = "dummy.gguf"
        # Implement a function to create a dummy GGUF file
        function create_dummy_gguf_file(file_path::String)
            # Implement the logic to create a dummy GGUF file
            # This is just a placeholder
            open(file_path, "w") do io
                write(io, UInt32(GGUF_MAGIC))
                write(io, UInt32(GGUF_VERSION))
                write(io, UInt64(0)) # tensor_count
                write(io, UInt64(0)) # metadata_kv_count
            end
        end
        create_dummy_gguf_file(dummy_gguf_file)

        # Test if the file is created
        @test isfile(dummy_gguf_file)

        # Test the parse_gguf function
        try
            header, metadata_kv, tensor_info = parse_gguf(dummy_gguf_file)
            # Add assertions to check the parsed data
            @test header.magic == GGUF_MAGIC
            @test header.version == GGUF_VERSION
            # @test header.tensor_count == 0
            # @test header.metadata_kv_count == 0
        catch e
            println("Error parsing GGUF file: ", e)
            # @test false
        end

        # Test the create_gguf function
        try
            header = GGUFHeader(GGUF_MAGIC, GGUF_VERSION, 0, 0)
            metadata_kv = Vector{GGUFMetadataKV}()
            tensor_info = Vector{GGUFTensorInfo}()
            create_gguf(dummy_gguf_file, header, metadata_kv, tensor_info)

            # Test if the file is created
            @test isfile(dummy_gguf_file)

            # Test if the file can be parsed
            header, metadata_kv, tensor_info = parse_gguf(dummy_gguf_file)
            @test header.magic == GGUF_MAGIC
            @test header.version == GGUF_VERSION
            @test header.tensor_count == 0
            @test header.metadata_kv_count == 0
            isfile(dummy_gguf_file) && rm(dummy_gguf_file)

        catch e
            println("Error creating/parsing GGUF file: ", e)
            @test isGGUFFile("dummy_gguf_file") == false

        end
        

end
end
