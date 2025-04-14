# Swaps the byte order of a UInt32.
function byteswap(x::UInt32)
    return (x & 0x000000ff) << 24 | (x & 0x0000ff00) << 8 | (x & 0x00ff0000) >> 8 | (x & 0xff000000) >> 24
end

"""
    isGGUFFile(file_path::String)

Checks if a given file is a GGUF file.

# Arguments
- `file_path::String`: The path to the file.

# Returns
- `Bool`: True if the file is a GGUF file, false otherwise.
"""
function isGGUFFile(file_path::String)
    try
        header,_ , _ = parse_gguf(file_path)
        # Check magick bytes
        if header.magic != GGUF_MAGIC && byteswap(header.magic) != GGUF_MAGIC
            println("Invalid Magick bytes") 
            return false
        end

        # Check version
        if header.version != GGUF_VERSION
            println("Invalid version")
            return false
        end

        return true
    catch
        println("Couldn't open the file $file_path")
        return false
    end

end
export isGGUFFile