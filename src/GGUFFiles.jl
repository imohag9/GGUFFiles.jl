module GGUFFiles

using Mmap
using LinearAlgebra
using KernelAbstractions
import Adapt

include("GGUFTypes.jl")
include("GGUFRead.jl")
include("GGUFWrite.jl")
include("GGUFUtils.jl")
include("GGUFQuants.jl")
include("GGUFFile.jl")
include("QuantizedArray.jl")

export GGUFFile, GGUFTensor, dequantize, dequantize!, QuantizedArray, dequantize_columns
export block_size, type_size, tensor_nbytes, is_quantized, can_dequantize

end
