# GGUFFiles

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://imohag9.github.io/GGUFFiles.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://imohag9.github.io/GGUFFiles.jl/dev/)
[![Build Status](https://github.com/imohag9/GGUFFiles.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/imohag9/GGUFFiles.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Introduction

GGUFFiles is a Julia package for reading and writing GGUF (GGUF: GGML Unified File Format) files. It provides the necessary tools to parse, create, and manipulate GGUF files, which are used for storing and distributing machine learning models.

## Features

*   Reading GGUF headers, metadata, and tensor information.
*   Writing GGUF headers, metadata, and tensor information.
*   Support for various GGUF value types

## Installation

To install GGUFFiles, use the Julia package manager:

```julia
import Pkg
Pkg.add("GGUFFiles")
```

## Usage

Here's a basic example of how to parse a GGUF file:

```julia
using GGUFFiles

# Replace "your_gguf_file.gguf" with the path to your GGUF file
file_path = "your_gguf_file.gguf"

try
    header, metadata_kv, tensor_info = parse_gguf(file_path)

    println("Header: ", header)
    println("Metadata count: ", length(metadata_kv))
    println("Tensor count: ", length(tensor_info))
catch e
    println("Error parsing GGUF file: ", e)
end
```

And here's an example of how to create a GGUF file:

```julia
using GGUFFiles

# Define header
header = GGUFHeader(GGUF_MAGIC, GGUF_VERSION, 0, 0)

# Define metadata
metadata_kv = Vector{GGUFMetadataKV}()

# Define tensor info
tensor_info = Vector{GGUFTensorInfo}()

# Replace "your_gguf_file.gguf" with the desired file path
file_path = "your_gguf_file.gguf"

try
    create_gguf(file_path, header, metadata_kv, tensor_info)
    println("GGUF file created successfully.")
catch e
    println("Error creating GGUF file: ", e)
end
```

## API Reference

### Functions

* `create_gguf(file_path::String, header::GGUFHeader, metadata_kv::Vector{GGUFMetadataKV}, tensor_info::Vector{GGUFTensorInfo})`: Creates a GGUF file.

* `parse_gguf(file_path::String)`: Parses a GGUF file.

* Full API is listed in the docs.
