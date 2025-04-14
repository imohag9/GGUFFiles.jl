using GGUFFiles
using Documenter

DocMeta.setdocmeta!(GGUFFiles, :DocTestSetup, :(using GGUFFiles); recursive=true)

makedocs(;
    modules=[GGUFFiles],
    authors="imohag9 <souidi.hamza90@gmail.com> and contributors",
    sitename="GGUFFiles.jl",
    format=Documenter.HTML(;
        canonical="https://imohag9.github.io/GGUFFiles.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/imohag9/GGUFFiles.jl",
    devbranch="main",
)
