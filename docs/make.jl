using ParetoGAN
using Documenter

DocMeta.setdocmeta!(ParetoGAN, :DocTestSetup, :(using ParetoGAN); recursive=true)

makedocs(;
    modules=[ParetoGAN],
    authors="José Manuel de Frutos Porras",
    sitename="ParetoGAN.jl",
    format=Documenter.HTML(;
        canonical="https://josemanuel22.github.io/ParetoGAN.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/josemanuel22/ParetoGAN.jl",
    devbranch="main",
)
