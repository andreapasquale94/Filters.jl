using Documenter
using Filters

makedocs(;
    authors="Andrea Pasquale<andrea.pasquale@outlook.it>",
    sitename="Filters.jl",
    modules=[Filters],
    format=Documenter.HTML(; highlights=["yaml"], ansicolor=true),
    pages=[
    ],
    clean=true,
    checkdocs=:none
)
