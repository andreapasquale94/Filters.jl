using Documenter
using DocumenterMermaid
using Filters

const CI = get(ENV, "CI", "false") == "true"

makedocs(;
    authors = "Andrea Pasquale<andrea.pasquale@outlook.it>",
    sitename = "Filters.jl",
    modules = [Filters],
    format = Documenter.HTML(; prettyurls = CI, highlights = ["yaml"], ansicolor = true),
    pages = [
        "Home" => "index.md",
        "Interfaces" => [
            "State" => "interfaces/state.md",
            "Model" => "interfaces/model.md",
            "Filter" => "interfaces/filter.md"
        ]
    ],
    clean = true,
    checkdocs = :none
)