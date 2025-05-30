using Documenter
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
            "Model" => "interfaces/model.md",
            "State" => "interfaces/state.md"
            # "Filter" => "interfaces/filter.md"
        ],
        "Filters" => [
            "Kalman" => [
                "State" => "kalman/state.md"
                "Interface" => "kalman/filter.md"
                "Filters" => "kalman/impl.md"
            ]
        ]
    ],
    clean = true,
    checkdocs = :none
)