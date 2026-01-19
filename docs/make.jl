using Documenter
using JCGEOutput

makedocs(
    sitename = "JCGEOutput",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        assets = [
        "assets/favicon.ico",
        "assets/logo.css",
        "assets/deepwiki-chat.css",
        "assets/deepwiki-chat.js",
        "assets/logo-theme.js",
    ]
    ),
    pages = [
        "Home" => "index.md",
        "Usage" => "usage.md",
        "API" => "api.md",
        "Citation" => "citation.md"
    ],
)


deploydocs(
    repo = "github.com/equicirco/JCGEOutput.jl",
    versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
)
