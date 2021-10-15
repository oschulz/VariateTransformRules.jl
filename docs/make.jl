# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using VariateTransformations

# Doctest setup
DocMeta.setdocmeta!(
    VariateTransformations,
    :DocTestSetup,
    :(using VariateTransformations);
    recursive=true,
)

makedocs(
    sitename = "VariateTransformations",
    modules = [VariateTransformations],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://oschulz.github.io/VariateTransformations.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    strict = !("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/oschulz/VariateTransformations.jl.git",
    forcepush = true,
    push_preview = true,
)
