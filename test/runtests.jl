# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).

import Test
import VariateTransformations
import Documenter

Test.@testset "Package VariateTransformations" begin
    include("test_standard_dist.jl")
    #include("test_trafo_utils.jl")
    #include("test_var_trafo_result.jl")

    # doctests
    Documenter.DocMeta.setdocmeta!(
        VariateTransformations,
        :DocTestSetup,
        :(using VariateTransformations);
        recursive=true,
    )
    Documenter.doctest(VariateTransformations)
end # testset
