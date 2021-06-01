# This file is a part of VariateTransformRules.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package VariateTransformRules" begin
    #include("test_standard_uniform.jl")
    #include("test_standard_normal.jl")
    #include("test_trafo_utils.jl")
    #include("test_var_trafo_result.jl")
end # testset
