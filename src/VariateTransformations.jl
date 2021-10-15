# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).

"""
    VariateTransformations

Transform variates of one distribution to variates of another.
"""
module VariateTransformations

using ChainRulesCore
using Distributions
using LinearAlgebra

import ForwardDiff
import ForwardDiffPullbacks

#include("standard_uniform.jl")
#include("standard_normal.jl")
#include("trafo_utils.jl")
#include("var_trafo_result.jl")

using Requires

function __init__()
    @require DistributionsAD = "ced4e74d-a319-5a8a-b0ac-84af2272839c" include("distributions_ad.jl")
end

end # module
