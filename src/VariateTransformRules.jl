# This file is a part of VariateTransformRules.jl, licensed under the MIT License (MIT).

__precompile__(true)

"""
    VariateTransformRules

Rules to transform variates from one distribution to another.
"""
module VariateTransformRules

include("hello_world.jl")

function __init__()
    # @require DistributionsAD = "ced4e74d-a319-5a8a-b0ac-84af2272839c" include("distributions_ad.jl")
end

end # module
