# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).

eff_ndof(d::DistributionsAD.TuringDirichlet) = length(d) - 1

intermediate(trg::DistributionsAD.TuringDirichlet) = StandardMvUniform(eff_ndof(trg))

function transform_variate(trg::Dirichlet, src::StandardMvUniform, x::AbstractVector{<:Real})
    return to_dirichlet(trg.alpha, src, x)
end
