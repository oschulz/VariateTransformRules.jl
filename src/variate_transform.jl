# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).


"""
    struct VariateTransformation <: Function

Transforms variates between distributions.

Constructor:

```julia
VariateTransformation(
    target_dist::ContinuousDistribution,
    source_dist::ContinuousDistribution
)
```

`y = (f::VariateTransformation)(x)` transforms a value `x` drawn
from `f.source_dist` into a value `y` drawn from `f.target_dist`.

Supports `InverseFunctions.inverse` and
`ChangesOfVariables.with_logabsdet_jacobian`.
"""
struct VariateTransformation{
    DT <: ContinuousDistribution,
    DF <: ContinuousDistribution,
} <: Function
    target_dist::DT
    source_dist::DF
end

export VariateTransformation


function VariateTransformation(target_dist::Distribution, source_dist::Distribution)
    eff_ndof_trg = eff_totalndof(target_dist)
    eff_ndof_src = eff_totalndof(source_dist)
    if eff_ndof_trg != eff_ndof_src
        throw(ArgumentError("Eff. variate ndof is $eff_ndof_trg for target dist but $eff_ndof_src for source dist"))
    end
    VariateTransformation(target_dist, source_dist)
end


"""
    transform_variate(target_dist::Distribution, source_dist::Distribution)::VariateTransformation

Returns a [`VariateTransformation`](@ref), a function from `source_dist` to
`target_dist` that supports `InverseFunctions.inverse` and
`ChangesOfVariables.with_logabsdet_jacobian`.
"""
transform_variate(target_dist::Distribution, source_dist::Distribution) = VariateTransformation(target_dist::Distribution, source_dist::Distribution)


(f::VariateTransformation)(x) = transform_variate(f.target_dist, f.source_dist, x)


InverseFunctions.inverse(f::VariateTransformation) = VariateTransformation(f.source_dist, f.target_dist)


function ChangesOfVariables.with_logabsdet_jacobian(f::VariateTransformation, x)
    y = f(x)
    ladj = logpdf(f.source_dist, x) - logpdf(f.target_dist, y)
    return (y, ladj)
end


function Base.:∘(outer::VariateTransformation, inner::VariateTransformation)
    if !(outer.source_dist == inner.target_dist || isequal(outer.source_dist, inner.target_dist) || outer.source_dist ≈ inner.target_dist)
        throw(ArgumentError("Cannot compose VariateTransformations if source dist of outer doesn't equal target dist of inner."))
    end 
    VariateTransformation(outer.target_dist, inner.source_dist)
end


show_distribution(io::IO, d::Distribution) = show(io, d)
function show_distribution(io::IO, d::NamedTupleDist)
    print(io, Base.typename(typeof(d)).name, "{")
    show(io, propertynames(d))
    print(io, "}(…)")
end
    
function Base.show(io::IO, f::VariateTransformation)
    print(io, Base.typename(typeof(f)).name, "(")
    show_distribution(io, f.target_dist)
    print(io, ", ")
    show_distribution(io, f.source_dist)
    print(io, ")")
end

Base.show(io::IO, M::MIME"text/plain", f::VariateTransformation) = show(io, f)


#= !!!!!!!!!!!!!!!!!!!!!!!!!!!!
const _StdDistType = Union{Uniform, Normal}

_trg_disttype(::Type{Uniform}, ::Type{Univariate}) = StandardUvUniform
_trg_disttype(::Type{Uniform}, ::Type{Multivariate}) = StandardMvUniform
_trg_disttype(::Type{Normal}, ::Type{Univariate}) = StandardUvNormal
_trg_disttype(::Type{Normal}, ::Type{Multivariate}) = StandardMvNormal

function _trg_dist(disttype::Type{<:_StdDistType}, source_dist::Distribution{Univariate,Continuous})
    trg_dt = _trg_disttype(disttype, Univariate)
    trg_dt()
end

function _trg_dist(disttype::Type{<:_StdDistType}, source_dist::Distribution{Multivariate,Continuous})
    trg_dt = _trg_disttype(disttype, Multivariate)
    trg_dt(eff_totalndof(source_dist))
end

function _trg_dist(disttype::Type{<:_StdDistType}, source_dist::ContinuousDistribution)
    trg_dt = _trg_disttype(disttype, Multivariate)
    trg_dt(eff_totalndof(source_dist))
end


function DistributionTransform(disttype::Type{<:_StdDistType}, source_dist::ContinuousDistribution)
    trg_d = _trg_dist(disttype, source_dist)
    DistributionTransform(trg_d, source_dist)
end
=#


#= !!!!!!! move to ValueShapes !!!!!!
ValueShapes.varshape(f::VariateTransformation) = varshape(f.source_dist)
ValueShapes.valshape(f::VariateTransformation) = varshape(f.target_dist)

ValueShapes.unshaped(f::VariateTransformation) =
    VariateTransformation(unshaped(f.target_dist), unshaped(f.source_dist))
=#
