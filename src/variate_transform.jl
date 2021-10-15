# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).


struct VariateTransformation{
    DT <: ContinuousDistribution,
    DF <: ContinuousDistribution,
    VT <: AbstractValueShape,
    VF <: AbstractValueShape,
} <: VariateTransform{VT,VF}
    target_dist::DT
    source_dist::DF
    _valshape::VT
    _varshape::VF
end

# ToDo: Add specialized dist trafo types able to cache relevant quantities, etc.


function _distrafo_ctor_impl(target_dist::Distribution, source_dist::Distribution)
    @argcheck eff_totalndof(target_dist) == eff_totalndof(source_dist)
    _valshape = varshape(target_dist)
    _varshape = varshape(source_dist)
    VariateTransformation(target_dist, source_dist, _valshape, _varshape)
end

VariateTransformation(target_dist::Distribution{VF,Continuous}, source_dist::Distribution{VF,Continuous}) where VF =
    _distrafo_ctor_impl(target_dist, source_dist)

VariateTransformation(target_dist::Distribution{Multivariate,Continuous}, source_dist::Distribution{VF,Continuous}) where VF =
    _distrafo_ctor_impl(target_dist, source_dist)

VariateTransformation(target_dist::Distribution{VF,Continuous}, source_dist::Distribution{Multivariate,Continuous}) where VF =
    _distrafo_ctor_impl(target_dist, source_dist)

VariateTransformation(target_dist::Distribution{Multivariate,Continuous}, source_dist::Distribution{Multivariate,Continuous}) =
    _distrafo_ctor_impl(target_dist, source_dist)


show_distribution(io::IO, d::Distribution) = show(io, d)
function show_distribution(io::IO, d::NamedTupleDist)
    print(io, Base.typename(typeof(d)).name, "{")
    show(io, propertynames(d))
    print(io, "}(…)")
end
    
function Base.show(io::IO, trafo::VariateTransformation)
    print(io, Base.typename(typeof(trafo)).name, "(")
    show_distribution(io, trafo.target_dist)
    print(io, ", ")
    show_distribution(io, trafo.source_dist)
    print(io, ")")
end

Base.show(io::IO, M::MIME"text/plain", trafo::VariateTransformation) = show(io, trafo)


import Base.∘
function ∘(a::VariateTransformation, b::VariateTransformation)
    @argcheck a.source_dist == b.target_dist
    VariateTransformation(a.target_dist, b.source_dist)
end

Base.inv(trafo::VariateTransformation) = VariateTransformation(trafo.source_dist, trafo.target_dist)


ValueShapes.varshape(trafo::VariateTransformation) = trafo._varshape
ValueShapes.valshape(trafo::VariateTransformation) = trafo._valshape

ValueShapes.unshaped(trafo::VariateTransformation) =
    VariateTransformation(unshaped(trafo.target_dist), unshaped(trafo.source_dist))


function apply_vartrafo_impl(trafo::VariateTransformation, v::Any, prev_ladj::OptionalLADJ)
    apply_dist_trafo(trafo.target_dist, trafo.source_dist, v, prev_ladj)
end

