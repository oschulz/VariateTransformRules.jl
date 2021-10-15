# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).

#= !!!!!!!!!!!!! move
const StdUvDist = Union{StandardUvUniform, StandardUvNormal}
const StdMvDist = Union{StandardMvUniform, StandardMvNormal}
=#


"""
    transform_variate(
        trg::ContinuousDistribution,
        src::ContinuousDistribution,
        x
    )

Transforms a value `x` drawn from distribution `src` into a value drawn
from distribution `trg`.
"""
function transform_variate end
export transform_variate


# Use ForwardDiffPullbacks for univariate distribution transformations:
@inline function ChainRulesCore.rrule(::typeof(transform_variate), trg::Distribution{Univariate}, src::Distribution{Univariate}, x::Any)
    ChainRulesCore.rrule(fwddiff(transform_variate), trg, src, x)
end


"""
    VariateTransformation.eff_ndof(d::Distribution)

Returns the effective number of degrees of freedom of variates of
distribution `d`.

The effective NDOF my differ from the length of the variates. For example,
the effective NDOF for a Dirichlet distribution with variates of length `n`
is `n - 1`.

Also see [`VariateTransformation.check_compatibility`](@ref).
"""
function eff_ndof end


"""
    VariateTransformation.check_compatibility(trg::Distribution, src::Distribution)::Nothing

Check if `trg` and `src` are compatible in respect to variate
transformations, throws an `ArgumentError` if not.

Distributions are considered compatible if their variates have the same
effective number of degrees of freedom according to
[`VariateTransformation.eff_ndof`](@ref).
"""
function check_compatibility end

ChainRulesCore.rrule(::typeof(check_compatibility), trg::Distribution, src::Distribution) = nothing, _nogradient_pullback2

function check_compatibility(trg::Distribution, src::Distribution)
    trg_d_n = eff_ndof(trg)
    src_d_n = eff_ndof(src)
    if trg_d_n != src_d_n
        throw(ArgumentError("Can't convert to $(typeof(trg).name) with $(trg_d_n) eff. DOF from $(typeof(src).name) with $(src_d_n) eff. DOF"))
    end
    return nothing
end


"""
    VariateTransformation.default_intermediate(d::Distribution)

Default intermediate transformation to use when transforming between
`d` and another distribution.

The default intermediate should be the [`StandardDist`](@ref) that variates
of `d` are easiest to transform from/to.

See [`VariateTransformation.intermediate`](@ref).
"""
function default_intermediate end


"""
    VariateTransformation.intermediate(trg::Distribution, src::Distribution)

`intermediate(trg, src)` determines the [`StandardDist`](@ref) to use in
between if variates can't be transformed from directly directly.

Uses [`VariateTransformation.default_intermediate`](@ref) of `src` and
`trg`. If `src` and `trg` have different default intermediates, the uniform
distribution is selected.
"""
function intermediate end


function transform_variate(trg::Distribution, src::Distribution, x::Any)
    check_compatibility(trg, src)
    intermediate_d = intermediate(trg, src)
    intermediate_x = transform_variate(intermediate_d, src, x)
    return transform_variate(trg, intermediate_d, intermediate_x)
end


@inline intermediate(::StandardDist, src::Distribution) = intermediate(src)

@inline intermediate(trg::Distribution, ::StandardDist) = intermediate(trg)

function intermediate(::StandardDist, ::StandardDist)
    throw(ArgumentError("Direct conversions must be used between standard intermediate distributions"))
end

@inline function intermediate(trg_:Distribution, src::Distribution)
    return _select_intermediate(intermediate(trg), intermediate(src))
end

@inline _select_intermediate(a::D, ::D) where D<:Union{StdUvDist,StdMvDist} = a
@inline _select_intermediate(a::D, ::D) where D<:Union{StandardUvUniform,StandardMvUniform} = a
@inline _select_intermediate(a::Union{StandardUvUniform,StandardMvUniform}, ::Union{StdUvDist,StdMvDist}) = a
@inline _select_intermediate(::Union{StdUvDist,StdMvDist}, b::Union{StandardUvUniform,StandardMvUniform}) = b


"""
    VariateTransformation.check_variate(d::Distribution, x)::Nothing

Checks if `x` has the correct shape/size for a variate of `d`, throws an
`ArgumentError` if not.
"""
function check_variate_shape end

ChainRulesCore.rrule(::typeof(check_variate_shape), d::Distribution, x) = nothing, _nogradient_pullback2

function check_variate_shape(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{T,N}) where {T,N}
    dist_size = size(d)
    var_size = size(x)
    if dist_size != var_size
        throw(ArgumentError("Distribution has variates of size $dist_size but given variate has size $var_size"))
    end
end


"""
    VariateTransformation.check_insupport(d::Distribution, x)::Nothing

Checks if `x` is in the support of `d`, throws an `ArgumentError` if not.
"""
function check_insupport end

ChainRulesCore.rrule(::typeof(check_insupport), d::Distribution, x) = nothing, _nogradient_pullback2

function check_insupport(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{T,N}) where {T,N}
    if !insupport(d, x)
        throw(ArgumentError("Variate value is not within the support of distribution"))
    end
end


function transform_variate(trg::StandardDist{T,N}, src::StandardDist{T,N}, x) where {T,N}
    check_compatibility(trg, src)
    check_variate_shape(src, x)
    check_insupport(src, x)
    return x
end


function transform_variate(trg::Distribution{Univariate}, src::StdMvDist, x::AbstractVector{<:Real})
    @_adignore if !(length(src) == length(eachindex(x)) == 1)
        throw(ArgumentError("Length of src and length of x must be one"))
    end
    return transform_variate(trg, view(src, 1), first(x))
end

function transform_variate(trg::StdMvDist, src::Distribution{Univariate}, x::Real)
    @_adignore length(trg) == 1 || throw(ArgumentError("Length of trg must be one"))
    return Fill(transform_variate(view(trg, 1), src, x))
end
