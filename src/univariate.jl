# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).

intermediate(src::Distribution{Univariate,Continuous}) = StandardUvUniform()

check_compatibility(trg::UnivariateDistribution, src::UnivariateDistribution) = nothing


@inline _trafo_cdf(d::Distribution{Univariate,Continuous}, x::Real) = _trafo_cdf_impl(params(d), d, x)

@inline _trafo_cdf_impl(::NTuple, d::Distribution{Univariate,Continuous}, x::Real) = cdf(d, x)

@inline function _trafo_cdf_impl(::NTuple{N,Union{Integer,AbstractFloat}}, d::Distribution{Univariate,Continuous}, x::ForwardDiff.Dual{TAG}) where {N,TAG}
    x_v = ForwardDiff.value(x)
    u = cdf(d, x_v)
    dudx = pdf(d, x_v)
    return ForwardDiff.Dual{TAG}(u, dudx * ForwardDiff.partials(x))
end


@inline _trafo_quantile(d::Distribution{Univariate,Continuous}, u::Real) = _trafo_quantile_impl(params(d), d, u)

@inline _trafo_quantile_impl(::NTuple, d::Distribution{Univariate,Continuous}, u::Real) = _trafo_quantile_impl_generic(d, u)

@inline function _trafo_quantile_impl(::NTuple{N,Union{Integer,AbstractFloat}}, d::Distribution{Univariate,Continuous}, u::ForwardDiff.Dual{TAG}) where {N,TAG}
    x = _trafo_quantile_impl_generic(d, ForwardDiff.value(u))
    dxdu = inv(pdf(d, x))
    ForwardDiff.Dual{TAG}(x, dxdu * ForwardDiff.partials(u))
end

# Workaround for Beta dist, ForwardDiff doesn't work for parameters:
@inline _trafo_quantile_impl(::NTuple{N,ForwardDiff.Dual}, d::Beta, u::Real) where N = convert(float(typeof(u)), NaN)
# Workaround for Beta dist, current quantile implementation only supports Float64:
@inline _trafo_quantile_impl(::NTuple{N,Real}, d::Beta, u::Float32) where N = quantile(d, convert(promote_type(Float64, typeof(u)), u))

@inline _trafo_quantile_impl_generic(d::Distribution{Univariate,Continuous}, u::Real) = quantile(d, u)

# Workaround for rounding errors that can result in quantile values outside of support of Truncated:
@inline function _trafo_quantile_impl_generic(d::Truncated{<:Distribution{Univariate,Continuous}}, u::Real)
    x = quantile(d, u)
    T = typeof(x)
    min_x = T(minimum(d))
    max_x = T(maximum(d))
    if x < min_x && isapprox(x, min_x, atol = 4 * eps(T))
        min_x
    elseif x > max_x && isapprox(x, max_x, atol = 4 * eps(T))
        max_x
    else
        x
    end
end


@inline function result_numtype(d::Distribution{Univariate}, x::T) where {T<:Real}
    firsttype(first(typeof(x), promote_type(map(eltype, params(d))...)))
end

function transform_variate(::StandardUvUniform, src::Distribution{Univariate,Continuous}, x::T) where {T<:Real}
    R = result_numtype(src, x)
    if insupport(d, x)
        convert(R, _trafo_cdf(d, x))
    else
        convert(R, NaN)
    end
end


function transform_variate(trg::Distribution{Univariate,Continuous}, ::StandardUvUniform, x::T) where {T<:Real}
    R = result_numtype(trg, x)
    TF = float(T)
    if 0 <= x <= 1
        # Avoid x ≈ 0 and x ≈ 1 to avoid infinite variate values for target distributions with infinite support:
        mod_x = ifelse(x == 0, zero(TF) + eps(TF), ifelse(x == 1, one(TF) - eps(TF), convert(TF, x)))
        convert(R, _trafo_quantile(d, mod_x))
    else
        convert(R, NaN)
    end
end


function _rescaled_to_standard(src::UnivariateDistribution, x::T) where {T<:Real}
    src_offs, src_scale = location(src), scale(src)
    y = (x - src_offs) / src_scale
    convert(result_numtype(src, x), y)
end

function _standard_to_rescaled(trg::UnivariateDistribution, x::T) where {T<:Real}
    trg_offs, trg_scale = location(trg), scale(trg)
    y = muladd(x, trg_scale, trg_offs)
    convert(result_numtype(src, x), y)
end

@inline transform_variate(trg::StandardUvUniform, src::Uniform, x::Real) = _rescaled_to_standard(src, x)
@inline transform_variate(trg::Uniform, src::StandardUvUniform, x::Real) = _standard_to_rescaled(trg, x)

@inline transform_variate(trg::StandardUvNormal, src::Normal, x::Real) = _rescaled_to_standard(src, x)
@inline transform_variate(trg::Normal, src::StandardUvNormal, x::Real) = _standard_to_rescaled(trg, x)


# ToDo: Optimized implementation for Distributions.Truncated <-> StandardUvUniform
