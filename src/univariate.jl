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


@inline function _eval_dist_trafo_func(f::typeof(_trafo_cdf), d::Distribution{Univariate,Continuous}, x::Real, prev_ladj::OptionalLADJ)
    R_V = float(promote_type(typeof(x), eltype(params(d)), ismissing(prev_ladj) ? Bool : typeof(prev_ladj)))
    R_LADJ = !ismissing(prev_ladj) ? R_V : Missing
    if insupport(d, x)
        y = f(d, x)
        trafo_ladj = !ismissing(prev_ladj) ? + logpdf(d, x) : missing
        var_trafo_result(convert(R_V, y), x, convert(R_LADJ, trafo_ladj), prev_ladj)
    else
        var_trafo_result(convert(R_V, NaN), x, convert(R_LADJ, !ismissing(prev_ladj) ? NaN : missing), prev_ladj)
    end
end

@inline function _eval_dist_trafo_func(f::typeof(_trafo_quantile), d::Distribution{Univariate,Continuous}, x::Real, prev_ladj::OptionalLADJ)
    R_V = float(promote_type(typeof(x), eltype(params(d)), ismissing(prev_ladj) ? Bool : typeof(prev_ladj)))
    R_LADJ = !ismissing(prev_ladj) ? R_V : Missing
    if 0 <= x <= 1
        y = f(d, x)
        trafo_ladj = !ismissing(prev_ladj) ? - logpdf(d, y) : missing
        var_trafo_result(convert(R_V, y), x, convert(R_LADJ, trafo_ladj), prev_ladj)
    else
        var_trafo_result(convert(R_V, NaN), x, convert(R_LADJ, !ismissing(prev_ladj) ? NaN : missing), prev_ladj)
    end
end



function apply_dist_trafo(::StandardUvUniform, src::Distribution{Univariate,Continuous}, x::Real, prev_ladj::OptionalLADJ)
    _eval_dist_trafo_func(_trafo_cdf, src, x, prev_ladj)
end


function apply_dist_trafo(trg::Distribution{Univariate,Continuous}, ::StandardUvUniform, x::Real, prev_ladj::OptionalLADJ)
    TV = float(typeof(x))
    # Avoid x ≈ 0 and x ≈ 1 to avoid infinite variate values for target distributions with infinite support:
    mod_src_v = ifelse(x == 0, zero(TV) + eps(TV), ifelse(x == 1, one(TV) - eps(TV), convert(TV, x)))
    y, ladj = _eval_dist_trafo_func(_trafo_quantile, trg, mod_src_v, prev_ladj)
    (v = y, ladj = ladj)
end


function _dist_trafo_rescale_impl(trg, src, x::Real, prev_ladj::OptionalLADJ)
    R = float(typeof(x))
    trg_offs, trg_scale = location(trg), scale(trg)
    src_offs, src_scale = location(src), scale(src)
    rescale_factor = trg_scale / src_scale
    y = (x - src_offs) * rescale_factor + trg_offs
    trafo_ladj = !ismissing(prev_ladj) ? log(rescale_factor) : missing
    var_trafo_result(y, x, trafo_ladj, prev_ladj)
end

@inline apply_dist_trafo(trg::Uniform, src::Uniform, x::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg, src, x, prev_ladj)
@inline apply_dist_trafo(trg::StandardUvUniform, src::Uniform, x::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg, src, x, prev_ladj)
@inline apply_dist_trafo(trg::Uniform, src::StandardUvUniform, x::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg, src, x, prev_ladj)

# ToDo: Use StandardUvNormal as standard intermediate dist for Normal? Would
# be useful if StandardUvNormal would be a better standard intermediate than
# StandardUvUniform for some other uniform distributions as well.
#
#     intermediate(src::Normal) = StandardUvNormal()
#     intermediate(trg::Normal) = StandardUvNormal()

@inline apply_dist_trafo(trg::Normal, src::Normal, x::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg, src, x, prev_ladj)
@inline apply_dist_trafo(trg::StandardUvNormal, src::Normal, x::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg, src, x, prev_ladj)
@inline apply_dist_trafo(trg::Normal, src::StandardUvNormal, x::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg, src, x, prev_ladj)


# ToDo: Optimized implementation for Distributions.Truncated <-> StandardUvUniform
