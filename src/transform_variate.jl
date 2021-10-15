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
    VariateTransformation.check_compatibility(trg::Distribution, src::Distribution)

Check if `trg` and `src` are compatible in respect to variate
transformations, throws an `ArgumentError` if not.

Distributions are considered compatible if their variates have the same
effective number of degrees of freedom according to
[`VariateTransformation.eff_ndof`](@ref).
"""
function check_compatibility end


function check_compatibility(trg::Distribution, src::Distribution)
    trg_d_n = eff_ndof(trg)
    src_d_n = eff_ndof(src)
    if trg_d_n != src_d_n
        throw(ArgumentError("Can't convert to $(typeof(trg).name) with $(trg_d_n) eff. DOF from $(typeof(src).name) with $(src_d_n) eff. DOF"))
    end
    nothing
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

@inline intermediate(::StandardDist, src::Distribution) = intermediate(src)

@inline intermediate(trg::Distribution, ::StandardDist) = intermediate(trg)

function intermediate(::StandardDist, ::StandardDist)
    throw(ArgumentError("Direct conversions must be used between standard intermediate distributions"))
end

@inline function intermediate(trg_:Distribution, src::Distribution)
    _select_intermediate(intermediate(trg), intermediate(src))
end

@inline _select_intermediate(a::D, ::D) where D<:Union{StdUvDist,StdMvDist} = a
@inline _select_intermediate(a::D, ::D) where D<:Union{StandardUvUniform,StandardMvUniform} = a
@inline _select_intermediate(a::Union{StandardUvUniform,StandardMvUniform}, ::Union{StdUvDist,StdMvDist}) = a
@inline _select_intermediate(::Union{StdUvDist,StdMvDist}, b::Union{StandardUvUniform,StandardMvUniform}) = b



function apply_dist_trafo(trg::Distribution, src::Distribution, x::Any, prev_ladj::OptionalLADJ)
    check_compatibility(trg, src)
    intermediate_d = intermediate(trg, src)
    intermediate_v, intermediate_ladj = apply_dist_trafo(intermediate_d, src, x, prev_ladj)
    apply_dist_trafo(trg, intermediate_d, intermediate_v, intermediate_ladj)
end


function apply_dist_trafo(trg::DT, src::DT, x::Real, prev_ladj::OptionalLADJ) where {DT <: StdUvDist}
    (v = x, ladj = prev_ladj)
end

function apply_dist_trafo(trg::DT, src::DT, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ) where {DT <: StdMvDist}
    @argcheck length(trg) == length(src) == length(eachindex(x))
    (v = x, ladj = prev_ladj)
end


function apply_dist_trafo(trg::Distribution{Univariate}, src::StdMvDist, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck length(src) == length(eachindex(x)) == 1
    apply_dist_trafo(trg, view(src, 1), first(x), prev_ladj)
end

function apply_dist_trafo(trg::StdMvDist, src::Distribution{Univariate}, x::Real, prev_ladj::OptionalLADJ)
    @argcheck length(trg) == 1
    r = apply_dist_trafo(view(trg, 1), src, first(x), prev_ladj)
    (v = unshaped(r.v), ladj = r.ladj)
end


@inline _trafo_cdf(d::Distribution{Univariate,Continuous}, x::Real) = _trafo_cdf_impl(params(d), d, x)

@inline _trafo_cdf_impl(::NTuple, d::Distribution{Univariate,Continuous}, x::Real) = cdf(d, x)

@inline function _trafo_cdf_impl(::NTuple{N,Union{Integer,AbstractFloat}}, d::Distribution{Univariate,Continuous}, x::ForwardDiff.Dual{TAG}) where {N,TAG}
    x_v = ForwardDiff.value(x)
    u = cdf(d, x_v)
    dudx = pdf(d, x_v)
    ForwardDiff.Dual{TAG}(u, dudx * ForwardDiff.partials(x))
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


intermediate(src::Distribution{Univariate,Continuous}) = StandardUvUniform()

function apply_dist_trafo(::StandardUvUniform, src::Distribution{Univariate,Continuous}, x::Real, prev_ladj::OptionalLADJ)
    _eval_dist_trafo_func(_trafo_cdf, src, x, prev_ladj)
end

intermediate(trg::Distribution{Univariate,Continuous}) = StandardUvUniform()

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


@inline function apply_dist_trafo(trg::StandardMvUniform, src::StandardMvNormal, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg) == eff_ndof(src)
    _product_dist_trafo_impl(StandardUvUniform(), StandardUvNormal(), x, prev_ladj)
end

@inline function apply_dist_trafo(trg::StandardMvNormal, src::StandardMvUniform, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg) == eff_ndof(src)
    _product_dist_trafo_impl(StandardUvNormal(), StandardUvUniform(), x, prev_ladj)
end


intermediate(src::MvNormal) = StandardMvNormal(length(src))

function apply_dist_trafo(trg::StandardMvNormal, src::MvNormal, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @argcheck length(trg) == length(src)
    A = cholesky(src.Σ).U
    y = transpose(A) \ (x - src.μ)
    trafo_ladj = !ismissing(prev_ladj) ? - logabsdet(A)[1] : missing
    var_trafo_result(y, x, trafo_ladj, prev_ladj)
end

intermediate(trg::MvNormal) = StandardMvNormal(length(trg))

function apply_dist_trafo(trg::MvNormal, src::StandardMvNormal, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @argcheck length(trg) == length(src)
    A = cholesky(trg.Σ).U
    y = transpose(A) * x + trg.μ
    trafo_ladj = !ismissing(prev_ladj) ? + logabsdet(A)[1] : missing
    var_trafo_result(y, x, trafo_ladj, prev_ladj)
end


eff_ndof(d::Dirichlet) = length(d) - 1
eff_ndof(d::DistributionsAD.TuringDirichlet) = length(d) - 1

intermediate(trg::Dirichlet) = StandardMvUniform(eff_ndof(trg))
intermediate(trg::DistributionsAD.TuringDirichlet) = StandardMvUniform(eff_ndof(trg))


function apply_dist_trafo(trg::Dirichlet, src::StandardMvUniform, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    apply_dist_trafo(DistributionsAD.TuringDirichlet(trg.alpha), src, x, prev_ladj)
end

function _dirichlet_beta_trafo(α::Real, β::Real, x::Real)
    R = float(promote_type(typeof(α), typeof(β), typeof(x)))
    convert(R, apply_dist_trafo(Beta(α, β), StandardUvUniform(), x, missing).v)::R
end

_a_times_one_minus_b(a::Real, b::Real) = a * (1 - b)

function apply_dist_trafo(trg::DistributionsAD.TuringDirichlet, src::StandardMvUniform, x::AbstractVector{<:Real}, prev_ladj::Missing)
    # See M. J. Betancourt, "Cruising The Simplex: Hamiltonian Monte Carlo and the Dirichlet Distribution",
    # https://arxiv.org/abs/1010.3436

    @_adignore @argcheck length(trg) == length(src) + 1
    αs = _dropfront(_rev_cumsum(trg.alpha))
    βs = _dropback(trg.alpha)
    beta_v = fwddiff(_dirichlet_beta_trafo).(αs, βs, x)
    beta_v_cp = _exp_cumsum_log(_pushfront(beta_v, 1))
    beta_v_ext = _pushback(beta_v, 0)
    y = fwddiff(_a_times_one_minus_b).(beta_v_cp, beta_v_ext)
    var_trafo_result(y, x, missing, prev_ladj)
end

function apply_dist_trafo(trg::DistributionsAD.TuringDirichlet, src::StandardMvUniform, x::AbstractVector{<:Real}, prev_ladj::Real)
    y = apply_dist_trafo(trg, src, x, missing).v
    trafo_ladj = - logpdf(trg, y)
    var_trafo_result(y, x, trafo_ladj, prev_ladj)
end


g_debug = nothing
function _product_dist_trafo_impl(trg_ds, src_ds, x::AbstractVector{<:Real}, prev_ladj::Missing)
    global g_debug = (trg_ds=trg_ds, src_ds=src_ds, x=x, prev_ladj=missing)
    y = fwddiff(apply_dist_trafo_noladj).(trg_ds, src_ds, x)
    var_trafo_result(y, x, missing, missing)
end

function _product_dist_trafo_impl(trg_ds, src_ds, x::AbstractVector{<:Real}, prev_ladj::Real)
    rs = fwddiff(apply_dist_trafo).(trg_ds, src_ds, x, zero(prev_ladj))
    y = map(r -> r.v, rs)
    trafo_ladj = sum(map(r -> r.ladj, rs))
    var_trafo_result(y, x, trafo_ladj, prev_ladj)
end

function apply_dist_trafo(trg::Distributions.Product, src::Distributions.Product, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg) == eff_ndof(src)
    _product_dist_trafo_impl(trg.v, src.v, x, prev_ladj)
end

function apply_dist_trafo(trg::StandardMvUniform, src::Distributions.Product, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg) == eff_ndof(src)
    _product_dist_trafo_impl(StandardUvUniform(), src.v, x, prev_ladj)
end

function apply_dist_trafo(trg::StandardMvNormal, src::Distributions.Product, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg) == eff_ndof(src)
    _product_dist_trafo_impl(StandardUvNormal(), src.v, x, prev_ladj)
end

function apply_dist_trafo(trg::Distributions.Product, src::StandardMvUniform, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg) == eff_ndof(src)
    _product_dist_trafo_impl(trg.v, StandardUvUniform(), x, prev_ladj)
end

function apply_dist_trafo(trg::Distributions.Product, src::StandardMvNormal, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg) == eff_ndof(src)
    _product_dist_trafo_impl(trg.v, StandardUvNormal(), x, prev_ladj)
end


function _ntdistelem_to_stdmv(trg::StdMvDist, sd::Distribution, src_v_unshaped::AbstractVector{<:Real}, trg_acc::ValueAccessor, init_ladj::OptionalLADJ)
    td = view(trg, ValueShapes.view_range(Base.OneTo(length(trg)), trg_acc))
    sv = stripscalar(view(src_v_unshaped, trg_acc))
    apply_dist_trafo(td, sd, sv, init_ladj)
end

function _ntdistelem_to_stdmv(trg::StdMvDist, sd::ConstValueDist, src_v_unshaped::AbstractVector{<:Real}, trg_acc::ValueAccessor, init_ladj::OptionalLADJ)
    (v = Bool[], ladj = init_ladj)
end


_transformed_ntd_elshape(d::Distribution{Univariate}) = varshape(d)
_transformed_ntd_elshape(d::Distribution{Multivariate}) = ArrayShape{Real}(eff_ndof(d))
function _transformed_ntd_elshape(d::Distribution)
    vs = varshape(d)
    @argcheck totalndof(vs) == eff_ndof(d)
    vs
end

function _transformed_ntd_accessors(d::NamedTupleDist{names}) where names
    shapes = map(_transformed_ntd_elshape, values(d))
    vs = NamedTupleShape(NamedTuple{names}(shapes))
    values(vs)
end

function apply_dist_trafo(trg::StdMvDist, src::ValueShapes.UnshapedNTD, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    src_vs = varshape(src.shaped)
    @argcheck length(src) == length(eachindex(x))
    trg_accessors = _transformed_ntd_accessors(src.shaped)
    init_ladj = ismissing(prev_ladj) ? missing : zero(Float32)
    rs = map((acc, sd) -> _ntdistelem_to_stdmv(trg, sd, x, acc, init_ladj), trg_accessors, values(src.shaped))
    y = vcat(map(r -> r.v, rs)...)
    trafo_ladj = !ismissing(prev_ladj) ? sum(map(r -> r.ladj, rs)) : missing
    var_trafo_result(y, x, trafo_ladj, prev_ladj)
end

function apply_dist_trafo(trg::StdMvDist, src::NamedTupleDist, x::Union{NamedTuple,ShapedAsNT}, prev_ladj::OptionalLADJ)
    src_v_unshaped = unshaped(x, varshape(src))
    apply_dist_trafo(trg, unshaped(src), src_v_unshaped, prev_ladj)
end

function _stdmv_to_ntdistelem(td::Distribution, src::StdMvDist, x::AbstractVector{<:Real}, src_acc::ValueAccessor, init_ladj::OptionalLADJ)
    sd = view(src, ValueShapes.view_range(Base.OneTo(length(src)), src_acc))
    sv = view(x, ValueShapes.view_range(axes(x, 1), src_acc))
    apply_dist_trafo(td, sd, sv, init_ladj)
end

function _stdmv_to_ntdistelem(td::ConstValueDist, src::StdMvDist, x::AbstractVector{<:Real}, src_acc::ValueAccessor, init_ladj::OptionalLADJ)
    (v = Bool[], ladj = init_ladj)
end

function apply_dist_trafo(trg::ValueShapes.UnshapedNTD, src::StdMvDist, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    trg_vs = varshape(trg.shaped)
    @argcheck length(src) == length(eachindex(x))
    src_accessors = _transformed_ntd_accessors(trg.shaped)
    init_ladj = ismissing(prev_ladj) ? missing : zero(Float32)
    rs = map((acc, td) -> _stdmv_to_ntdistelem(td, src, x, acc, init_ladj), src_accessors, values(trg.shaped))
    trg_v_unshaped = vcat(map(r -> unshaped(r.v), rs)...)
    trafo_ladj = !ismissing(prev_ladj) ? sum(map(r -> r.ladj, rs)) : missing
    var_trafo_result(trg_v_unshaped, x, trafo_ladj, prev_ladj)
end

function apply_dist_trafo(trg::NamedTupleDist, src::StdMvDist, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    unshaped_result = apply_dist_trafo(unshaped(trg), src, x, prev_ladj)
    (v = strip_shapedasnt(varshape(trg)(unshaped_result.v)), ladj = unshaped_result.ladj)
end


const AnyReshapedDist = Union{ReshapedDist,MatrixReshaped}

function apply_dist_trafo(trg::Distribution{Multivariate}, src::AnyReshapedDist, x::Any, prev_ladj::OptionalLADJ)
    src_vs = varshape(src)
    @argcheck length(trg) == totalndof(src_vs)
    apply_dist_trafo(trg, unshaped(src), unshaped(x, src_vs), prev_ladj)
end

function apply_dist_trafo(trg::AnyReshapedDist, src::Distribution{Multivariate}, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    trg_vs = varshape(trg)
    @argcheck totalndof(trg_vs) == length(src)
    r = apply_dist_trafo(unshaped(trg), src, x, prev_ladj)
    (v = trg_vs(r.v), ladj = r.ladj)
end

function apply_dist_trafo(trg::AnyReshapedDist, src::AnyReshapedDist, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    trg_vs = varshape(trg)
    src_vs = varshape(src)
    @argcheck totalndof(trg_vs) == totalndof(src_vs)
    r = apply_dist_trafo(unshaped(trg), unshaped(src), unshaped(x, src_vs), prev_ladj)
    (v = trg_vs(r.v), ladj = r.ladj)
end


function apply_dist_trafo(trg::StdMvDist, src::UnshapedHDist, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    src_v_primary, src_v_secondary = _hd_split(src, x)
    trg_d_primary = typeof(trg)(length(eachindex(src_v_primary)))
    trg_d_secondary = typeof(trg)(length(eachindex(src_v_secondary)))
    trg_v_primary, ladj_primary = apply_dist_trafo(trg_d_primary, _hd_pridist(src), src_v_primary, prev_ladj)
    trg_v_secondary, ladj = apply_dist_trafo(trg_d_secondary, _hd_secdist(src, src_v_primary), src_v_secondary, ladj_primary)
    y = vcat(trg_v_primary, trg_v_secondary)
    (v = y, ladj = ladj)
end

function apply_dist_trafo(trg::StdMvDist, src::HierarchicalDistribution, x::Any, prev_ladj::OptionalLADJ)
    src_v_unshaped = unshaped(x, varshape(src))
    apply_dist_trafo(trg, unshaped(src), src_v_unshaped, prev_ladj)
end

function apply_dist_trafo(trg::UnshapedHDist, src::StdMvDist, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    src_v_primary, src_v_secondary = _hd_split(trg, x)
    src_d_primary = typeof(src)(length(eachindex(src_v_primary)))
    src_d_secondary = typeof(src)(length(eachindex(src_v_secondary)))
    trg_v_primary, ladj_primary = apply_dist_trafo(_hd_pridist(trg), src_d_primary, src_v_primary, prev_ladj)
    trg_v_secondary, ladj = apply_dist_trafo(_hd_secdist(trg, trg_v_primary), src_d_secondary, src_v_secondary, ladj_primary)
    y = vcat(trg_v_primary, trg_v_secondary)
    (v = y, ladj = ladj)
end

function apply_dist_trafo(trg::HierarchicalDistribution, src::StdMvDist, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    unshaped_result = apply_dist_trafo(unshaped(trg), src, x, prev_ladj)
    (v = varshape(trg)(unshaped_result.v), ladj = unshaped_result.ladj)
end
