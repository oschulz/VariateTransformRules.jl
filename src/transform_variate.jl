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
    ChainRulesCore.rrule(fwddiff(transform_variate), trg_d, src_d, src_v)
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


function check_compatibility(trg_d::Distribution, src_d::Distribution)
    trg_d_n = eff_ndof(trg_d)
    src_d_n = eff_ndof(src_d)
    if trg_d_n != src_d_n
        throw(ArgumentError("Can't convert to $(typeof(trg_d).name) with $(trg_d_n) eff. DOF from $(typeof(src_d).name) with $(src_d_n) eff. DOF"))
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



function apply_dist_trafo(trg_d::Distribution, src_d::Distribution, src_v::Any, prev_ladj::OptionalLADJ)
    check_compatibility(trg_d, src_d)
    intermediate_d = intermediate(trg_d, src_d)
    intermediate_v, intermediate_ladj = apply_dist_trafo(intermediate_d, src_d, src_v, prev_ladj)
    apply_dist_trafo(trg_d, intermediate_d, intermediate_v, intermediate_ladj)
end


function apply_dist_trafo(trg_d::DT, src_d::DT, src_v::Real, prev_ladj::OptionalLADJ) where {DT <: StdUvDist}
    (v = src_v, ladj = prev_ladj)
end

function apply_dist_trafo(trg_d::DT, src_d::DT, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ) where {DT <: StdMvDist}
    @argcheck length(trg_d) == length(src_d) == length(eachindex(src_v))
    (v = src_v, ladj = prev_ladj)
end


function apply_dist_trafo(trg_d::Distribution{Univariate}, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck length(src_d) == length(eachindex(src_v)) == 1
    apply_dist_trafo(trg_d, view(src_d, 1), first(src_v), prev_ladj)
end

function apply_dist_trafo(trg_d::StdMvDist, src_d::Distribution{Univariate}, src_v::Real, prev_ladj::OptionalLADJ)
    @argcheck length(trg_d) == 1
    r = apply_dist_trafo(view(trg_d, 1), src_d, first(src_v), prev_ladj)
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


@inline function _eval_dist_trafo_func(f::typeof(_trafo_cdf), d::Distribution{Univariate,Continuous}, src_v::Real, prev_ladj::OptionalLADJ)
    R_V = float(promote_type(typeof(src_v), eltype(params(d)), ismissing(prev_ladj) ? Bool : typeof(prev_ladj)))
    R_LADJ = !ismissing(prev_ladj) ? R_V : Missing
    if insupport(d, src_v)
        trg_v = f(d, src_v)
        trafo_ladj = !ismissing(prev_ladj) ? + logpdf(d, src_v) : missing
        var_trafo_result(convert(R_V, trg_v), src_v, convert(R_LADJ, trafo_ladj), prev_ladj)
    else
        var_trafo_result(convert(R_V, NaN), src_v, convert(R_LADJ, !ismissing(prev_ladj) ? NaN : missing), prev_ladj)
    end
end

@inline function _eval_dist_trafo_func(f::typeof(_trafo_quantile), d::Distribution{Univariate,Continuous}, src_v::Real, prev_ladj::OptionalLADJ)
    R_V = float(promote_type(typeof(src_v), eltype(params(d)), ismissing(prev_ladj) ? Bool : typeof(prev_ladj)))
    R_LADJ = !ismissing(prev_ladj) ? R_V : Missing
    if 0 <= src_v <= 1
        trg_v = f(d, src_v)
        trafo_ladj = !ismissing(prev_ladj) ? - logpdf(d, trg_v) : missing
        var_trafo_result(convert(R_V, trg_v), src_v, convert(R_LADJ, trafo_ladj), prev_ladj)
    else
        var_trafo_result(convert(R_V, NaN), src_v, convert(R_LADJ, !ismissing(prev_ladj) ? NaN : missing), prev_ladj)
    end
end


intermediate(src_d::Distribution{Univariate,Continuous}) = StandardUvUniform()

function apply_dist_trafo(::StandardUvUniform, src_d::Distribution{Univariate,Continuous}, src_v::Real, prev_ladj::OptionalLADJ)
    _eval_dist_trafo_func(_trafo_cdf, src_d, src_v, prev_ladj)
end

intermediate(trg_d::Distribution{Univariate,Continuous}) = StandardUvUniform()

function apply_dist_trafo(trg_d::Distribution{Univariate,Continuous}, ::StandardUvUniform, src_v::Real, prev_ladj::OptionalLADJ)
    TV = float(typeof(src_v))
    # Avoid src_v ≈ 0 and src_v ≈ 1 to avoid infinite variate values for target distributions with infinite support:
    mod_src_v = ifelse(src_v == 0, zero(TV) + eps(TV), ifelse(src_v == 1, one(TV) - eps(TV), convert(TV, src_v)))
    trg_v, ladj = _eval_dist_trafo_func(_trafo_quantile, trg_d, mod_src_v, prev_ladj)
    (v = trg_v, ladj = ladj)
end



function _dist_trafo_rescale_impl(trg_d, src_d, src_v::Real, prev_ladj::OptionalLADJ)
    R = float(typeof(src_v))
    trg_offs, trg_scale = location(trg_d), scale(trg_d)
    src_offs, src_scale = location(src_d), scale(src_d)
    rescale_factor = trg_scale / src_scale
    trg_v = (src_v - src_offs) * rescale_factor + trg_offs
    trafo_ladj = !ismissing(prev_ladj) ? log(rescale_factor) : missing
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end

@inline apply_dist_trafo(trg_d::Uniform, src_d::Uniform, src_v::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)
@inline apply_dist_trafo(trg_d::StandardUvUniform, src_d::Uniform, src_v::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)
@inline apply_dist_trafo(trg_d::Uniform, src_d::StandardUvUniform, src_v::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)

# ToDo: Use StandardUvNormal as standard intermediate dist for Normal? Would
# be useful if StandardUvNormal would be a better standard intermediate than
# StandardUvUniform for some other uniform distributions as well.
#
#     intermediate(src_d::Normal) = StandardUvNormal()
#     intermediate(trg_d::Normal) = StandardUvNormal()

@inline apply_dist_trafo(trg_d::Normal, src_d::Normal, src_v::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)
@inline apply_dist_trafo(trg_d::StandardUvNormal, src_d::Normal, src_v::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)
@inline apply_dist_trafo(trg_d::Normal, src_d::StandardUvNormal, src_v::Real, prev_ladj::OptionalLADJ) = _dist_trafo_rescale_impl(trg_d, src_d, src_v, prev_ladj)


# ToDo: Optimized implementation for Distributions.Truncated <-> StandardUvUniform


@inline function apply_dist_trafo(trg_d::StandardMvUniform, src_d::StandardMvNormal, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg_d) == eff_ndof(src_d)
    _product_dist_trafo_impl(StandardUvUniform(), StandardUvNormal(), src_v, prev_ladj)
end

@inline function apply_dist_trafo(trg_d::StandardMvNormal, src_d::StandardMvUniform, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg_d) == eff_ndof(src_d)
    _product_dist_trafo_impl(StandardUvNormal(), StandardUvUniform(), src_v, prev_ladj)
end


intermediate(src_d::MvNormal) = StandardMvNormal(length(src_d))

function apply_dist_trafo(trg_d::StandardMvNormal, src_d::MvNormal, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @argcheck length(trg_d) == length(src_d)
    A = cholesky(src_d.Σ).U
    trg_v = transpose(A) \ (src_v - src_d.μ)
    trafo_ladj = !ismissing(prev_ladj) ? - logabsdet(A)[1] : missing
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end

intermediate(trg_d::MvNormal) = StandardMvNormal(length(trg_d))

function apply_dist_trafo(trg_d::MvNormal, src_d::StandardMvNormal, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @argcheck length(trg_d) == length(src_d)
    A = cholesky(trg_d.Σ).U
    trg_v = transpose(A) * src_v + trg_d.μ
    trafo_ladj = !ismissing(prev_ladj) ? + logabsdet(A)[1] : missing
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end


eff_ndof(d::Dirichlet) = length(d) - 1
eff_ndof(d::DistributionsAD.TuringDirichlet) = length(d) - 1

intermediate(trg_d::Dirichlet) = StandardMvUniform(eff_ndof(trg_d))
intermediate(trg_d::DistributionsAD.TuringDirichlet) = StandardMvUniform(eff_ndof(trg_d))


function apply_dist_trafo(trg_d::Dirichlet, src_d::StandardMvUniform, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    apply_dist_trafo(DistributionsAD.TuringDirichlet(trg_d.alpha), src_d, src_v, prev_ladj)
end

function _dirichlet_beta_trafo(α::Real, β::Real, src_v::Real)
    R = float(promote_type(typeof(α), typeof(β), typeof(src_v)))
    convert(R, apply_dist_trafo(Beta(α, β), StandardUvUniform(), src_v, missing).v)::R
end

_a_times_one_minus_b(a::Real, b::Real) = a * (1 - b)

function apply_dist_trafo(trg_d::DistributionsAD.TuringDirichlet, src_d::StandardMvUniform, src_v::AbstractVector{<:Real}, prev_ladj::Missing)
    # See M. J. Betancourt, "Cruising The Simplex: Hamiltonian Monte Carlo and the Dirichlet Distribution",
    # https://arxiv.org/abs/1010.3436

    @_adignore @argcheck length(trg_d) == length(src_d) + 1
    αs = _dropfront(_rev_cumsum(trg_d.alpha))
    βs = _dropback(trg_d.alpha)
    beta_v = fwddiff(_dirichlet_beta_trafo).(αs, βs, src_v)
    beta_v_cp = _exp_cumsum_log(_pushfront(beta_v, 1))
    beta_v_ext = _pushback(beta_v, 0)
    trg_v = fwddiff(_a_times_one_minus_b).(beta_v_cp, beta_v_ext)
    var_trafo_result(trg_v, src_v, missing, prev_ladj)
end

function apply_dist_trafo(trg_d::DistributionsAD.TuringDirichlet, src_d::StandardMvUniform, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    trg_v = apply_dist_trafo(trg_d, src_d, src_v, missing).v
    trafo_ladj = - logpdf(trg_d, trg_v)
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end


g_debug = nothing
function _product_dist_trafo_impl(trg_ds, src_ds, src_v::AbstractVector{<:Real}, prev_ladj::Missing)
    global g_debug = (trg_ds=trg_ds, src_ds=src_ds, src_v=src_v, prev_ladj=missing)
    trg_v = fwddiff(apply_dist_trafo_noladj).(trg_ds, src_ds, src_v)
    var_trafo_result(trg_v, src_v, missing, missing)
end

function _product_dist_trafo_impl(trg_ds, src_ds, src_v::AbstractVector{<:Real}, prev_ladj::Real)
    rs = fwddiff(apply_dist_trafo).(trg_ds, src_ds, src_v, zero(prev_ladj))
    trg_v = map(r -> r.v, rs)
    trafo_ladj = sum(map(r -> r.ladj, rs))
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end

function apply_dist_trafo(trg_d::Distributions.Product, src_d::Distributions.Product, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg_d) == eff_ndof(src_d)
    _product_dist_trafo_impl(trg_d.v, src_d.v, src_v, prev_ladj)
end

function apply_dist_trafo(trg_d::StandardMvUniform, src_d::Distributions.Product, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg_d) == eff_ndof(src_d)
    _product_dist_trafo_impl(StandardUvUniform(), src_d.v, src_v, prev_ladj)
end

function apply_dist_trafo(trg_d::StandardMvNormal, src_d::Distributions.Product, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg_d) == eff_ndof(src_d)
    _product_dist_trafo_impl(StandardUvNormal(), src_d.v, src_v, prev_ladj)
end

function apply_dist_trafo(trg_d::Distributions.Product, src_d::StandardMvUniform, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg_d) == eff_ndof(src_d)
    _product_dist_trafo_impl(trg_d.v, StandardUvUniform(), src_v, prev_ladj)
end

function apply_dist_trafo(trg_d::Distributions.Product, src_d::StandardMvNormal, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    @_adignore @argcheck eff_ndof(trg_d) == eff_ndof(src_d)
    _product_dist_trafo_impl(trg_d.v, StandardUvNormal(), src_v, prev_ladj)
end


function _ntdistelem_to_stdmv(trg_d::StdMvDist, sd::Distribution, src_v_unshaped::AbstractVector{<:Real}, trg_acc::ValueAccessor, init_ladj::OptionalLADJ)
    td = view(trg_d, ValueShapes.view_range(Base.OneTo(length(trg_d)), trg_acc))
    sv = stripscalar(view(src_v_unshaped, trg_acc))
    apply_dist_trafo(td, sd, sv, init_ladj)
end

function _ntdistelem_to_stdmv(trg_d::StdMvDist, sd::ConstValueDist, src_v_unshaped::AbstractVector{<:Real}, trg_acc::ValueAccessor, init_ladj::OptionalLADJ)
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

function apply_dist_trafo(trg_d::StdMvDist, src_d::ValueShapes.UnshapedNTD, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    src_vs = varshape(src_d.shaped)
    @argcheck length(src_d) == length(eachindex(src_v))
    trg_accessors = _transformed_ntd_accessors(src_d.shaped)
    init_ladj = ismissing(prev_ladj) ? missing : zero(Float32)
    rs = map((acc, sd) -> _ntdistelem_to_stdmv(trg_d, sd, src_v, acc, init_ladj), trg_accessors, values(src_d.shaped))
    trg_v = vcat(map(r -> r.v, rs)...)
    trafo_ladj = !ismissing(prev_ladj) ? sum(map(r -> r.ladj, rs)) : missing
    var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
end

function apply_dist_trafo(trg_d::StdMvDist, src_d::NamedTupleDist, src_v::Union{NamedTuple,ShapedAsNT}, prev_ladj::OptionalLADJ)
    src_v_unshaped = unshaped(src_v, varshape(src_d))
    apply_dist_trafo(trg_d, unshaped(src_d), src_v_unshaped, prev_ladj)
end

function _stdmv_to_ntdistelem(td::Distribution, src_d::StdMvDist, src_v::AbstractVector{<:Real}, src_acc::ValueAccessor, init_ladj::OptionalLADJ)
    sd = view(src_d, ValueShapes.view_range(Base.OneTo(length(src_d)), src_acc))
    sv = view(src_v, ValueShapes.view_range(axes(src_v, 1), src_acc))
    apply_dist_trafo(td, sd, sv, init_ladj)
end

function _stdmv_to_ntdistelem(td::ConstValueDist, src_d::StdMvDist, src_v::AbstractVector{<:Real}, src_acc::ValueAccessor, init_ladj::OptionalLADJ)
    (v = Bool[], ladj = init_ladj)
end

function apply_dist_trafo(trg_d::ValueShapes.UnshapedNTD, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    trg_vs = varshape(trg_d.shaped)
    @argcheck length(src_d) == length(eachindex(src_v))
    src_accessors = _transformed_ntd_accessors(trg_d.shaped)
    init_ladj = ismissing(prev_ladj) ? missing : zero(Float32)
    rs = map((acc, td) -> _stdmv_to_ntdistelem(td, src_d, src_v, acc, init_ladj), src_accessors, values(trg_d.shaped))
    trg_v_unshaped = vcat(map(r -> unshaped(r.v), rs)...)
    trafo_ladj = !ismissing(prev_ladj) ? sum(map(r -> r.ladj, rs)) : missing
    var_trafo_result(trg_v_unshaped, src_v, trafo_ladj, prev_ladj)
end

function apply_dist_trafo(trg_d::NamedTupleDist, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    unshaped_result = apply_dist_trafo(unshaped(trg_d), src_d, src_v, prev_ladj)
    (v = strip_shapedasnt(varshape(trg_d)(unshaped_result.v)), ladj = unshaped_result.ladj)
end


const AnyReshapedDist = Union{ReshapedDist,MatrixReshaped}

function apply_dist_trafo(trg_d::Distribution{Multivariate}, src_d::AnyReshapedDist, src_v::Any, prev_ladj::OptionalLADJ)
    src_vs = varshape(src_d)
    @argcheck length(trg_d) == totalndof(src_vs)
    apply_dist_trafo(trg_d, unshaped(src_d), unshaped(src_v, src_vs), prev_ladj)
end

function apply_dist_trafo(trg_d::AnyReshapedDist, src_d::Distribution{Multivariate}, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    trg_vs = varshape(trg_d)
    @argcheck totalndof(trg_vs) == length(src_d)
    r = apply_dist_trafo(unshaped(trg_d), src_d, src_v, prev_ladj)
    (v = trg_vs(r.v), ladj = r.ladj)
end

function apply_dist_trafo(trg_d::AnyReshapedDist, src_d::AnyReshapedDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    trg_vs = varshape(trg_d)
    src_vs = varshape(src_d)
    @argcheck totalndof(trg_vs) == totalndof(src_vs)
    r = apply_dist_trafo(unshaped(trg_d), unshaped(src_d), unshaped(src_v, src_vs), prev_ladj)
    (v = trg_vs(r.v), ladj = r.ladj)
end


function apply_dist_trafo(trg_d::StdMvDist, src_d::UnshapedHDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    src_v_primary, src_v_secondary = _hd_split(src_d, src_v)
    trg_d_primary = typeof(trg_d)(length(eachindex(src_v_primary)))
    trg_d_secondary = typeof(trg_d)(length(eachindex(src_v_secondary)))
    trg_v_primary, ladj_primary = apply_dist_trafo(trg_d_primary, _hd_pridist(src_d), src_v_primary, prev_ladj)
    trg_v_secondary, ladj = apply_dist_trafo(trg_d_secondary, _hd_secdist(src_d, src_v_primary), src_v_secondary, ladj_primary)
    trg_v = vcat(trg_v_primary, trg_v_secondary)
    (v = trg_v, ladj = ladj)
end

function apply_dist_trafo(trg_d::StdMvDist, src_d::HierarchicalDistribution, src_v::Any, prev_ladj::OptionalLADJ)
    src_v_unshaped = unshaped(src_v, varshape(src_d))
    apply_dist_trafo(trg_d, unshaped(src_d), src_v_unshaped, prev_ladj)
end

function apply_dist_trafo(trg_d::UnshapedHDist, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    src_v_primary, src_v_secondary = _hd_split(trg_d, src_v)
    src_d_primary = typeof(src_d)(length(eachindex(src_v_primary)))
    src_d_secondary = typeof(src_d)(length(eachindex(src_v_secondary)))
    trg_v_primary, ladj_primary = apply_dist_trafo(_hd_pridist(trg_d), src_d_primary, src_v_primary, prev_ladj)
    trg_v_secondary, ladj = apply_dist_trafo(_hd_secdist(trg_d, trg_v_primary), src_d_secondary, src_v_secondary, ladj_primary)
    trg_v = vcat(trg_v_primary, trg_v_secondary)
    (v = trg_v, ladj = ladj)
end

function apply_dist_trafo(trg_d::HierarchicalDistribution, src_d::StdMvDist, src_v::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    unshaped_result = apply_dist_trafo(unshaped(trg_d), src_d, src_v, prev_ladj)
    (v = varshape(trg_d)(unshaped_result.v), ladj = unshaped_result.ladj)
end
