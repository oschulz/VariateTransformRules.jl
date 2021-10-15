# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).

function _product_dist_trafo_impl(trg_ds, src_ds, x::AbstractVector{<:Real}, prev_ladj::Missing)
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
