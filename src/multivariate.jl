# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).


function apply_dist_trafo(trg::Distribution{Multivariate}, src::MatrixReshaped, x::Any, prev_ladj::OptionalLADJ)
    src_vs = varshape(src)
    @argcheck length(trg) == totalndof(src_vs)
    apply_dist_trafo(trg, unshaped(src), unshaped(x, src_vs), prev_ladj)
end

function apply_dist_trafo(trg::MatrixReshaped, src::Distribution{Multivariate}, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    trg_vs = varshape(trg)
    @argcheck totalndof(trg_vs) == length(src)
    r = apply_dist_trafo(unshaped(trg), src, x, prev_ladj)
    (v = trg_vs(r.v), ladj = r.ladj)
end

function apply_dist_trafo(trg::MatrixReshaped, src::MatrixReshaped, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    trg_vs = varshape(trg)
    src_vs = varshape(src)
    @argcheck totalndof(trg_vs) == totalndof(src_vs)
    r = apply_dist_trafo(unshaped(trg), unshaped(src), unshaped(x, src_vs), prev_ladj)
    (v = trg_vs(r.v), ladj = r.ladj)
end



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



#= !!!!!!!!!!!!!!! Move to ValueShapes !!!!!!!!!!!!!!!!!!!!!!!
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
    (v = varshape(trg)(unshaped_result.v), ladj = unshaped_result.ladj)
end


function apply_dist_trafo(trg::Distribution{Multivariate}, src::ReshapedDist, x::Any, prev_ladj::OptionalLADJ)
    src_vs = varshape(src)
    @argcheck length(trg) == totalndof(src_vs)
    apply_dist_trafo(trg, unshaped(src), unshaped(x, src_vs), prev_ladj)
end

function apply_dist_trafo(trg::ReshapedDist, src::Distribution{Multivariate}, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
    trg_vs = varshape(trg)
    @argcheck totalndof(trg_vs) == length(src)
    r = apply_dist_trafo(unshaped(trg), src, x, prev_ladj)
    (v = trg_vs(r.v), ladj = r.ladj)
end

function apply_dist_trafo(trg::ReshapedDist, src::ReshapedDist, x::AbstractVector{<:Real}, prev_ladj::OptionalLADJ)
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

=#
