# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).

eff_ndof(d::Dirichlet) = length(d) - 1

intermediate(trg::Dirichlet) = StandardMvUniform(eff_ndof(trg))

function _dirichlet_beta_trafo(α::Real, β::Real, x::Real)
    R = float(promote_type(typeof(α), typeof(β), typeof(x)))
    convert(R, apply_dist_trafo(Beta(α, β), StandardUvUniform(), x, missing).v)::R
end

function transform_variate(trg::DistributionsAD.TuringDirichlet, src::StandardMvUniform, x::AbstractVector{<:Real})
    return to_dirichlet(trg.alpha, src, x)
end

_a_times_one_minus_b(a::Real, b::Real) = a * (1 - b)

function to_dirichlet(alpha::AbstractVector{<:Real}, src::StandardMvUniform, x::AbstractVector{<:Real})
    # See M. J. Betancourt, "Cruising The Simplex: Hamiltonian Monte Carlo and the Dirichlet Distribution",
    # https://arxiv.org/abs/1010.3436

    @_adignore @argcheck length(trg) == length(src) + 1
    αs = _dropfront(_rev_cumsum(alpha))
    βs = _dropback(alpha)
    beta_v = fwddiff(_dirichlet_beta_trafo).(αs, βs, x)
    beta_v_cp = _exp_cumsum_log(_pushfront(beta_v, 1))
    beta_v_ext = _pushback(beta_v, 0)
    y = fwddiff(_a_times_one_minus_b).(beta_v_cp, beta_v_ext)
    var_trafo_result(y, x, missing, prev_ladj)
end
