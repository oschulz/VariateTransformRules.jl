# This file is a part of VariateTransformRules.jl, licensed under the MIT License (MIT).

using VariateTransformRules
using Test

using LinearAlgebra
using ValueShapes, Distributions, ArraysOfArrays
using ForwardDiff, Zygote, DistributionsAD


@testset "trafo_utils" begin
    xs = rand(5)
    @test Zygote.jacobian(VariateTransformRules._pushfront, xs, 42)[1] ≈ ForwardDiff.jacobian(xs -> VariateTransformRules._pushfront(xs, 1), xs)
    @test Zygote.jacobian(VariateTransformRules._pushfront, xs, 42)[2] ≈ vec(ForwardDiff.jacobian(x -> VariateTransformRules._pushfront(xs, x[1]), [42]))
    @test Zygote.jacobian(VariateTransformRules._pushback, xs, 42)[1] ≈ ForwardDiff.jacobian(xs -> VariateTransformRules._pushback(xs, 1), xs)
    @test Zygote.jacobian(VariateTransformRules._pushback, xs, 42)[2] ≈ vec(ForwardDiff.jacobian(x -> VariateTransformRules._pushback(xs, x[1]), [42]))
    @test Zygote.jacobian(VariateTransformRules._rev_cumsum, xs)[1] ≈ ForwardDiff.jacobian(VariateTransformRules._rev_cumsum, xs)
    @test Zygote.jacobian(VariateTransformRules._exp_cumsum_log, xs)[1] ≈ ForwardDiff.jacobian(VariateTransformRules._exp_cumsum_log, xs) ≈ ForwardDiff.jacobian(cumprod, xs)
end
