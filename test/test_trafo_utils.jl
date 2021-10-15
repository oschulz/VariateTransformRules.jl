# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).

using VariateTransformations
using Test

using LinearAlgebra
using ValueShapes, Distributions, ArraysOfArrays
using ForwardDiff, Zygote, DistributionsAD


@testset "trafo_utils" begin
    xs = rand(5)
    @test Zygote.jacobian(VariateTransformations._pushfront, xs, 42)[1] ≈ ForwardDiff.jacobian(xs -> VariateTransformations._pushfront(xs, 1), xs)
    @test Zygote.jacobian(VariateTransformations._pushfront, xs, 42)[2] ≈ vec(ForwardDiff.jacobian(x -> VariateTransformations._pushfront(xs, x[1]), [42]))
    @test Zygote.jacobian(VariateTransformations._pushback, xs, 42)[1] ≈ ForwardDiff.jacobian(xs -> VariateTransformations._pushback(xs, 1), xs)
    @test Zygote.jacobian(VariateTransformations._pushback, xs, 42)[2] ≈ vec(ForwardDiff.jacobian(x -> VariateTransformations._pushback(xs, x[1]), [42]))
    @test Zygote.jacobian(VariateTransformations._rev_cumsum, xs)[1] ≈ ForwardDiff.jacobian(VariateTransformations._rev_cumsum, xs)
    @test Zygote.jacobian(VariateTransformations._exp_cumsum_log, xs)[1] ≈ ForwardDiff.jacobian(VariateTransformations._exp_cumsum_log, xs) ≈ ForwardDiff.jacobian(cumprod, xs)
end
