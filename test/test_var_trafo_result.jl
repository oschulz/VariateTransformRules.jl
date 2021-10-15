# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).

using VariateTransformations
using Test

using LinearAlgebra
using ValueShapes, Distributions, ArraysOfArrays
using ForwardDiff, Zygote, DistributionsAD
#using VariateTransformations: var_trafo_result #, var_trafo_ladj


@testset "var_trafo_result" begin
    function test_back_and_forth(trg_d, src_d)
        @testset "transform $(typeof(trg_d).name) <-> $(typeof(src_d).name)" begin
            src_v = rand(src_d)
            prev_ladj = 7.9

            @test @inferred(VariateTransformations.apply_dist_trafo(trg_d, src_d, src_v, prev_ladj)) isa NamedTuple{(:v,:ladj)}
            trg_v, trg_ladj = VariateTransformations.apply_dist_trafo(trg_d, src_d, src_v, prev_ladj)

            @test VariateTransformations.apply_dist_trafo(src_d, trg_d, trg_v, trg_ladj) isa NamedTuple{(:v,:ladj)}
            src_v_reco, prev_ladj_reco = VariateTransformations.apply_dist_trafo(src_d, trg_d, trg_v, trg_ladj)

            @test src_v ≈ src_v_reco
            @test prev_ladj ≈ prev_ladj_reco
            @test trg_ladj ≈ logabsdet(ForwardDiff.jacobian(x -> unshaped(VariateTransformations.apply_dist_trafo(trg_d, src_d, x, prev_ladj).v), src_v))[1] + prev_ladj
        end
    end

    function get_trgxs(trg_d, src_d, X)
        return (x -> VariateTransformations.apply_dist_trafo(trg_d, src_d, x, 0.0).v).(nestedview(X))
    end

    function get_trgxs(trg_d, src_d::Distribution{Univariate}, X)
        return (x -> VariateTransformations.apply_dist_trafo(trg_d, src_d, x, 0.0).v).(X)
    end

    function test_dist_trafo_moments(trg_d, src_d)
        @testset "check moments of trafo $(typeof(trg_d).name) <- $(typeof(src_d).name)" begin
            let trg_d = trg_d, src_d = src_d
                X = flatview(rand(src_d, 10^5))
                trgxs = get_trgxs(trg_d, src_d, X)
                unshaped_trgxs = map(unshaped, trgxs)
                @test isapprox(mean(unshaped_trgxs), mean(unshaped(trg_d)), atol = 0.1)
                @test isapprox(cov(unshaped_trgxs), cov(unshaped(trg_d)), rtol = 0.1)
            end
        end
    end

    uniform1 = Uniform(-5.0, -0.01)
    uniform2 = Uniform(0.01, 5.0)

    normal1 = Normal(-10, 1)
    normal2 = Normal(10, 5)

    uvnorm = VariateTransformations.StandardUvNormal()

    standnorm1 = VariateTransformations.StandardMvNormal(1)
    standnorm2 = VariateTransformations.StandardMvNormal(2)

    standuni2 = VariateTransformations.StandardMvUniform(2)

    standnorm2_reshaped = ReshapedDist(standnorm2, varshape(standnorm2))

    mvnorm = MvNormal([0.3, -2.9], [1.7 0.5; 0.5 2.3])
    beta = Beta(3,1)
    gamma = Gamma(0.1,0.7)
    dirich = Dirichlet([0.1,4])

    ntdist = NamedTupleDist(
        a = uniform1,
        b = mvnorm,
        c = [4.2, 3.7],
        x = beta,
        y = gamma
    )

    test_back_and_forth(beta, standnorm1)
    test_back_and_forth(gamma, standnorm1)

    test_back_and_forth(mvnorm, mvnorm)
    test_back_and_forth(mvnorm, standnorm2)
    test_back_and_forth(mvnorm, standuni2)

    test_dist_trafo_moments(normal2, normal1)
    test_dist_trafo_moments(uniform2, uniform1)

    test_dist_trafo_moments(beta, gamma)
    test_dist_trafo_moments(uvnorm, standnorm1)

    test_dist_trafo_moments(beta, standnorm1)
    test_dist_trafo_moments(gamma, standnorm1)

    test_dist_trafo_moments(mvnorm, standnorm2)
    test_dist_trafo_moments(dirich, standnorm1)

    test_dist_trafo_moments(mvnorm, standuni2)
    test_dist_trafo_moments(standuni2, mvnorm)

    test_dist_trafo_moments(standnorm2, standuni2)

    test_dist_trafo_moments(mvnorm, standnorm2_reshaped)
    test_dist_trafo_moments(standnorm2_reshaped, mvnorm)
    test_dist_trafo_moments(standnorm2, standnorm2_reshaped)
    test_dist_trafo_moments(standnorm2_reshaped, standnorm2_reshaped)
    
    test_back_and_forth(ntdist, VariateTransformations.StandardMvNormal(5))
    test_back_and_forth(ntdist, VariateTransformations.StandardMvUniform(5))

    let
        mvuni = product_distribution([Uniform(), Uniform()])

        x = rand()
        @test_throws ArgumentError VariateTransformations.apply_dist_trafo(uvnorm, mvnorm, x, 0.0)
        @test_throws ArgumentError VariateTransformations.apply_dist_trafo(uvnorm, standnorm1, x, 0.0)
        @test_throws ArgumentError VariateTransformations.apply_dist_trafo(uvnorm, standnorm2, x, 0.0)

        x = rand(2)
        @test_throws ArgumentError VariateTransformations.apply_dist_trafo(mvuni, mvnorm, x, 0.0)
        @test_throws ArgumentError VariateTransformations.apply_dist_trafo(mvnorm, mvuni, x, 0.0)
        @test_throws ArgumentError VariateTransformations.apply_dist_trafo(uvnorm, mvnorm, x, 0.0)
        @test_throws ArgumentError VariateTransformations.apply_dist_trafo(uvnorm, standnorm1, x, 0.0)
        @test_throws ArgumentError VariateTransformations.apply_dist_trafo(uvnorm, standnorm2, x, 0.0)
    end

    let
        primary_dist = NamedTupleDist(x = Normal(2), c = 5)
        f = x -> NamedTupleDist(y = Normal(x.x, 3), z = MvNormal([1.3 0.5; 0.5 2.2]))
        trg_d = @inferred(HierarchicalDistribution(f, primary_dist))
        src_d = VariateTransformations.StandardMvNormal(totalndof(varshape(trg_d)))
        test_back_and_forth(trg_d, src_d)
        test_dist_trafo_moments(trg_d, src_d)
    end


    @testset "Custom cdf and quantile for dual numbers" begin
        Dual = ForwardDiff.Dual

        @test VariateTransformations._trafo_cdf(Normal(Dual(0, 1, 0, 0), Dual(1, 0, 1, 0)), Dual(0.5, 0, 0, 1)) == cdf(Normal(Dual(0, 1, 0, 0), Dual(1, 0, 1, 0)), Dual(0.5, 0, 0, 1))
        @test VariateTransformations._trafo_cdf(Normal(0, 1), Dual(0.5, 1)) == cdf(Normal(0, 1), Dual(0.5, 1))

        @test VariateTransformations._trafo_quantile(Normal(0, 1), Dual(0.5, 1)) == quantile(Normal(0, 1), Dual(0.5, 1))
        @test VariateTransformations._trafo_quantile(Normal(Dual(0, 1, 0, 0), Dual(1, 0, 1, 0)), Dual(0.5, 0, 0, 1)) == quantile(Normal(Dual(0, 1, 0, 0), Dual(1, 0, 1, 0)), Dual(0.5, 0, 0, 1))
    end


    @testset "trafo autodiff pullbacks" begin
        # ToDo: Test for type stability and fix where necessary.

        src_v = [0.6, 0.7, 0.8, 0.9]
        f = inv(VariateTransformations.DistributionTransform(Uniform, DistributionsAD.TuringDirichlet([3.0, 4.0, 5.0, 6.0, 7.0])))
        @test isapprox(ForwardDiff.jacobian(f, src_v), Zygote.jacobian(f, src_v)[1], rtol = 10^-4)
        f = inv(VariateTransformations.DistributionTransform(Uniform, Dirichlet([3.0, 4.0, 5.0, 6.0, 7.0])))
        @test isapprox(ForwardDiff.jacobian(f, src_v), Zygote.jacobian(f, src_v)[1], rtol = 10^-4)
        f = inv(VariateTransformations.DistributionTransform(Normal, Dirichlet([3.0, 4.0, 5.0, 6.0, 7.0])))
        @test isapprox(ForwardDiff.jacobian(f, src_v), Zygote.jacobian(f, src_v)[1], rtol = 10^-4)
    end
end
