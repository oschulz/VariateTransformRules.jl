# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).

using VariateTransformations
using Test

using Random, Statistics, LinearAlgebra
using Distributions, PDMats
using StableRNGs


@testset "standard_dist" begin
    stblrng() = StableRNG(789990641)

    D = Normal
    T = Float64
    dref = Normal()
    sz = ()

    @testset "StandardDist{D,T}(sz...)" begin
        @test @inferred(StandardDist{D}(sz...)) isa StandardDist{D,Float64}
        @test @inferred(StandardDist{D,T}(sz...)) isa StandardDist{D,T}
        @test @inferred(size(StandardDist{D}(sz...))) == size(dref)
        @test @inferred(size(StandardDist{D,T}(sz...))) == size(dref)

        d = StandardDist{D,T}(sz...)

        if length(size(d)) < 3
            @test @inferred(VariateTransformations.nonstddist(d)) == dref
        end

        @test @inferred(minimum(d)) == minimum(dref)
        @test @inferred(maximum(d)) == maximum(dref)
        
        @test @inferred(params(d)) == params(dref)
        @test @inferred(partype(d)) == partype(dref)
        
        @test @inferred(location(d)) == location(dref)
        @test @inferred(scale(d)) == scale(dref)
        
        @test @inferred(eltype(typeof(d))) == eltype(typeof(dref))
        @test @inferred(eltype(d)) == eltype(dref)

        @test @inferred(length(d)) == length(dref)
        @test @inferred(size(d)) == size(dref)

        @test @inferred(mean(d)) == mean(dref)
        @test @inferred(median(d)) == median(dref)
        @test @inferred(mode(d)) == mode(dref)
        @test @inferred(modes(d)) ≈ modes(dref)
        
        @test @inferred(var(d)) == var(dref)
        @test @inferred(std(d)) == std(dref)
        @test @inferred(skewness(d)) == skewness(dref)
        @test @inferred(kurtosis(d)) == kurtosis(dref)
        
        @test @inferred(entropy(d)) == entropy(dref)
        
        xs = [minimum(dref), quantile(dref, 1//3), quantile(dref, 1//2), quantile(dref, 2//3), maximum(dref)]

        for x in xs
            @test @inferred(gradlogpdf(d, x)) == gradlogpdf(dref, x)

            @test @inferred(logpdf(d, x)) == logpdf(dref, x)
            @test @inferred(pdf(d, x)) == pdf(dref, x)
            @test @inferred(logcdf(d, x)) == logcdf(dref, x)
            @test @inferred(cdf(d, x)) == cdf(dref, x)
            @test @inferred(logccdf(d, x)) == logccdf(dref, x)
            @test @inferred(ccdf(d, x)) == ccdf(dref, x)
        end

        for p in [0.0, 0.25, 0.75, 1.0]
            @test @inferred(quantile(d, p)) == quantile(dref, p)
            @test @inferred(cquantile(d, p)) == cquantile(dref, p)
        end

        for t in [-3, 0, 3]
            @test @inferred(mgf(d, t)) == mgf(dref, t)
            @test @inferred(cf(d, t)) == cf(dref, t)
        end

        @test @inferred(rand(stblrng(), d)) == rand(stblrng(), d)
        @test @inferred(rand(stblrng(), d, 5)) == rand(stblrng(), d, 5)

        if size(d) == ()
            @test @inferred(truncated(d, quantile(dref, 1//3), quantile(dref, 2//3))) == truncated(dref, quantile(dref, 1//3), quantile(dref, 2//3))
        end

        if length(size(d)) == 1
            @test @inferred(convert(Distributions.Product, d)) isa Distributions.Product
            d_as_prod = convert(Distributions.Product, d)
            @test d_as_prod.v == fill(StandardDist{D,T}(), size(d)...)
        end

        @test @inferred(rand(stblrng(), d)) == rand(stblrng(), dref)
        @test @inferred(rand(stblrng(), d, 5)) == rand(stblrng(), dref, 5)
        @test @inferred(rand!(stblrng(), d, zeros(size(d)...))) == rand!(stblrng(), dref, zeros(size(dref)...))
        if length(size(d)) == 1
            @test @inferred(rand!(stblrng(), d, zeros(size(d)..., 5))) == rand!(stblrng(), dref, zeros(size(dref)..., 5))
        end
    end

    @testset "StandardDist{Normal}()" begin
        # TODO: Add @inferred
        @test (MvNormal(StandardDist{Normal,Float64}(4))) == MvNormal(fill(1.0, 4))
        @test (Base.convert(MvNormal, StandardDist{Normal,Float64}(4))) == MvNormal(fill(1.0, 4))

        @test @inferred(product_distribution(fill(VariateTransformations.StandardUvNormal{Float32}(), 3))) isa VariateTransformations.StandardMvNormal{Float32}
        @test product_distribution(fill(VariateTransformations.StandardUvNormal{Float32}(), 3)) == VariateTransformations.StandardMvNormal{Float32}(3)
    end
end
