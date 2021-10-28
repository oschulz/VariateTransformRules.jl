# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).


"""
    struct StandardDist{D<:Distribution{Univariate,Continuous},T<:Real,N} <: Distributions.Distribution{ArrayLikeVariate{N},Continuous}

A standard uniform distribution between zero and one.

Constructor:
```
    StandardDist{Uniform}(size...)
    StandardDist{Normal}(size...)
```
"""
struct StandardDist{D<:Distribution{Univariate,Continuous},T<:Real,N} <: Distributions.Distribution{ArrayLikeVariate{N},Continuous}
    _size::Dims{N}
end

const StandardUnivariateDist{D<:Distribution{Univariate,Continuous},T<:Real} = StandardDist{D,T,0}
const StandardMultivariteDist{D<:Distribution{Multivariate,Continuous},T<:Real} = StandardDist{D,T,1}


StandardDist{D,T}(dims::Vararg{Integer,N}) where {D<:Distribution{Univariate,Continuous},N,T<:Real} = StandardDist{D,T,N}(dims)
StandardDist{D}(dims::Vararg{Integer,N}) where {D<:Distribution{Univariate,Continuous},N} = StandardDist{D,Float64}(dims...)


nonstddist(d::StandardDist{D,T,0}) where {D,T} = D()

(::Type{D}, d::StandardDist{D,T,0}) where {D<:Distribution{Univariate,Continuous},T} = nonstddist(d)
(::Type{Distributions.Product})(d::StandardDist{D,T,1}) where {D<:Distribution{S},T<:Real} = Distributions.Product(Fill(StandardDist{D,T}(), length(d)))

Base.convert(::Type{D}, d::StandardDist{D,T,0}) where {D<:Distribution{Univariate,Continuous},T<:Real} = D(d)
Base.convert(::Type{Distributions.Product}, d::StandardDist{D,T,1}) where {D<:Distribution{S},T<:Real} = Distributions.Product(d)



@inline Base.size(d::StandardDist) = d._size
#!!!! necessary? ### @inline Base.length(d::StandardMvNormal{T}) = product(size(d))

Base.eltype(::Type{StandardDist{D,T,N}}) where {D,T,N} = T

# Base.view(d::StandardDist{D,T,N}, idxs::VarArg{Union{Integer,AbstractArray{<:Integer}},N}) where {D,T,N}

@inline Distributions.partype(d::StandardDist{D,T,0}) where {D,T} = T


function Distributions.insupport(d::StandardDist{D,T,N}, x::AbstractArray{U,N}) where {D,T,N,U<:Real}
    size(d) == size(x) && all(xi -> insupport(StandardDist{D,T}(), xi), x)
end


Statistics.mean(d::StandardDist{D,T,0}) where {D,T} = mean(nonstddist(d))
Statistics.mean(d::StandardDist{D,T,N}) where {D,T,N} = Fill(mean(StandardDist{Uniform,T,0}), length(d))

Statistics.var(d::StandardDist{D,T,0}) where {D,T} = var(nonstddist(d))
Statistics.var(d::StandardDist{D,T,N}) where {D,T,N} = Fill(var(StandardDist{Uniform,T,0}), length(d))

# ToDo: Define cov for N!=1?
Statistics.cov(d::StandardDist{D,T,1}) where {D,T} = Diagonal(var(d))


Base.rand(rng::AbstractRNG, d::StandardDist{D,T,0}) where {D,T} = rand(rng, nonstddist(d))

Distributions.truncated(d::StandardUvUniform, l::Real, u::Real) = truncated(nonstddist(d), l, u)

function Distributions.product_distribution(dists::AbstractArray{StandardDist{D,T,0}}) where {D,T}
    StandardDist{D,T}(size(dists)...)
end
