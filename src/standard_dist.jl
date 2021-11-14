# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).


"""
    struct StandardDist{D<:Distribution{Univariate,Continuous},T<:Real,N} <: Distributions.Distribution{ArrayLikeVariate{N},Continuous}

Represents `D()` or a product distribution of `D()` in a dispatchable fashion.

Constructor:
```
    StandardDist{Uniform}(size...)
    StandardDist{Normal}(size...)
```
"""
struct StandardDist{D<:Distribution{Univariate,Continuous},T<:Real,N} <: Distributions.Distribution{ArrayLikeVariate{N},Continuous}
    _size::Dims{N}
end
export StandardDist

const StandardUnivariateDist{D<:Distribution{Univariate,Continuous},T<:Real} = StandardDist{D,T,0}
const StandardMultivariteDist{D<:Distribution{Multivariate,Continuous},T<:Real} = StandardDist{D,T,1}


StandardDist{D,T,N}(dims::Vararg{Integer,N}) where {D<:Distribution{Univariate,Continuous},N,T<:Real} = StandardDist{D,T,N}(dims)
StandardDist{D,T}(dims::Vararg{Integer,N}) where {D<:Distribution{Univariate,Continuous},N,T<:Real} = StandardDist{D,T,N}(dims)
StandardDist{D}(dims::Vararg{Integer,N}) where {D<:Distribution{Univariate,Continuous},N} = StandardDist{D,Float64}(dims...)


function Base.show(io::IO, d::StandardDist{D,T}) where {D,T}
    print(io, nameof(typeof(d)), "{", D, ",", T, "}")
    show(io, d._size)
end


@inline nonstddist(d::StandardDist{D,T,0}) where {D,T} = D(map(T, params(D()))...)


(::Type{D}, d::StandardDist{D,T,0}) where {D<:Distribution{Univariate,Continuous},T} = nonstddist(d)

# TODO: Replace `fill` by `FillArrays.Fill` once Distributions fully supports this:
(::Type{Distributions.Product})(d::StandardDist{D,T,1}) where {D,T<:Real} = Distributions.Product(fill(StandardDist{D,T}(), length(d)))

Base.convert(::Type{D}, d::StandardDist{D,T,0}) where {D<:Distribution{Univariate,Continuous},T<:Real} = D(d)
Base.convert(::Type{Distributions.Product}, d::StandardDist{D,T,1}) where {D,T<:Real} = Distributions.Product(d)


function _checkvarsize(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{T,N}) where {T,N}
    size(d) == size(x) || throw(DimensionMismatch("Size of variate doesn't match distribution"))
end

ChainRulesCore.rrule(::typeof(_checkvarsize), d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{T,N}) where {T,N} = _checkvarsize(d, x), _nogradient_pullback2



@inline Base.size(d::StandardDist) = d._size
@inline Base.length(d::StandardDist) = prod(size(d))

Base.eltype(::Type{StandardDist{D,T,N}}) where {D,T,N} = T

@inline Distributions.partype(d::StandardDist{D,T}) where {D,T} = T

@inline StatsBase.params(d::StandardDist) = ()

function Distributions.insupport(d::StandardDist{D,T,N}, x::AbstractArray{U,N}) where {D,T,N,U<:Real}
    _checkvarsize(d, x)
    all(xi -> insupport(StandardDist{D,T}(), xi), x)
end

for f in (:(Base.minimum), :(Base.maximum), :(Statistics.mean), :(StatsBase.mode), :(Statistics.var), :(Statistics.std))
    @eval begin
        ($f)(d::StandardDist{D,T,0}) where {D,T} = ($f)(nonstddist(d))
        ($f)(d::StandardDist{D,T,N}) where {D,T,N} = Fill(($f)(StandardDist{D,T}()), size(d)...)
    end
end

StatsBase.modes(d::StandardDist) = [mode(d)]

# ToDo: Define cov for N!=1?
Statistics.cov(d::StandardDist{D,T,1}) where {D,T} = Diagonal(var(d))
Distributions.invcov(d::StandardDist{D,T,1}) where {D,T} = Diagonal(Fill(inv(var(StandardDist{D,T}())), length(d)))
Distributions.logdetcov(d::StandardDist{D,T,1}) where {D,T} = log(var(StandardDist{D,T}())) + length(d)

StatsBase.entropy(d::StandardDist{D,T,0}) where {D,T} = entropy(nonstddist(d))
StatsBase.entropy(d::StandardDist{D,T,N}) where {D,T,N} = length(d) * entropy(StandardDist{D,T}())


Distributions.insupport(d::StandardDist{D,T,0}, x::Real) where {D,T} = insupport(nonstddist(d), x)

function Distributions.insupport(d::StandardDist{D,T,N}, x::AbstractArray{T,N}) where {D,T,N}
    _checkvarsize(d, x)
    all(Base.Fix1(insupport, StandardDist{D,T}()), x)
end

@inline Distributions.logpdf(d::StandardDist{D,T,0}, x::U) where {D,T,U} = logpdf(nonstddist(d), x)

function Distributions.logpdf(d::StandardDist{D,T}, x::AbstractVector{U}) where {D,T,U<:Real}
    _checkvarsize(d, x)
    Distributions._logpdf(d, x)
end

function Distributions._logpdf(d::StandardDist{D,T}, x::AbstractArray{U}) where {D,T,U<:Real}
    sum(x_i -> logpdf(StandardDist{D,T,0}(), x_i), x)
end

function Distributions.logpdf(d::StandardDist{D,T,N}, x::AbstractArray{U,N}) where {D,T,N,U<:Real}
    _checkvarsize(d, x)
    Distributions._logpdf(d, x)
end

function Distributions._logpdf(d::StandardDist{D,T,N}, x::AbstractArray{U,N}) where {D,T,N,U<:Real}
    sum(x_i -> logpdf(StandardDist{D,T,0}(), x_i), x)
end


Distributions.gradlogpdf(d::StandardDist{D,T,0}, x::Real) where {D,T} = gradlogpdf(nonstddist(d), x)

function Distributions.gradlogpdf(d::StandardDist{D,T,N}, x::AbstractArray{<:Real}) where {D,T,N}
    _checkvarsize(d, x)
    gradlogpdf.(StandardDist{D,T,0}(), x)
end


#@inline Distributions.pdf(d::StandardDist{D,T,0}, x::U) where {D,T,U} = pdf(nonstddist(d), x)

function Distributions.pdf(d::StandardDist{D,T}, x::AbstractVector{U}) where {D,T,U<:Real}
    _checkvarsize(d, x)
    Distributions._pdf(d, x)
end

function Distributions._pdf(d::StandardDist{D,T}, x::AbstractVector{U}) where {D,T,U<:Real}
    exp(Distributions._logpdf(d, x))
end

function Distributions.pdf(d::StandardDist{D,T,N}, x::AbstractArray{U,N}) where {D,T,N,U<:Real}
    _checkvarsize(d, x)
    Distributions._pdf(d, x)
end

function Distributions._pdf(d::StandardDist{D,T,N}, x::AbstractArray{U,N}) where {D,T,N,U<:Real}
    exp(Distributions._logpdf(d, x))
end


@inline Distributions.logcdf(d::StandardDist, x::Real) = logcdf(nonstddist(d), x)
@inline Distributions.cdf(d::StandardDist, x::Real) = cdf(nonstddist(d), x)
@inline Distributions.logccdf(d::StandardDist, x::Real) = logccdf(nonstddist(d), x)
@inline Distributions.ccdf(d::StandardDist, x::Real) = ccdf(nonstddist(d), x)
@inline Distributions.quantile(d::StandardDist, p::Real) = quantile(nonstddist(d), p)
@inline Distributions.cquantile(d::StandardDist, p::Real) = cquantile(nonstddist(d), p)
@inline Distributions.mgf(d::StandardDist, t::Real) = mgf(nonstddist(d), t)
@inline Distributions.cf(d::StandardDist, t::Real) = cf(nonstddist(d), t)


Base.rand(rng::AbstractRNG, d::StandardDist{D,T,0}) where {D,T} = rand(rng, nonstddist(d))

function Distributions._rand!(rng::AbstractRNG, d::StandardDist{D,T,N}, A::AbstractArray{U,N}) where {D,T,N,U<:Real}
    broadcast!(() -> rand(rng, StandardDist{D,U}()), A)
end


Distributions.truncated(d::StandardDist{D,T,0}, l::Real, u::Real) where {D,T} = truncated(nonstddist(d), l, u)

Distributions.product_distribution(dists::AbstractVector{StandardDist{D,T,0}}) where {D,T} = StandardDist{D,T}(size(dists)...)
Distributions.product_distribution(dists::AbstractArray{StandardDist{D,T,0}}) where {D,T} = StandardDist{D,T}(size(dists)...)
