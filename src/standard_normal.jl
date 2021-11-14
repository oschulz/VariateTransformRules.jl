# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).

const StandardNormal{T<:Real} = StandardDist{Normal,T,0}
const StandardMvNormal{T<:Real} = StandardDist{Normal,T,1}


Distributions.MvNormal(d::StandardDist{Normal,T,1}) where T = MvNormal(ScalMat(length(d), one(T)))
Base.convert(::Type{Distributions.MvNormal}, d::StandardDist{Normal,T,1}) where T = MvNormal(d)

Base.minimum(d::StandardDist{Normal,T,0}) where T = convert(float(T), -Inf)
Base.maximum(d::StandardDist{Normal,T,0}) where T = convert(float(T), +Inf)

Distributions.insupport(d::StandardDist{Normal,T,0}, x::Real) where T = !isnan(x)

Distributions.location(d::StandardDist{Normal,T,0}) where T = mean(d)
Distributions.scale(d::StandardDist{Normal,T,0}) where T = var(d)

Statistics.mean(d::StandardDist{Normal,T,0}) where T = zero(T)
Statistics.mean(d::StandardDist{Normal,T,N}) where {T,N} = FillArrays.Zeros{T}(size(d)...)

StatsBase.median(d::StandardDist{Normal}) = mean(d)
StatsBase.mode(d::StandardDist{Normal}) = mean(d)

StatsBase.modes(d::StandardDist{Normal,T,0}) where T = FillArrays.Zeros{T}(1)

Statistics.var(d::StandardDist{Normal,T,0}) where T = one(T)
Statistics.var(d::StandardDist{Normal,T,N}) where {T,N} = FillArrays.Ones{T}(size(d)...)

StatsBase.std(d::StandardDist{Normal,T,0}) where T = one(T)
StatsBase.std(d::StandardDist{Normal,T,N}) where {T,N} = FillArrays.Ones{T}(size(d)...)

StatsBase.skewness(d::StandardDist{Normal,T,0}) where T = zero(T)
StatsBase.kurtosis(d::StandardDist{Normal,T,0}) where T = zero(T)

StatsBase.entropy(d::StandardDist{Normal,T,0}) where T = muladd(log2π, one(T)/2, one(T)/2)

Distributions.logpdf(d::StandardDist{Normal,T,0}, x::U) where {T,U<:Real} = muladd(abs2(x), -U(1)/U(2), -log2π/U(2))
Distributions.pdf(d::StandardDist{Normal,T,0}, x::U) where {T,U<:Real} = invsqrt2π * exp(-abs2(x)/U(2))

@inline Distributions.gradlogpdf(d::StandardDist{Normal,T,0}, x::Real) where T = -x

@inline Distributions.logcdf(d::StandardDist{Normal,<:Real,0}, x::Real) = StatsFuns.normlogcdf(x)
@inline Distributions.cdf(d::StandardDist{Normal,<:Real,0}, x::Real) = StatsFuns.normcdf(x)
@inline Distributions.logccdf(d::StandardDist{Normal,<:Real,0}, x::Real) = StatsFuns.normlogccdf(x)
@inline Distributions.ccdf(d::StandardDist{Normal,<:Real,0}, x::Real) = StatsFuns.normccdf(x)
@inline Distributions.quantile(d::StandardDist{Normal,<:Real,0}, p::Real) = StatsFuns.norminvcdf(p)
@inline Distributions.cquantile(d::StandardDist{Normal,<:Real,0}, p::Real) = StatsFuns.norminvccdf(p)
@inline Distributions.invlogcdf(d::StandardDist{Normal,<:Real,0}, p::Real) = StatsFuns.norminvlogcdf(p)
@inline Distributions.invlogccdf(d::StandardDist{Normal,<:Real,0}, p::Real) = StatsFuns.norminvlogccdf(p)

Base.rand(rng::AbstractRNG, d::StandardDist{Normal,T,N}) where {T,N} = randn(rng, float(T), size(d)...)

Distributions.invcov(d::StandardDist{Normal,T,1}) where T = cov(d)
Distributions.logdetcov(d::StandardDist{Normal,T,1}) where T = zero(T)


function Distributions.sqmahal(d::StandardDist{Normal,T,N}, x::AbstractArray{<:Real,N}) where {T,N}
    _checkvarsize(d, x)
    dot(x, x)
end

function Distributions. sqmahal!(r::AbstractVector, d::StandardDist{Normal,T,N}, x::AbstractMatrix) where {T,N}
    _checkvarsize(d, first(eachcol(x)))
    r .= dot.(eachcol(x), eachcol(x))
end
