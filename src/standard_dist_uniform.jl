# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).

const StandardUvUniform{T<:Real} = StandardDist{Uniform,T,0}
const StandardMvUniform{T<:Real} = StandardDist{Uniform,T,1}


Base.minimum(::StandardDist{Uniform,T,0}) where T = zero(T)
Base.maximum(::StandardDist{Uniform,T,0}) where T = one(T)

StatsBase.params(::StandardDist{Uniform,T,0}) where T = (zero(T), one(T))

Distributions.location(::StandardDist{Uniform,T,0}) where T = zero(T)
Distributions.scale(::StandardDist{Uniform,T,0}) where T = one(T)


Statistics.mean(d::StandardDist{Uniform,T,0}) where T = convert(float(T), 1//2)
StatsBase.median(d::StandardDist{Uniform,T,0}) where T = mean(d)
StatsBase.mode(d::StandardDist{Uniform,T,0}) where T = mean(d)
StatsBase.modes(d::StandardDist{Uniform,T,0}) where T = Fill(mode(d), 0)

Statistics.var(d::StandardDist{Uniform,T,0}) where T = convert(float(T), 1//12)
StatsBase.std(d::StandardDist{Uniform,T,0}) where T = sqrt(var(d))
StatsBase.skewness(d::StandardDist{Uniform,T,0}) where T = zero(T)
StatsBase.kurtosis(d::StandardDist{Uniform,T,0}) where T = convert(T, -6//5)

StatsBase.entropy(d::StandardDist{Uniform,T,0}) where T = zero(T)


function Distributions.logpdf(d::StandardDist{Uniform,T,0}, x::U) where {T,U<:Real}
    R = float(promote_type(T,U))
    ifelse(insupport(d, x), R(0), R(-Inf))
end

function Distributions.pdf(d::StandardDist{Uniform,T,0}, x::U) where {T,U<:Real}
    R = promote_type(T,U)
    ifelse(insupport(d, x), one(R), zero(R))
end

Distributions.logcdf(d::StandardUvUniform, x::Real) = log(cdf(d, x))

function Distributions.cdf(d::StandardDist{Uniform,T,0}, x::U) where {T,U<:Real}
    R = promote_type(T,U)
    ifelse(x < zero(U), zero(R), ifelse(x < one(U), x, one(R)))
end

Distributions.logccdf(d::StandardUvUniform, x::Real) = log(ccdf(d, x))

function Distributions.ccdf(d::StandardDist{Uniform,T,0}, x::U) where {T,U<:Real}
    y = cdf(d, x)
    one(y) - y
end

function Distributions.quantile(d::StandardDist{Uniform,T,0}, p::U) where {T,U<:Real}
   R = promote_type(T,U)
   convert(float(R), p)
end

function Distributions.cquantile(d::StandardDist{Uniform,T,0}, p::U) where {T,U<:Real}
    y = quantile(d, p)
    one(y) - y
end

Distributions.mgf(d::StandardDist{Uniform,T,0}, t::Real) where T = mgf(nonstddist(d), t)
Distributions.cf(d::StandardDist{Uniform,T,0}, t::Real) where T = cf(nonstddist(d), t)

# Distributions doesn't seem to support gradlogpdf for Uniform()
# # Distributions.gradlogpdf(d::StandardUvUniform, x::Real) = zero(T)

Base.rand(rng::AbstractRNG, d::StandardDist{Uniform,T,0}) where T = rand(rng, float(T))





# Not applicable:
# # StatsBase.params(d::StandardMvUniform)
# # @inline Distributions.partype(d::StandardDist{Uniform,T,N})


StatsBase.mode(d::StandardMvUniform) = mean(d)
StatsBase.modes(d::StandardMvUniform) = Fill(mean(d), 0)

Distributions.invcov(d::StandardDist{Uniform,T,N}) where T = Diagonal(Fill(inv(var(StandardDist{Uniform,T,0})), length(d)))
Distributions.logdetcov(d::StandardDist{Uniform,T,N}) where T = logdet(cov(d))

StatsBase.entropy(d::StandardDist{Uniform,T,N}) where T = zero(T)


function Distributions._logpdf(d::StandardDist{Uniform,T,N}, x::AbstractVector{U}) where {T,U<:Real}
    R = float(promote_type(T,U))
    ifelse(insupport(d, x), R(0), R(-Inf))
end

function Distributions._pdf(d::StandardDist{Uniform,T,N}, x::AbstractVector{U}) where {T,U<:Real}
    R = promote_type(T,U)
    ifelse(insupport(d, x), one(R), zero(R))
end

# Not useful:
# # Distributions.sqmahal(d::StandardMvUniform, x::AbstractVector{<:Real})
# # Distributions.sqmahal!(r::AbstractVector{<:Real}, d::StandardMvUniform, x::AbstractMatrix{<:Real})

function Distributions.gradlogpdf(d::StandardDist{Uniform,T,N}, x::AbstractVector{U}) where {T,U<:Real}
    FillArrays.Zeros(length(d))
end

function Distributions._rand!(rng::AbstractRNG, d::StandardMvUniform, A::AbstractVector{T}) where {T<:Real}
    broadcast!(x -> rand(rng, T), A, A)
end

function Distributions._rand!(rng::AbstractRNG, d::StandardMvUniform, A::AbstractMatrix{T}) where {T<:Real}
    broadcast!(x -> rand(rng, T), A, A)
end


const StandardUniformDist = Union{StandardUvUniform, StandardMvUniform}
