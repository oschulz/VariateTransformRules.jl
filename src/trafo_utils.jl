# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).


"""
    firsttype(::Type{T}, ::Type{U}) where {T<:Real,U<:Real}

Return the first type, but as a dual number type if the second one is dual.

If `U <: ForwardDiff.Dual{tag,<:Real,N}`, returns `ForwardDiff.Dual{tag,T,N}`,
otherwise returns `T`
"""
function firsttype end

firsttype(::Type{T}, ::Type{U}) where {T<:Real,U<:Real} = T
firsttype(::Type{T}, ::Type{<:ForwardDiff.Dual{tag,<:Real,N}}) where {T<:Real,tag,N} = ForwardDiff.Dual{tag,T,N}


"""
    dconvert(::Type{T}, x::Real)
    dconvert(::Type{T}, x::NTuple{N,Real})
    dconvert(::Type{T}, x::AbstractArray{<:Real})

Convert `x`, resp. the elements of `x`, to type T, preserving
dual numbers.

If `x` or it's elements are have type `ForwardDiff.Dual{tag,U}` the return
value or it's elements will have type `ForwardDiff.Dual{tag,T}`, if not,
then just `T`.
"""
function dconvert end

dconvert(::Type{T}, x::T) where {T<:Real} = x
dconvert(::Type{T}, x::Real) where {T<:Real} = convert(T, x)
function dconvert(::Type{T}, x::ForwardDiff.Dual{tag,<:Real}) where {T<:Real,tag}
    ForwardDiff.Dual{tag}(dconvert(T, x.value), dconvert(T, x.partials.values))
end

dconvert(::Type{T}, x::NTuple{N,T}) where {T<:Real,N} = x
dconvert(::Type{T}, x::NTuple{N,<:Real}) where {T<:Real,N} = map(Base.Fix1(convert, T), x)
dconvert(::Type{T}, x::NTuple{N,ForwardDiff.Dual{tag,T}}) where {T<:Real,tag,N} = x
function dconvert(::Type{T}, x::NTuple{N,ForwardDiff.Dual{tag,<:Real}}) where {T<:Real,tag,N}
    map(Base.Fix1(dconvert, T), x)
end

dconvert(::Type{T}, x::AbstractArray{T}) where {T<:Real} = x
dconvert(::Type{T}, x::AbstractArray{<:Real}) where {T<:Real} = map(Base.Fix1(convert, T), x)
dconvert(::Type{T}, x::AbstractArray{ForwardDiff.Dual{tag,T}}) = x
function dconvert(::Type{T}, x::AbstractArray{ForwardDiff.Dual{tag,<:Real}}) where {T<:Real,tag}
    map(Base.Fix1(dconvert, T), x)
end


_nogradient_pullback1(ΔΩ) = (NoTangent(), ZeroTangent())
_nogradient_pullback2(ΔΩ) = (NoTangent(), ZeroTangent(), ZeroTangent())


_adignore(f) = f()

function ChainRulesCore.rrule(::typeof(_adignore), f)
    result = _adignore(f)
    return result, _nogradient_pullback1
end

macro _adignore(expr)
    :(_adignore(() -> $(esc(expr))))
end


function _pushfront(v::AbstractVector, x)
    T = promote_type(eltype(v), typeof(x))
    r = similar(v, T, length(eachindex(v)) + 1)
    r[firstindex(r)] = x
    r[firstindex(r)+1:lastindex(r)] = v
    r
end

function ChainRulesCore.rrule(::typeof(_pushfront), v::AbstractVector, x)
    result = _pushfront(v, x)
    function _pushfront_pullback(thunked_ΔΩ)
        ΔΩ = ChainRulesCore.unthunk(thunked_ΔΩ)
        (NoTangent(), ΔΩ[firstindex(ΔΩ)+1:lastindex(ΔΩ)], ΔΩ[firstindex(ΔΩ)])
    end
    return result, _pushfront_pullback
end


function _pushback(v::AbstractVector, x)
    T = promote_type(eltype(v), typeof(x))
    r = similar(v, T, length(eachindex(v)) + 1)
    r[lastindex(r)] = x
    r[firstindex(r):lastindex(r)-1] = v
    r
end

function ChainRulesCore.rrule(::typeof(_pushback), v::AbstractVector, x)
    result = _pushback(v, x)
    function _pushback_pullback(thunked_ΔΩ)
        ΔΩ = ChainRulesCore.unthunk(thunked_ΔΩ)
        (NoTangent(), ΔΩ[firstindex(ΔΩ):lastindex(ΔΩ)-1], ΔΩ[lastindex(ΔΩ)])
    end
    return result, _pushback_pullback
end


_dropfront(v::AbstractVector) = v[firstindex(v)+1:lastindex(v)]

_dropback(v::AbstractVector) = v[firstindex(v):lastindex(v)-1]


_rev_cumsum(xs::AbstractVector) = reverse(cumsum(reverse(xs)))

function ChainRulesCore.rrule(::typeof(_rev_cumsum), xs::AbstractVector)
    result = _rev_cumsum(xs)
    function _rev_cumsum_pullback(ΔΩ)
        ∂xs = ChainRulesCore.@thunk cumsum(ChainRulesCore.unthunk(ΔΩ))
        (NoTangent(), ∂xs)
    end
    return result, _rev_cumsum_pullback
end


# Equivalent to `cumprod(xs)``:
_exp_cumsum_log(xs::AbstractVector) = exp.(cumsum(log.(xs)))

function ChainRulesCore.rrule(::typeof(_exp_cumsum_log), xs::AbstractVector)
    result = _exp_cumsum_log(xs)
    function _exp_cumsum_log_pullback(ΔΩ)
        ∂xs = inv.(xs) .* _rev_cumsum(exp.(cumsum(log.(xs))) .* ChainRulesCore.unthunk(ΔΩ))
        (NoTangent(), ∂xs)
    end
    return result, _exp_cumsum_log_pullback
end
