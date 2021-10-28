# This file is a part of VariateTransformations.jl, licensed under the MIT License (MIT).

const StandardUvNormal{T<:Real} = StandardDist{Normal,T,0}
const StandardMvNormal{T<:Real} = StandardDist{Normal,T,1}


Distributions.MvNormal(d::StandardDist{Normal,T,1}) where T = MvNormal(ScalMat(length(d), one(T)))
Base.convert(::Type{Distributions.MvNormal}, d::StandardDist{Normal,T,1}) = MvNormal(d)
