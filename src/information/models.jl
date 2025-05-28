
"""
    InformationState{T}

State estimate represented using information vector an matrix.
"""
struct InformationState{T <: Number} <: AbstractStateEstimate
    η::Vector{T}  # information vector   (Λ * μ)
    Λ::Matrix{T}  # information matrix   (Σ⁻¹)
end

@inline covariance(s::InformationState) = s.Λ \ I

@inline estimate(s::InformationState) = covariance(s) * s.η

@inline estimate!(out, s::InformationState) = mul!(out, covariance(s), s.η)

@inline covariance!(out, s::InformationState) = out .= s.Λ \ I
