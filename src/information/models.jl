
struct InformationState{T <: Number} <: AbstractStateEstimate
    η::Vector{T}  # information vector   (Λ * μ)
    Λ::Matrix{T}  # information matrix   (Σ⁻¹)
end