
struct InformationState{T <: Number} <: AbstractStateEstimate
    η::Vector{T}  # information vector   (Λ * μ)
    Λ::Matrix{T}  # information matrix   (Σ⁻¹)
end

@inline covariance(s::InformationState) = s.Λ \ I
@inline estimate(s::InformationState) = covariance(s) * s.η

struct SquareRootInformationState{T <: Number} <: AbstractStateEstimate
    η::Vector{T}
    Y::UpperTriangular{T}
end

@inline covariance(s::SquareRootInformationState) = (s.Y' * s.Y) \ I # TODO: improve
@inline estimate(s::SquareRootInformationState) = covariance(s) * s.η