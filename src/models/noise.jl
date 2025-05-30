"""
    GaussianWhiteNoise{T}

Gaussian white noise with covariance `P` and cholesky, lower triangular matrix `L`.
"""
struct GaussianWhiteNoise{T <: Number} <: AbstractWhiteNoiseModel
    P::Matrix{T}
    L::LowerTriangular{T}
    function GaussianWhiteNoise(Σ::Matrix{T}) where {T}
        return new{T}(Σ, cholesky(Σ).L)
    end
end

@inline covariance(noise::GaussianWhiteNoise) = noise.P

@inline LinearAlgebra.cholesky(noise::GaussianWhiteNoise) = noise.L