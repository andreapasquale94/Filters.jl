"""
    ParticleState{T}

State estimate represented by a weighted ensemble of particles.
"""
struct ParticleState{T <: Number} <: AbstractStateEstimate
    p::Matrix{T}
    w::Vector{T}
end

"""
    estimate(est::ParticleState)

Return the weighted mean of the particles.
"""
@inline function estimate(est::ParticleState)
    return est.p' * est.w
end

@inline estimate!(out, est::ParticleState) = mul!(out, est.p', est.w)

"""
    covariance(est::ParticleState)

Return the weighted covariance of the particles.
"""
function covariance(est::ParticleState)
    X = est.p .- estimate(est)
    return X * Diagonal(est.w) * X'
end

@inline covariance!(out, est::ParticleState) = out .= covariance(est) # TODO: improve

function variance!(out, est::ParticleState; μ = estimate(est))
    fill!(out, 0.0)
    n, N = size(est.p)
    for i in 1:N
        @inbounds @simd for j in 1:n
            δ = est.p[j, i] - μ[j]
            out[j] += est.w[i] * δ * δ
        end
    end
    nothing
end


function variance(est::ParticleState)
    out = zeros(eltype(est.p), size(est.p, 1))
    variance!(out, est)
    return out
end

function skewness!(out, est::ParticleState; μ = estimate(est), σ² = variance(est))
    fill!(out, 0.0)
    n, N = size(est.p)
    for i in 1:N
        @inbounds @simd for j in 1:n
            δ = est.p[j, i] - μ[j]
            out[j] += est.w[i] * (δ * δ * δ)
        end
    end
    ϵ = eps(eltype(out)) # Avoid division by zero
    @inbounds out ./= (σ² .^ (3 / 2)) .+ ϵ
    nothing
end


function skewness(est::ParticleState)
    out = zeros(eltype(est.p), size(est.p, 1))
    σ² = similar(out)
    μ = estimate(est)

    variance!(σ², est, μ = μ)
    skewness!(out, est, μ = μ, σ² = σ²)
    return out
end

"""
    normalize!(s::ParticleState)

Normalize the particles weights.
"""
function LinearAlgebra.normalize!(s::ParticleState{T}) where {T}
    wt = sum(s.w)
    if wt == 0
        # Avoid division by zero: reset to uniform weights
        fill!(s.w, 1 / length(s))
    else
        s.w ./= wt
    end
    nothing
end

"""
    length(s::ParticleState)

Number of particles in the given state.
"""
@inline Base.length(s::ParticleState) = length(s.w)

"""
    neffective(s::ParticleState)

Computes the effective samples size.
"""
@inline neffective(s::ParticleState) = 1 / sum(s.w .^ 2)