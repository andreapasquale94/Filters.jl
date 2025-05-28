# ——————————————————————————————————————————————————————————————————————————————————————————
# State estimate model
# ------------------------------------------------------------------------------------------ 

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
    μ = estimate(est)
    X = est.p .- μ
    return X * Diagonal(est.w) * X'
end

@inline covariance!(out, est::ParticleState) = out .= covariance(est) # TODO: improve

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

# ——————————————————————————————————————————————————————————————————————————————————————————
# Likelihood model API
# ------------------------------------------------------------------------------------------

"""
    AbstractLikelihoodModel 

Abstract type for likelihood models.
"""
abstract type AbstractLikelihoodModel <: AbstractModel end

"""
    likelihood(m::AbstractLikelihoodModel, x, z)

Compute the likelihood of the observation `z` given the state `x` under model `m`.
"""
function likelihood(m::AbstractLikelihoodModel, x, z; kwargs...)
    throw(MethodError(likelihood, (m, x, z)))
end
