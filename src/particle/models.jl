# State 

"""
    ParticleState{T} <: AbstractStateEstimate

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

"""
    covariance(est::ParticleState)

Return the weighted covariance of the particles.
"""
function covariance(est::ParticleState)
    μ = estimate(est)
    X = est.p .- μ
    return X * Diagonal(est.w) * X'
end

function normalize!(s::ParticleState{T <: Number}) where {T}
    wt = sum(s.w)
    if wt == 0
        # Avoid division by zero: reset to uniform weights
        fill!(s.w, 1 / length(s))
    else
        s.w ./= wt
    end
    nothing
end

@inline Base.length(s::ParticleState) = length(s.w)

@inline effective_samples(s::ParticleState) = 1 / sum(s.w .^ 2)