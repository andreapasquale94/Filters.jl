
"""
    BootstrapParticleFilter 

A bootstrap particle filter implementation.
The filter consists of a state estimate, a prediction step, an update step, and a resampling strategy.
It is called "bootstrap" because it uses a resampling step to generate new particles based
on the current state estimate and the likelihood of the observations.
"""
struct BootstrapParticleFilter{
    T <: Number,
    S <: AbstractStateEstimate,
    P <: AbstractFilterPrediction,
    U <: AbstractFilterUpdate,
    R <: AbstractParticleResampling
} <: AbstractParticleFilter{T}
    est::S
    pre::P
    up::U
    resampling::R
    function BootstrapParticleFilter{T}(
        est::S,
        pre::P,
        up::U,
        resampling::R
    ) where {T, S, P, U, R}
        return new{T, S, P, U, R}(est, pre, up, resampling)
    end
end

struct BootstrapParticleFilterPrediction{T, F <: AbstractStateModel} <:
       AbstractFilterPrediction
    state::F
    function BootstrapParticleFilterPrediction{T}(state::F) where {T, F}
        return new{T, F}(state)
    end
end

function predict!(est::ParticleState, p::BootstrapParticleFilterPrediction; kwargs...)
    @inbounds for i in 1:length(est) # loop over all the particles
        transition!(p.state, @views(est.p[i, :]), @views(est.p[i, :]); kwargs...)
    end
    nothing
end

struct BootstrapParticleFilterUpdate{T, L <: AbstractLikelihoodModel} <:
       AbstractFilterUpdate
    likely::L
    function BootstrapParticleFilterUpdate{T}(likely::L) where {T, L}
        return new{T, L}(likely)
    end
end

function update!(est::ParticleState, u::BootstrapParticleFilterUpdate, z; kwargs...)
    @inbounds for i in 1:length(est)
        est.w[i] = likelihood(u.likely, @views(est.p[i, :]), z; kwargs...)
    end
    normalize!(est) # Normalize weights
    nothing
end

function step!(pf::BootstrapParticleFilter{T}, z::AbstractVector{T}; kwargs...) where {T}
    predict!(pf.est, pf.pre; kwargs...)
    update!(pf.est, pf.up, z; kwargs...)
    resample!(pf.est, pf.resampling; kwargs...)
    nothing
end