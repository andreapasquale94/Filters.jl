"""
    BootstrapParticleFilter 

A bootstrap particle filter implementation.
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

# ——— Predict API  ————————————————————————————————————————————————————————————————————————— 

struct BootstrapParticleFilterPrediction{T, F <: AbstractStateModel} <:
       AbstractFilterPrediction
    state::F
    function BootstrapParticleFilterPrediction{T}(state::F) where {T, F}
        return new{T, F}(state)
    end
end

function predict!(
    p::BootstrapParticleFilterPrediction,
    est::ParticleState;
    u = missing,
    θ = missing,
    kwargs...
)
    @inbounds for i in 1:length(est) # loop over all the particles
        propagate!(
            p.state,
            @views(est.p[i, :]),
            @views(est.p[i, :]);
            u = u,
            θ = θ,
            kwargs...
        )
    end
    nothing
end

# ——— Update API  —————————————————————————————————————————————————————————————————————————— 

struct BootstrapParticleFilterUpdate{T, L <: AbstractLikelihoodModel} <:
       AbstractFilterUpdate
    like::L
    function BootstrapParticleFilterUpdate{T}(like::L) where {T, L}
        return new{T, L}(like)
    end
end

function update!(
    up::BootstrapParticleFilterUpdate,
    est::ParticleState,
    z;
    u = missing,
    θ = missing,
    kwargs...
)
    @inbounds for i in 1:length(est)
        est.w[i] = likelihood(up.like, @views(est.p[i, :]), z; u = u, θ = θ, kwargs...)
    end
    normalize!(est) # Normalize weights
    nothing
end

# ——— Filter API  —————————————————————————————————————————————————————————————————————————— 

function predict!(f::BootstrapParticleFilter; kwargs...)
    predict!(f.pre, f.est; kwargs...)
end

function update!(f::BootstrapParticleFilter, z::AbstractVector; kwargs...)
    update!(f.up, f.est, z; kwargs...)
end

function step!(
    pf::BootstrapParticleFilter{T},
    z::AbstractVector{T};
    Δt = missing,
    θ = missing,
    u₋ = missing,
    u₊ = missing,
    kwargs...
) where {T}
    predict!(pf; Δt = Δt, u = u₋, θ = θ, kwargs...)
    update!(pf, z; Δt = Δt, u = u₊, θ = θ, kwargs...)
    resample!(pf.resampling, pf.est; kwargs...)
    nothing
end

function BootstrapParticleFilter{T}(
    s0::ParticleState{T},
    state_model::SM,
    likelihood::L,
    resampling::R
) where {
    T,
    SM <: AbstractStateModel,
    L <: AbstractLikelihoodModel,
    R <: AbstractParticleResampling
}
    return BootstrapParticleFilter{T}(
        s0,
        BootstrapParticleFilterPrediction{T}(state_model),
        BootstrapParticleFilterUpdate{T}(likelihood),
        resampling
    )
end