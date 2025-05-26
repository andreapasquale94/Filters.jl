
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
end

struct BootstrapParticleFilterPrediction{T, F <: AbstractStateModel} <:
       AbstractFilterPrediction
    state::F
end

function predict!(est::ParticleState, p::BootstrapParticleFilterPrediction; kwargs...)
    for i in 1:length(est) # loop over all the particles
        transition!(p.state, est.p[i, :], est.p[i, :]; kwargs...)
    end
    nothing
end

struct BootstrapParticleFilterUpdate{
    T,
    L <: AbstractModel,
    R <: AbstractParticleResampling
} <: AbstractFilterUpdate
    likelihood::L
end

function update!(est::ParticleState, u::BootstrapParticleFilterUpdate, z; kwargs...)
    for i in 1:length(est)
        est.w[i] = u.likelihood(est.p[i, :], z; kwargs...)
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