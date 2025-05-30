# ——— Prediction ———————————————————————————————————————————————————————————————————————————

struct SigmaPointsKalmanFilterPrediction{
    T <: Number,
    S <: AbstractStateModel,
    N <: AbstractWhiteNoiseModel,
    P <: AbstractSigmaPointsGenerator
} <: AbstractFilterPrediction
    state::S
    noise::N
    sigma::P
    dx::Vector{T}
    function SigmaPointsKalmanFilterPrediction{T}(
        state::S,
        noise::N,
        sigma::P,
        n_states::Int
    ) where {T, S, N, P}
        return new{T, S, N, P}(state, noise, sigma, zeros(T, n_states))
    end
end

function predict!(
    sfp::SigmaPointsKalmanFilterPrediction{T, <:Any, <:Any, <:Any},
    est::SigmaPointsKalmanState{T};
    u = missing,
    θ = missing,
    kwargs...
) where {T}
    # Update sigma points
    compute!(sfp.sigma, est)

    @inbounds begin
        @views for j in eachindex(est.Wc)
            # Propagate each sigma point through the state transition function
            propagate!(sfp.state, est.X[:, j], est.X[:, j]; u = u, θ = θ, kwargs...)
        end
        # Update state estimate
        estimate!(est.x, est)

        # Update estimation error covariance 
        covariance!(est.P, est; dx = sfp.dx)
    end
    nothing
end

function predict!(
    sfp::SigmaPointsKalmanFilterPrediction{N, <:Any, <:Any, <:Any},
    est::StateEstimate{T, SigmaPointsKalmanState{N}};
    Δt,
    u = missing,
    θ = missing,
    kwargs...
) where {N, T}
    # Update sigma sigma points
    compute!(sfp.sigma, est.x)

    @inbounds begin
        @views for j in eachindex(est.x.Wc)
            # Propagate each sigma point through the state transition function
            propagate!(
                sfp.state,
                est.x.X[:, j],
                est.x.X[:, j],
                est.t[];
                Δt = Δt,
                u = u,
                θ = θ,
                kwargs...
            )
        end
        # Update state estimate
        estimate!(est.x.x, est.x)
        # Update time
        est.t[] += Δt

        # Update estimation error covariance 
        covariance!(est.x.P, est.x; dx = sfp.dx)
    end
    nothing
end

# ——— Update ———————————————————————————————————————————————————————————————————————————————

struct SigmaPointsKalmanFilterUpdate{
    T <: Number,
    O <: AbstractObservationModel,
    N <: AbstractWhiteNoiseModel
} <: AbstractFilterUpdate
    obs::O
    noise::N
    K::Matrix{T}
    z::Vector{T}
    y::Vector{T}
    dx::Vector{T}
    dz::Vector{T}
    Z::Matrix{T}
    Pxz::Matrix{T}
    Pzz::Matrix{T}
    function SigmaPointsKalmanFilterUpdate{T}(
        obs::O,
        noise::N,
        n_states::Int,
        n_obs::Int
    ) where {T, O, N}
        return new{T, O, N}(
            obs,
            noise,
            zeros(T, n_states, n_obs),
            zeros(T, n_obs),
            zeros(T, n_obs),
            zeros(T, n_states),
            zeros(T, n_obs),
            zeros(T, n_obs, 2n_states + 1),
            zeros(T, n_states, n_obs),
            zeros(T, n_obs, n_obs)
        )
    end
end

function __covariance_update!(
    spu::SigmaPointsKalmanFilterUpdate{T, <:Any, <:Any},
    est::S,
    z::AbstractVector{T};
    kwargs...
) where {T, S <: AbstractKalmanStateEstimate}
    @inbounds begin
        # Measurement prediction 
        spu.z .= spu.Z * est.Wm
        # Residuals
        spu.y .= z - spu.z

        # Innovation covariance
        fill!(spu.Pxz, zero(T))
        R = covariance(spu.noise)
        spu.Pzz .= R

        @views for j in eachindex(est.Wc)
            spu.dx .= est.X[:, j] .- est.x
            spu.dz .= spu.Z[:, j] .- spu.z
            spu.Pxz .+= est.Wc[j] * (spu.dx * spu.dz')
            spu.Pzz .+= est.Wc[j] * (spu.dz * spu.dz')
        end

        # Kalman gain
        spu.K .= spu.Pxz / spu.Pzz
        # Update state estimate
        est.x .+= spu.K * spu.y
        # Update covariance estimate
        est.P .-= spu.K * spu.Pzz * spu.K'
    end
    nothing
end

function update!(
    spu::SigmaPointsKalmanFilterUpdate{T, <:Any, <:Any},
    est::SigmaPointsKalmanState{T},
    z::AbstractVector{T};
    u = missing,
    θ = missing,
    kwargs...
) where {T}
    @inbounds begin
        # Measurement prediction for all sigma points
        @views for j in eachindex(est.Wc)
            observe!(spu.obs, spu.Z[:, j], est.X[:, j]; u = u, θ = θ, kwargs...)
        end
        # Update covariance
        __covariance_update!(spu, est, z; u = u, θ = θ, kwargs...)
    end
    nothing
end

function update!(
    spu::SigmaPointsKalmanFilterUpdate{N, <:Any, <:Any},
    est::StateEstimate{T, SigmaPointsKalmanState{N}},
    z::AbstractVector{N};
    Δt,
    u = missing,
    θ = missing,
    kwargs...
) where {N, T}
    @inbounds begin
        # Measurement prediction for all sigma points
        @views for j in eachindex(est.Wc)
            observe!(
                spu.obs,
                spu.Z[:, j],
                est.x.X[:, j],
                est.t[];
                Δt = Δt,
                u = u,
                θ = θ,
                kwargs...
            )
        end
        # Update covariance
        __covariance_update!(spu, est.x, z; Δt = Δt, u = u, θ = θ, kwargs...)
    end
    nothing
end

"""
    SigmaPointsKalmanFilter{T, S}

Implements a generic Sigma-Points Kalman Filter (SPKF).

----

    SigmaPointsKalmanFilter(
        s0::Union{SigmaPointsKalmanState{S}, StateEstimate{T, SigmaPointsKalmanState{S}}}, 
        state_model::SM, obs_model::OM, process_noise::PN, obs_noise::ON, sigma::P, 
        n_states::Int, n_obs::Int)

Constructs a new SPKF with the initial state `s0`, a state model, an observation model,
process (additive) noise, and observation (additive) noise.
The filter is parameterized by the number of states `n_states` and observations `n_obs`.
"""
const SigmaPointsKalmanFilter{T, S} = BaseKalmanFilter{
    S,
    Union{SigmaPointsKalmanState{S}, StateEstimate{T, SigmaPointsKalmanState{S}}},
    SigmaPointsKalmanFilterPrediction{
        S,
        <:AbstractStateModel,
        <:AbstractWhiteNoiseModel,
        <:AbstractSigmaPointsGenerator
    },
    SigmaPointsKalmanFilterUpdate{S, <:AbstractObservationModel, <:AbstractWhiteNoiseModel}
}

function SigmaPointsKalmanFilter(
    s0::SigmaPointsKalmanState{T},
    state_model::SM,
    obs_model::OM,
    process_noise::PN,
    obs_noise::ON,
    sigma::P,
    n_states::Int,
    n_obs::Int
) where {
    T,
    SM <: AbstractTimeConstantStateModel,
    OM <: AbstractTimeConstantObservationModel,
    PN <: AbstractWhiteNoiseModel,
    ON <: AbstractWhiteNoiseModel,
    P <: AbstractSigmaPointsGenerator
}
    return SigmaPointsKalmanFilter{T, T}(
        s0,
        SigmaPointsKalmanFilterPrediction{T}(state_model, process_noise, sigma, n_states),
        SigmaPointsKalmanFilterUpdate{T}(obs_model, obs_noise, n_states, n_obs)
    )
end

function SigmaPointsKalmanFilter(
    s0::StateEstimate{T, SigmaPointsKalmanState{N}},
    state_model::SM,
    obs_model::OM,
    process_noise::PN,
    obs_noise::ON,
    sigma::P,
    n_states::Int,
    n_obs::Int
) where {
    T,
    N,
    SM <: AbstractStateModel,
    OM <: AbstractObservationModel,
    PN <: AbstractWhiteNoiseModel,
    ON <: AbstractWhiteNoiseModel,
    P <: AbstractSigmaPointsGenerator
}
    return SigmaPointsKalmanFilter{T, N}(
        s0,
        SigmaPointsKalmanFilterPrediction{N}(state_model, process_noise, sigma, n_states),
        SigmaPointsKalmanFilterUpdate{N}(obs_model, obs_noise, n_states, n_obs)
    )
end
