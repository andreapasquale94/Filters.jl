# ——— Prediction ———————————————————————————————————————————————————————————————————————————

struct KalmanFilterPrediction{
    T <: Number,
    S <: AbstractStateModel,
    N <: AbstractWhiteNoiseModel
} <: AbstractFilterPrediction
    state::S
    noise::N
    function KalmanFilterPrediction{T}(state::S, noise::N) where {T, S, N}
        return new{T, S, N}(state, noise)
    end
end

function __covariance_prediction!(
    kfp::KalmanFilterPrediction{T, <:Any, <:Any},
    est::KalmanState{T};
    kwargs...
) where {T}
    # Prediction error covariance time update 
    Q = covariance(kfp.noise)
    F = stm(kfp.state)
    @inbounds est.P .= F * est.P * F' .+ Q
end

function predict!(
    kfp::KalmanFilterPrediction{T, <:Any, <:Any},
    est::S;
    u = missing,
    θ = missing,
    kwargs...
) where {T, S <: AbstractKalmanStateEstimate}
    # State estimate time update
    propagate!(kfp.state, est.x, est.x; u = u, θ = θ, kwargs...)
    # Prediction error covariance time update 
    __covariance_prediction!(kfp, est; u = u, θ = θ, kwargs...)
    nothing
end

function predict!(
    kfp::KalmanFilterPrediction{T, <:Any, <:Any},
    est::StateEstimate{T, S};
    Δt,
    u = missing,
    θ = missing,
    kwargs...
) where {T, S <: AbstractKalmanStateEstimate}
    # State estimate time update
    propagate!(kfp.state, est.x.x, est.x.x, est.t[]; Δt = Δt, u = u, θ = θ, kwargs...)
    est.t[] += Δt
    # Prediction error covariance time update 
    __covariance_prediction!(kfp, est.x; Δt = Δt, u = u, θ = θ, kwargs...)
    nothing
end

# ——— Update ———————————————————————————————————————————————————————————————————————————————

struct KalmanFilterUpdate{
    T <: Number,
    O <: AbstractObservationModel,
    N <: AbstractWhiteNoiseModel
} <: AbstractFilterUpdate
    obs::O
    noise::N
    K::Matrix{T}
    S::Matrix{T}
    z::Vector{T}
    y::Vector{T}
    function KalmanFilterUpdate{T}(
        obs::O,
        noise::N,
        n_states::Int,
        n_obs::Int
    ) where {T, O, N}
        return new{T, O, N}(
            obs,
            noise,
            zeros(T, n_states, n_obs),
            zeros(T, n_obs, n_obs),
            zeros(T, n_obs),
            zeros(T, n_obs)
        )
    end
end

function __covariance_update!(
    kfu::KalmanFilterUpdate{T, <:Any, <:Any},
    est::S,
    z::AbstractVector{T};
    kwargs...
) where {T, S <: AbstractKalmanStateEstimate}
    @inbounds begin
        # Compute the innovation
        kfu.y .= z .- kfu.z

        # Compute the innovation covariance
        R = covariance(kfu.noise)
        H = ojac(kfu.obs)
        PHT = est.P * H' # TODO: cache
        kfu.S .= H * PHT .+ R

        # Compute the Kalman gain
        kfu.K .= PHT / kfu.S

        # Update state estimate
        est.x .+= kfu.K * kfu.y

        # Update covariance estimate
        IKH = I - kfu.K * H # TODO: cache
        est.P .= IKH * est.P * IKH' .+ kfu.K * R * kfu.K'
    end
    nothing
end

function update!(
    kfu::KalmanFilterUpdate{T, <:Any, <:Any},
    est::KalmanState{T},
    z::AbstractVector{T};
    u = missing,
    θ = missing,
    kwargs...
) where {T}
    # Measurement prediction
    observe!(kfu.obs, kfu.z, est.x; u = u, θ = θ, kwargs...)
    # Update covariance
    __covariance_update!(kfu, est, z; u = u, θ = θ, kwargs...)
    nothing
end

function update!(
    kfu::KalmanFilterUpdate{N, <:Any, <:Any},
    est::StateEstimate{T, KalmanState{N}},
    z::AbstractVector{N};
    Δt,
    u = missing,
    θ = missing,
    kwargs...
) where {N, T}
    # Measurement prediction
    observe!(kfu.obs, kfu.z, est.x.x, est.t[]; Δt, u = u, θ = θ, kwargs...)
    # Update covariance
    __covariance_update!(kfu, est.x, z; Δt = Δt, u = u, θ = θ, kwargs...)
    nothing
end

"""
    KalmanFilter{T, S}

Implements a generic Kalman Filter (KF).

* * *

    KalmanFilter(s0::Union{KalmanState{S}, StateEstimate{T, KalmanState{S}}}, 
        state_model::SM, obs_model::OM, process_noise::PN, obs_noise::ON, n_states::Int, n_obs::Int)

Constructs a new KF with the initial state `s0`, a state model, an observation model,
process (additive) noise, and observation (additive) noise.
The filter is parameterized by the number of states `n_states` and observations `n_obs`.
"""
const KalmanFilter{T, S} = BaseKalmanFilter{
    S,
    Union{KalmanState{S}, StateEstimate{T, KalmanState{S}}},
    KalmanFilterPrediction{S, <:AbstractStateModel, <:AbstractWhiteNoiseModel},
    KalmanFilterUpdate{S, <:AbstractObservationModel, <:AbstractWhiteNoiseModel}
}

function KalmanFilter(
    s0::KalmanState{T},
    state_model::SM,
    obs_model::OM,
    process_noise::PN,
    obs_noise::ON,
    n_states::Int,
    n_obs::Int
) where {
    T,
    SM <: AbstractTimeConstantStateModel,
    OM <: AbstractTimeConstantObservationModel,
    PN <: AbstractWhiteNoiseModel,
    ON <: AbstractWhiteNoiseModel
}
    return KalmanFilter{T, T}(
        s0,
        KalmanFilterPrediction{T}(state_model, process_noise),
        KalmanFilterUpdate{T}(obs_model, obs_noise, n_states, n_obs)
    )
end

function KalmanFilter(
    s0::StateEstimate{T, KalmanState{N}},
    state_model::SM,
    obs_model::OM,
    process_noise::PN,
    obs_noise::ON,
    n_states::Int,
    n_obs::Int
) where {
    T,
    N,
    SM <: AbstractStateModel,
    OM <: AbstractObservationModel,
    PN <: AbstractWhiteNoiseModel,
    ON <: AbstractWhiteNoiseModel
}
    return KalmanFilter{T, N}(
        s0,
        KalmanFilterPrediction{N}(state_model, process_noise),
        KalmanFilterUpdate{N}(obs_model, obs_noise, n_states, n_obs)
    )
end