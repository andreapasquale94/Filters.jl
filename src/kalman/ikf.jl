struct IteratedKalmanFilterUpdate{
    T <: Number,
    O <: AbstractObservationModel,
    N <: AbstractWhiteNoiseModel
} <: AbstractFilterUpdate
    obs::O
    noise::N
    N::Int
    K::Matrix{T}
    S::Matrix{T}
    z::Vector{T}
    y::Vector{T}
    function IteratedKalmanFilterUpdate{T}(
        obs::O,
        noise::N,
        n_iter::Int,
        n_states::Int,
        n_obs::Int
    ) where {T, O, N}
        return new{T, O, N}(
            obs,
            noise,
            n_iter,
            zeros(T, n_states, n_obs),
            zeros(T, n_obs, n_obs),
            zeros(T, n_obs),
            zeros(T, n_obs)
        )
    end
end

function __covariance_update!(
    kfu::IteratedKalmanFilterUpdate{T, <:Any, <:Any},
    est::KalmanState{T},
    z::AbstractVector{T};
    kwargs...
) where {T}
    @inbounds begin
        # Compute the innovation
        kfu.y .= z .- kfu.z

        # Compute the innovation covariance
        H = ojac(kfu.obs)
        R = covariance(kfu.noise)
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
    kfu::IteratedKalmanFilterUpdate{T, <:Any, <:Any},
    est::KalmanState{T},
    z::AbstractVector{T};
    u = missing,
    θ = missing,
    kwargs...
) where {T}
    for _ in 1:kfu.N
        # Measurement prediction
        observe!(kfu.obs, kfu.z, est.x; u = u, θ = θ, kwargs...)
        # Update covariance
        __covariance_update!(kfu, est, z; u = u, θ = θ, kwargs...)
    end
    nothing
end

function update!(
    kfu::IteratedKalmanFilterUpdate{N, <:Any, <:Any},
    est::StateEstimate{T, KalmanState{N}},
    z::AbstractVector{N};
    Δt,
    u = missing,
    θ = missing,
    kwargs...
) where {N, T}
    for _ in 1:kfu.N
        # Measurement prediction
        observe!(kfu.obs, kfu.z, est.x.x, est.t[]; Δt = Δt, u = u, θ = θ, kwargs...)
        # Update covariance
        __covariance_update!(kfu, est.x, z; Δt = Δt, u = u, θ = θ, kwargs...)
    end
    nothing
end

"""
    IteratedKalmanFilter{T, S}

Implements a generic Iterated Kalman Filter (IKF).

----

    IteratedKalmanFilter(s0::Union{KalmanState{S}, StateEstimate{T, KalmanState{S}}}, 
        state_model::SM, obs_model::OM, process_noise::PN, obs_noise::ON, n_iter::Int,
        n_states::Int, n_obs::Int)

Constructs a new IKF with the initial state `s0`, a state model, an observation model,
process (additive) noise, and observation (additive) noise.
The filter is parameterized by the number of states `n_states` and observations `n_obs`.
"""
const IteratedKalmanFilter{T, S} = BaseKalmanFilter{
    S,
    Union{KalmanState{S}, StateEstimate{T, KalmanState{S}}},
    KalmanFilterPrediction{S, <:AbstractStateModel, <:AbstractWhiteNoiseModel},
    IteratedKalmanFilterUpdate{S, <:AbstractObservationModel, <:AbstractWhiteNoiseModel}
}

function IteratedKalmanFilter(
    s0::KalmanState{T},
    state_model::SM,
    obs_model::OM,
    process_noise::PN,
    obs_noise::ON,
    n_iter::Int,
    n_states::Int,
    n_obs::Int
) where {
    T,
    SM <: AbstractTimeConstantStateModel,
    OM <: AbstractTimeConstantObservationModel,
    PN <: AbstractWhiteNoiseModel,
    ON <: AbstractWhiteNoiseModel
}
    return IteratedKalmanFilter{T, T}(
        s0,
        KalmanFilterPrediction{T}(state_model, process_noise),
        IteratedKalmanFilterUpdate{T}(obs_model, obs_noise, n_iter, n_states, n_obs)
    )
end

function IteratedKalmanFilter(
    s0::StateEstimate{T, KalmanState{N}},
    state_model::SM,
    obs_model::OM,
    process_noise::PN,
    obs_noise::ON,
    n_iter::Int,
    n_states::Int,
    n_obs::Int,
) where {
    T,
    N,
    SM <: AbstractStateModel,
    OM <: AbstractObservationModel,
    PN <: AbstractWhiteNoiseModel,
    ON <: AbstractWhiteNoiseModel
}
    return IteratedKalmanFilter{T, N}(
        s0,
        KalmanFilterPrediction{N}(state_model, process_noise),
        IteratedKalmanFilterUpdate{N}(obs_model, obs_noise, n_iter, n_states, n_obs)
    )
end