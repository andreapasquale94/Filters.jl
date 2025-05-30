# ——— Prediction ———————————————————————————————————————————————————————————————————————————

struct SquareRootKalmanFilterPrediction{
    T <: Number,
    S <: AbstractStateModel,
    N <: AbstractWhiteNoiseModel
} <: AbstractFilterPrediction
    state::S
    noise::N
    M::Matrix{T}
    function SquareRootKalmanFilterPrediction{T}(
        state::S,
        noise::N,
        n_states::Int
    ) where {T, S, N}
        return new{T, S, N}(state, noise, zeros(T, n_states, 2n_states))
    end
end

function __covariance_prediction!(
    kfp::SquareRootKalmanFilterPrediction{T, <:Any, <:Any},
    est::SquareRootKalmanState{T};
    kwargs...
) where {T}
    # Extract the number of states
    n = length(est.x)
    F = stm(kfp.state)
    @inbounds begin
        mul!(@views(kfp.M[:, 1:n]), F, est.L)
        copyto!(@views(kfp.M[:, n+1:end]), cholesky(kfp.noise))
        _, R̃ = qr!(kfp.M')
        est.L .= LowerTriangular(R̃')
    end
end

function predict!(
    kfp::SquareRootKalmanFilterPrediction{T, <:Any, <:Any},
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
    kfp::SquareRootKalmanFilterPrediction{T, <:Any, <:Any},
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

struct SquareRootKalmanFilterUpdate{
    T <: Number,
    O <: AbstractObservationModel,
    N <: AbstractWhiteNoiseModel
} <: AbstractFilterUpdate
    obs::O
    noise::N
    K::Matrix{T}
    S::LowerTriangular{T}
    U::Matrix{T}
    z::Vector{T}
    y::Vector{T}
    M::Matrix{T}
    function SquareRootKalmanFilterUpdate{T}(
        obs::O,
        noise::N,
        n_states::Int,
        n_obs::Int
    ) where {T, O, N}
        return new{T, O, N}(
            obs,
            noise,
            zeros(T, n_states, n_obs),
            LowerTriangular{T}(I(n_obs)),
            zeros(T, n_states, n_obs),
            zeros(T, n_obs),
            zeros(T, n_obs),
            zeros(T, n_obs, n_obs + n_states)
        )
    end
end

function __covariance_update!(
    kfu::SquareRootKalmanFilterUpdate{T, <:Any, <:Any},
    est::S,
    z::AbstractVector{T};
    kwargs...
) where {T, S <: AbstractKalmanStateEstimate}
    H = ojac(kfu.obs)
    m, n = size(H)
    RL = cholesky(kfu.noise)

    @inbounds begin
        # Compute the innovation
        kfu.y .= z .- kfu.z

        # Innovation covariance cholesky factor
        mul!(@views(kfu.M[:, 1:n]), H, est.L)
        copyto!(@views(kfu.M[:, n+1:(n+m)]), RL)
        _, R̃ = qr!(kfu.M')
        kfu.S .= LowerTriangular(R̃')

        # Compute the Kalman gain
        kfu.K .= ((est.L * (H * est.L)') / kfu.S') / kfu.S

        # Update state estimate
        est.x .+= kfu.K * kfu.y
        # Covariance cholesky factor update
        kfu.U .= kfu.K * RL
        cholesky_downdate!(est.L, kfu.U)
    end
    nothing
end

function update!(
    kfu::SquareRootKalmanFilterUpdate{T, <:Any, <:Any},
    est::SquareRootKalmanState{T},
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
    kfu::SquareRootKalmanFilterUpdate{N, <:Any, <:Any},
    est::StateEstimate{T, SquareRootKalmanState{N}},
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
    SquareRootKalmanFilter{T, S}

Implements a generic Square-Root Kalman Filter (SRKF).

* * *

    SquareRootKalmanFilter(s0::Union{KalmanState{S}, StateEstimate{T, KalmanState{S}}}, 
        state_model::SM, obs_model::OM, process_noise::PN, obs_noise::ON, n_states::Int, n_obs::Int)

Constructs a new SRKF with the initial state `s0`, a state model, an observation model,
process (additive) noise, and observation (additive) noise.
The filter is parameterized by the number of states `n_states` and observations `n_obs`.
"""
const SquareRootKalmanFilter{T, S} = BaseKalmanFilter{
    S,
    Union{SquareRootKalmanState{S}, StateEstimate{T, SquareRootKalmanState{S}}},
    SquareRootKalmanFilterPrediction{S, <:AbstractStateModel, <:AbstractWhiteNoiseModel},
    SquareRootKalmanFilterUpdate{S, <:AbstractObservationModel, <:AbstractWhiteNoiseModel}
}

function SquareRootKalmanFilter(
    s0::SquareRootKalmanState{T},
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
    return SquareRootKalmanFilter{T, T}(
        s0,
        SquareRootKalmanFilterPrediction{T}(state_model, process_noise, n_states),
        SquareRootKalmanFilterUpdate{T}(obs_model, obs_noise, n_states, n_obs)
    )
end

function SquareRootKalmanFilter(
    s0::StateEstimate{T, SquareRootKalmanState{N}},
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
        SquareRootKalmanFilterPrediction{N}(state_model, process_noise, n_states),
        SquareRootKalmanFilterUpdate{N}(obs_model, obs_noise, n_states, n_obs)
    )
end