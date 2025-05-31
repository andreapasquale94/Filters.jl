struct FadingMemoryKalmanFilterPrediction{
    T <: Number,
    S <: AbstractStateModel,
    N <: AbstractWhiteNoiseModel
} <: AbstractFilterPrediction
    state::S
    noise::N
    alpha::T
    function FadingMemoryKalmanFilterPrediction{T}(
        state::S,
        noise::N,
        alpha::T = 1.01
    ) where {T, S, N}
        return new{T, S, N}(state, noise, alpha)
    end
end

function __covariance_prediction!(
    kfp::FadingMemoryKalmanFilterPrediction{T, <:Any, <:Any},
    est::KalmanState{T};
    kwargs...
) where {T}
    # Prediction error covariance time update 
    Q = covariance(kfp.noise)
    F = stm(kfp.state)
    @inbounds est.P .= (kfp.alpha^2) * F * est.P * F' .+ Q
end

function predict!(
    kfp::FadingMemoryKalmanFilterPrediction{T, <:Any, <:Any},
    est::KalmanState{T};
    u = missing,
    θ = missing,
    kwargs...
) where {T}
    # State estimate time update
    propagate!(kfp.state, est.x, est.x; u = u, θ = θ, kwargs...)
    # Prediction error covariance time update 
    __covariance_prediction!(kfp, est; u = u, θ = θ, kwargs...)
    nothing
end

function predict!(
    kfp::FadingMemoryKalmanFilterPrediction{N, <:Any, <:Any},
    est::StateEstimate{N, KalmanState{T}};
    Δt,
    u = missing,
    θ = missing,
    kwargs...
) where {N, T}
    # State estimate time update
    propagate!(kfp.state, est.x.x, est.x.x, est.t[]; Δt = Δt, u = u, θ = θ, kwargs...)
    est.t[] += Δt
    # Prediction error covariance time update 
    __covariance_prediction!(kfp, est.x; Δt = Δt, u = u, θ = θ, kwargs...)
    nothing
end

"""
    FadingMemoryKalmanFilter{T, S}

Implements a generic fading memory Kalman filter (FM-KF).
"""
const FadingMemoryKalmanFilter{T, S} = BaseKalmanFilter{
    S,
    Union{KalmanState{S}, StateEstimate{T, KalmanState{S}}},
    FadingMemoryKalmanFilterPrediction{S, <:AbstractStateModel, <:AbstractWhiteNoiseModel},
    KalmanFilterUpdate{S, <:AbstractObservationModel, <:AbstractWhiteNoiseModel}
}