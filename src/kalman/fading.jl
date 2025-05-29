struct FadingKalmanFilterPrediction{
    T <: Number,
    S <: AbstractStateModel,
    N <: AbstractWhiteNoiseModel
} <: AbstractFilterPrediction
    state::S
    noise::N
    alpha::T
    function FadingKalmanFilterPrediction{T}(
        state::S,
        noise::N,
        alpha::T = 1.01
    ) where {T, S, N}
        return new{T, S, N}(state, noise, alpha)
    end
end

function predict!(
    est::KalmanState{T},
    kfp::FadingKalmanFilterPrediction;
    u = missing,
    kwargs...
) where {T}
    # State estimate time update
    transition!(kfp.state, est.x, est.x; u = u, kwargs...)
    # Prediction error covariance time update
    Q = covariance(kfp.noise)
    F = transition_matrix(kfp.state)
    @inbounds est.P .= (kfp.alpha^2) * F * est.P * F' .+ Q
    nothing
end

"""
    FadingKalmanFilter{T}

Implements a generic fading memory Kalman filter.
"""
const FadingKalmanFilter{T} = BaseKalmanFilter{
    T,
    KalmanState{T},
    FadingKalmanFilterPrediction{T, <:AbstractStateModel, <:AbstractWhiteNoiseModel},
    KalmanFilterUpdate{T, <:AbstractObservationModel, <:AbstractWhiteNoiseModel}
}