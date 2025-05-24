
struct KalmanFilterPrediction{
    T <: Number,
    S <: AbstractStateModel,
    N <: AbstractNoiseModel
} <: AbstractFilterPrediction
    state::S
    noise::N
    function KalmanFilterPrediction{T}(state::S, noise::N) where {T, S, N}
        new{T, S, N}(state, noise)
    end
end

function predict!(
    est::KalmanState{T},
    kfp::KalmanFilterPrediction;
    u = missing,
    kwargs...
) where {T}
    # State estimate time update
    transition!(est.x, kfp.state, est.x; u = u, kwargs...)
    # Prediction error covariance time update
    Q = covariance(kfp.noise)
    F = jacobian(kfp.state)
    @inbounds est.P .= F * est.P * F' .+ Q
    nothing
end

struct KalmanFilterUpdate{
    T <: Number,
    O <: AbstractObservationModel,
    N <: AbstractNoiseModel
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
        new{T, O, N}(
            obs,
            noise,
            zeros(T, n_states, n_obs),
            zeros(T, n_obs, n_obs),
            zeros(T, n_obs),
            zeros(T, n_obs)
        )
    end
end

function update!(
    est::KalmanState{T},
    kfu::KalmanFilterUpdate{T, <:Any, <:Any},
    z::AbstractVector{T};
    u = missing,
    kwargs...
) where {T}
    # Measurement prediction
    observation!(kfu.z, kfu.obs, est.x; u = u, kwargs...)

    @inbounds begin
        # Compute the innovation
        kfu.y .= z .- kfu.z

        # Compute the innovation covariance
        R = covariance(kfu.noise)
        H = jacobian(kfu.obs)
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

struct KalmanFilter{T} <: AbstractSequentialFilter
    est::KalmanState{T}
    prediction::KalmanFilterPrediction{T, <:AbstractStateModel, <:AbstractNoiseModel}
    update::KalmanFilterUpdate{T, <:AbstractObservationModel, <:AbstractNoiseModel}
end

function predict!(kf::KalmanFilter{T}; u = missing) where {T}
    predict!(kf.est, kf.prediction; u = u)
end

function update!(
    kf::KalmanFilter{T},
    z::AbstractVector{T};
    u = missing,
    kwargs...
) where {T}
    update!(kf.est, kf.update, z; u = u, kwargs...)
end

function step!(
    kf::KalmanFilter{T},
    z::AbstractVector{T};
    uk = missing,
    uk1 = missing,
    kwargs...
) where {T}
    predict!(kf; u = uk)
    update!(kf, z; u = uk1)
end

estimate(kf::KalmanFilter) = kf.est