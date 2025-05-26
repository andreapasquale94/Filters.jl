
struct BaseKalmanFilter{
    T <: Number,
    S <: AbstractStateEstimate,
    P <: AbstractFilterPrediction,
    U <: AbstractFilterUpdate
} <: AbstractKalmanFilter{T}
    est::S
    pre::P
    up::U
end

function predict!(kf::BaseKalmanFilter{T}; u = missing, kwargs...) where {T}
    predict!(kf.est, kf.pre; u = u, kwargs...)
end

function update!(
    kf::BaseKalmanFilter{T},
    z::AbstractVector{T};
    u = missing,
    kwargs...
) where {T}
    update!(kf.est, kf.up, z; u = u, kwargs...)
end

function step!(
    kf::BaseKalmanFilter{T},
    z::AbstractVector{T};
    uk = missing,
    uk1 = missing,
    kwargs...
) where {T}
    predict!(kf; u = uk, kwargs...)
    update!(kf, z; u = uk1, kwargs...)
    nothing
end

@inline estimate(kf::BaseKalmanFilter) = kf.est

# ------------------------------------------------------------------------------------------
# Time-constant Kalman filters 
# ------------------------------------------------------------------------------------------

struct KalmanFilterPrediction{
    T <: Number,
    S <: AbstractStateModel,
    N <: AbstractTimeConstantNoiseModel
} <: AbstractFilterPrediction
    state::S
    noise::N
    function KalmanFilterPrediction{T}(state::S, noise::N) where {T, S, N}
        return new{T, S, N}(state, noise)
    end
end

function predict!(
    est::KalmanState{T},
    kfp::KalmanFilterPrediction;
    u = missing,
    kwargs...
) where {T}
    # State estimate time update
    transition!(kfp.state, est.x, est.x; u = u, kwargs...)
    # Prediction error covariance time update
    Q = covariance(kfp.noise)
    F = jacobian(kfp.state)
    @inbounds est.P .= F * est.P * F' .+ Q
    nothing
end

struct KalmanFilterUpdate{
    T <: Number,
    O <: AbstractObservationModel,
    N <: AbstractTimeConstantNoiseModel
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

function update!(
    est::KalmanState{T},
    kfu::KalmanFilterUpdate{T, <:Any, <:Any},
    z::AbstractVector{T};
    u = missing,
    kwargs...
) where {T}
    # Measurement prediction
    observation!(kfu.obs, kfu.z, est.x; u = u, kwargs...)

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

const KalmanFilter{T} = BaseKalmanFilter{
    T,
    KalmanState{T},
    KalmanFilterPrediction{T, <:AbstractStateModel, <:AbstractTimeConstantNoiseModel},
    KalmanFilterUpdate{T, <:AbstractObservationModel, <:AbstractTimeConstantNoiseModel}
}