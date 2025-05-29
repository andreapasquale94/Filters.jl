
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
        n_iterations::Int,
        n_states::Int,
        n_obs::Int
    ) where {T, O, N}
        return new{T, O, N}(
            obs,
            noise,
            n_iterations,
            zeros(T, n_states, n_obs),
            zeros(T, n_obs, n_obs),
            zeros(T, n_obs),
            zeros(T, n_obs)
        )
    end
end

function update!(
    est::KalmanState{T},
    kfu::IteratedKalmanFilterUpdate{T, <:Any, <:Any},
    z::AbstractVector{T};
    u = missing,
    kwargs...
) where {T}
    for _ in 1:kfu.N
        # Measurement prediction
        observation!(kfu.obs, kfu.z, est.x; u = u, kwargs...)

        @inbounds begin
            # Compute the innovation
            kfu.y .= z .- kfu.z

            # Compute the innovation covariance
            R = covariance(kfu.noise)
            H = transition_matrix(kfu.obs)
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
    end
    nothing
end

"""
    IteratedKalmanFilter{T}

Implements a generic iterated Kalman filter.
"""
const IteratedKalmanFilter{T} = BaseKalmanFilter{
    T,
    KalmanState{T},
    KalmanFilterPrediction{T, <:AbstractStateModel, <:AbstractWhiteNoiseModel},
    IteratedKalmanFilterUpdate{T, <:AbstractObservationModel, <:AbstractWhiteNoiseModel}
}