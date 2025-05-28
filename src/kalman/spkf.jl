
struct SigmaPointsKalmanFilterPrediction{
    T <: Number,
    S <: AbstractStateModel,
    N <: AbstractTimeConstantNoiseModel,
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
    est::SigmaPointKalmanState{T},
    sfp::SigmaPointsKalmanFilterPrediction{T, <:Any, <:Any, <:Any};
    u = missing,
    kwargs...
) where {T}
    # Compute sigma points
    compute!(sfp.sigma, est)
    @inbounds begin
        # State time update for all sigma points 
        @views for j in eachindex(est.Wc)
            transition!(kfp.state, est.X[:, j], est.X[:, j]; u = u, kwargs...)
        end
        # Update state estimate
        est.x .= est.X * est.Wm

        # Update estimation error covariance 
        fill!(est.P, zero(T))
        for j in eachindex(est.Wc)
            sfp.dx .= @view est.X[:, j] - est.x
            est.P .+= est.Wc[j] * (sfp.dx * sfp.dx')
        end
    end
    nothing
end

struct SigmaPointsKalmanFilterUpdate{
    T <: Number,
    O <: AbstractObservationModel,
    N <: AbstractTimeConstantNoiseModel
} <: AbstractFilterPrediction
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

function update!(
    est::SigmaPointKalmanState{T},
    spu::SigmaPointsKalmanFilterUpdate{T, <:Any, <:Any, <:Any},
    z::AbstractVector{T};
    u = missing,
    kwargs...
) where {T}
    @inbounds begin
        # Measurement prediction for all sigma points
        @views for j in eachindex(est.Wc)
            observation!(spu.obs, spu.Z[:, j], est.X[:, j]; u = u, kwargs...)
        end

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
        spu.K .= spu.Pxz / kfu.Pzz

        # Update state estimate
        est.x .+= spu.K * spu.y
        # Update covariance estimate
        est.P .-= spu.K * spu.Pzz * spu.K'
    end
    nothing
end

"""
    SigmaPointsKalmanFilter{T}

Implements a generic Sigma Points Kalman filter with a prediction and an update step.
"""
const SigmaPointsKalmanFilter{T} = BaseKalmanFilter{
    T,
    SigmaPointKalmanState{T},
    SigmaPointsKalmanFilterPrediction{
        T,
        <:AbstractStateModel,
        <:AbstractTimeConstantNoiseModel
    },
    SigmaPointsKalmanFilterUpdate{
        T,
        <:AbstractObservationModel,
        <:AbstractTimeConstantNoiseModel
    }
}