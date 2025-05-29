# ——————————————————————————————————————————————————————————————————————————————————————————
# Time-constant square-root Kalman filters 
# ------------------------------------------------------------------------------------------

struct SquareRootKalmanFilterPrediction{
    T <: Number,
    S <: AbstractStateModel,
    N <: AbstractWhiteNoiseModel
} <: AbstractFilterPrediction
    n::Int
    state::S
    noise::N
    cache::Matrix{T}
    function SquareRootKalmanFilterPrediction{T}(
        state::S,
        noise::N,
        n_states::Int
    ) where {T, S, N}
        return new{T, S, N}(n_states, state, noise, zeros(T, n_states, 2n_states))
    end
end

function predict!(
    est::SquareRootKalmanState{T},
    kfp::SquareRootKalmanFilterPrediction;
    u = missing,
    kwargs...
) where {T}
    # State estimate time update    
    transition!(kfp.state, est.x, est.x; u = u, kwargs...)

    # Predict covariance via QR of [F * √P  √Q]
    n = kfp.n
    F = jacobian(kfp.state)

    @inbounds begin
        mul!(@views(kfp.cache[:, 1:n]), F, est.L)
        copyto!(@views(kfp.cache[:, n+1:end]), cholesky(kfp.noise))
        _, R̃ = qr!(kfp.cache[:, 1:n]')
        est.L .= LowerTriangular(R̃')
    end
    nothing
end

struct SquareRootKalmanFilterUpdate{
    T <: Number,
    O <: AbstractObservationModel,
    N <: AbstractWhiteNoiseModel
} <: AbstractFilterUpdate
    n::Int
    m::Int
    obs::O
    noise::N
    K::Matrix{T}
    S::LowerTriangular{T}
    U::Matrix{T}
    z::Vector{T}
    y::Vector{T}
    cache::Matrix{T}
    function SquareRootKalmanFilterUpdate{T}(
        obs::O,
        noise::N,
        n_states::Int,
        n_obs::Int
    ) where {T, O, N}
        return new{T, O, N}(
            n_states,
            n_obs,
            obs,
            noise,
            zeros(T, n_states, n_obs),
            LowerTriangular(Matrix{T}(I, n_obs, n_obs)),
            zeros(T, n_states, n_obs),
            zeros(T, n_obs),
            zeros(T, n_obs),
            zeros(T, n_obs, n_obs + n_states)
        )
    end
end

function update!(
    est::SquareRootKalmanState{T},
    kfu::SquareRootKalmanFilterUpdate,
    z::AbstractVector{T};
    u = missing,
    kwargs...
) where {T}
    # Measurement prediction
    observation!(kfu.obs, kfu.z, est.x; u = u, kwargs...)

    n = kfu.n
    m = kfu.m
    H = jacobian(kfu.obs)
    RL = cholesky(kfu.noise)

    @inbounds begin
        # Compute the innovation
        kfu.y .= z .- kfu.z

        # Innovation covariance cholesky factor
        mul!(@views(kfu.cache[:, 1:n]), H, est.L)
        copyto!(@views(kfu.cache[:, n+1:(n+m)]), RL)
        _, R̃ = qr!(kfu.cache[:, 1:(n+m)]')
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

const SquareRootKalmanFilter{T} = BaseKalmanFilter{
    T,
    SquareRootKalmanState{T},
    SquareRootKalmanFilterPrediction{T, <:AbstractStateModel, <:AbstractWhiteNoiseModel},
    SquareRootKalmanFilterUpdate{T, <:AbstractObservationModel, <:AbstractWhiteNoiseModel}
}
