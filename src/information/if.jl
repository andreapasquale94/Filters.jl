struct InformationFilterPrediction{
    T <: Number,
    S <: AbstractStateModel,
    N <: AbstractTimeConstantNoiseModel
} <: AbstractFilterPrediction
    state::S
    noise::N
    x::Vector{T}
    M::Matrix{T}
    Finv::Matrix{T}
    function InformationFilterPrediction{T}(
        state::S,
        noise::N,
        n_states::Int
    ) where {T, S, N}
        return new{T, S, N}(
            state,
            noise,
            zeros(T, n_states),
            Matrix{T}(I, n_states, n_states),
            zeros(T, n_states, n_states)
        )
    end
end

function predict!(
    est::InformationState{T},
    ifp::InformationFilterPrediction;
    u = missing,
    kwargs...
) where {T}

    # Process noise
    Q = covariance(kfp.noise)

    if iszero(est.Λ)
        # No prior information: prediction is purely from process noise
        @. est.Λ = cholesky(Q) \ I
        fill!(est.η, 0)
        return
    end

    @inbounds begin
        # State estimate time update (to compute the jacobian)
        @. ifp.x = est.Λ \ est.η  # Convert information to state estimate 
        transition!(ifp.state, ifp.x, ifp.x; u = u, kwargs...)
        # Prediction error covariance time update
        F = jacobian(kfp.state)
        @. ifp.Finv = F / I
        @. ifp.M = ifp.Finv' * est.Λ * ifp.Finv
        # Information state time update
        @. est.Λ = (I + ifp.M * Q) / ifp.M
        @. est.η = est.Λ * ifp.x
    end
    nothing
end

struct InformationFilterUpdate{
    T <: Number,
    O <: AbstractObservationModel,
    N <: AbstractTimeConstantNoiseModel
} <: AbstractFilterUpdate
    obs::O
    noise::N
    I::Matrix{T}
    i::Vector{T}
    z::Vector{T}
    function InformationFilterUpdate{T}(
        obs::O,
        noise::N,
        n_states::Int,
        n_obs::Int
    ) where {T, O, N}
        return new{T, O, N}(
            obs,
            noise,
            zeros(T, n_states, n_states),
            zeros(T, n_states),
            zeros(T, n_obs)
        )
    end
end

function update!(
    est::InformationState{T},
    ifu::InformationFilterUpdate{T, <:Any, <:Any},
    z::AbstractVector{T};
    u = missing,
    kwargs...
) where {T}
    # Measurement prediction
    observation!(ifu.obs, ifu.z, est.x; u = u, kwargs...)

    @inbounds begin
        # Compute the innovation covariance
        R = covariance(ifu.noise)
        H = jacobian(ifu.obs)

        HRT_inv = H' * (R \ I) # TODO: cache
        @. ifu.i = HRT_inv * z
        @. ifu.I = HRT_inv * H

        # Compute the information update
        @. est.η = est.η + ifu.i
        @. est.Λ = est.Λ + ifu.I
    end
    nothing
end

struct InformationFilter{T <: Number} <: AbstractInformationFilter{T}
    est::InformationState{T}
    pre::InformationFilterPrediction{
        T,
        <:AbstractStateModel,
        <:AbstractTimeConstantNoiseModel
    }
    up::InformationFilterUpdate{
        T,
        <:AbstractObservationModel,
        <:AbstractTimeConstantNoiseModel
    }
end