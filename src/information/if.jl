struct InformationFilterPrediction{
    T <: Number,
    S <: AbstractStateModel,
    N <: AbstractWhiteNoiseModel
} <: AbstractFilterPrediction
    state::S
    noise::N
    x::Vector{T}
    M::Matrix{T}
    C::Matrix{T}
    L::Matrix{T}
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
            zeros(T, n_states, n_states),
            zeros(T, n_states, n_states)
        )
    end
end

function predict!(
    ifp::InformationFilterPrediction,
    est::InformationState{T};
    u = missing,
    θ = missing,
    kwargs...
) where {T}

    # Process noise
    Q = covariance(ifp.noise)
    Q_inv = Q \ I # TODO: improve, cache

    if iszero(est.Λ)
        # No prior information: prediction is purely from process noise
        est.Λ .= Q_inv
        fill!(est.η, 0)
        return
    end

    @inbounds begin
        # State estimate time update (to compute the stm)
        ifp.x .= est.Λ \ est.η  # Convert information to state estimate 
        propagate!(ifp.state, ifp.x, ifp.x; u = u, θ = θ, kwargs...)
        # Prediction error covariance time update
        F = stm(ifp.state)
        F_inv = F \ I # TODO: improve, cache 
        ifp.M .= F_inv' * est.Λ * F_inv
        ifp.C .= ifp.M / (ifp.M + Q_inv)
        ifp.L .= I - ifp.C
        # Information state time update
        est.Λ .= ifp.L * ifp.M .+ ifp.C * Q_inv * ifp.C'
        est.η .= est.Λ * ifp.x
    end
    nothing
end

struct InformationFilterUpdate{
    T <: Number,
    O <: AbstractObservationModel,
    N <: AbstractWhiteNoiseModel
} <: AbstractFilterUpdate
    obs::O
    noise::N
    I::Matrix{T}
    i::Vector{T}
    x::Vector{T}
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
            zeros(T, n_states),
            zeros(T, n_obs)
        )
    end
end

function update!(
    ifu::InformationFilterUpdate{T, <:Any, <:Any},
    est::InformationState{T},
    z::AbstractVector{T};
    u = missing,
    θ = missing,
    kwargs...
) where {T}
    # Measurement prediction
    ifu.x .= est.Λ \ est.η  # Convert information to state estimate 
    observe!(ifu.obs, ifu.z, ifu.x; u = u, θ = θ, kwargs...)

    @inbounds begin
        # Compute the innovation covariance
        R = covariance(ifu.noise)
        H = jacobian(ifu.obs)

        HRT_inv = H' * (R \ I) # TODO: improve, cache
        ifu.i .= HRT_inv * z
        ifu.I .= HRT_inv * H

        # Compute the information update
        est.η .= est.η .+ ifu.i
        est.Λ .= est.Λ .+ ifu.I
    end
    nothing
end

"""
    InformationFilter{T}

Implements a generic Information Filter (IF).
"""
const InformationFilter{S} = BaseKalmanFilter{
    S,
    InformationState{S}, # TODO: implement time dependent version
    InformationFilterPrediction{S, <:AbstractStateModel, <:AbstractWhiteNoiseModel},
    InformationFilterUpdate{S, <:AbstractObservationModel, <:AbstractWhiteNoiseModel}
}


function InformationFilter(
    s0::InformationState{T},
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
    return InformationFilter{T}(
        s0,
        InformationFilterPrediction{T}(state_model, process_noise, n_states),
        InformationFilterUpdate{T}(obs_model, obs_noise, n_states, n_obs)
    )
end