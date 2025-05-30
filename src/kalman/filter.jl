
"""
    AbstractKalmanFilter{T}

Abstract type for all Kalman-base sequential filters.
"""
abstract type AbstractKalmanFilter{T} <: AbstractSequentialFilter end

"""
    BaseKalmanFilter{T, S, P, U}

A generic Kalman filter container parametrized by types:

  - `T <: Number`: Numeric type for the state elements.
  - `S <: AbstractStateEstimate`: Type representing the state estimate.
  - `P <: AbstractFilterPrediction`: Type responsible for the prediction step.
  - `U <: AbstractFilterUpdate`: Type responsible for the update step.

### Fields

  - `est`: Current state estimate.
  - `pre`: Prediction component.
  - `up`: Update component.

Bundles prediction and update logic into a single filter object.
"""
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

"""
    init!(kf::BaseKalmanFilter)

Initialize or reset the Kalman filter state. Default method does nothing; can be overridden.
"""
@inline init!(::BaseKalmanFilter) = nothing

"""
    predict!(kf::BaseKalmanFilter{T}; kwargs...)

Perform the prediction step by delegating to the filter's prediction component.
Keyword arguments are forwarded to the prediction step.
"""
function predict!(f::BaseKalmanFilter{T}; kwargs...) where {T}
    predict!(f.pre, f.est; kwargs...)
end

"""
    update!(f::BaseKalmanFilter{T}, z::AbstractVector{T}; kwargs...)

Perform the update step with measurements `z` by delegating to the filter's update component.
Keyword arguments are forwarded to the update step.
"""
function update!(f::BaseKalmanFilter{T}, z::AbstractVector{T}; kwargs...) where {T}
    update!(f.up, f.est, z; kwargs...)
end

"""
    step!(f::BaseKalmanFilter{T}, z::AbstractVector{T}; Δt = missing, θ = missing, 
          u₋ = missing, u₊ = missing, kwargs...)

Perform one filter step: prediction followed by update.

The prediction step receives `Δt`, control input `u₋`, and parameters `θ`.
The update step receives the measurement `z`, `Δt`, control input `u₊`, and `θ`.
Additional keyword arguments are forwarded to both steps.
"""
function step!(
    f::BaseKalmanFilter{T},
    z::AbstractVector{T};
    Δt = missing,
    θ = missing,
    u₋ = missing,
    u₊ = missing,
    kwargs...
) where {T}
    predict!(f; Δt = Δt, u = u₋, θ = θ, kwargs...)
    update!(f, z; Δt = Δt, u = u₊, θ = θ, kwargs...)
    nothing
end

"""
    estimate(f::BaseKalmanFilter)

Return the current state estimate stored in the filter.
"""
@inline estimate(f::BaseKalmanFilter) = f.est
