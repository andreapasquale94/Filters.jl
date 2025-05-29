abstract type AbstractModel end

# ——————————————————————————————————————————————————————————————————————————————————————————
# State models
# ------------------------------------------------------------------------------------------

"""
    AbstractStateModel

Abstract supertype for state‑transition models.
"""
abstract type AbstractStateModel <: AbstractModel end

"""
    transition!(model::AbstractStateModel, xn, x; kwargs...)

Compute the next state `xn` from the current state `x` for time-constant models.
The result shall be written *in place* into `xn`.
"""
function transition!(m::AbstractStateModel, xn, x; kwargs...)
    throw(MethodError(transition!, (m, xn, x)))
end

"""
    transition!(model::AbstractStateModel, xn, t, Δt, x, p = missing; kwargs...)

Compute the next state `xn` at future time `t + Δt` given the state `x` at time `t` for
time- or parameter-dependent models. The result shall be written *in place* into `xn`.
"""
function transition!(m::AbstractStateModel, xn, t, Δt, x, p = missing; kwargs...)
    throw(MethodError(transition!, (m, xn, t, Δt, x, p)))
end

"""
    jacobian(model::AbstractStateModel)

Return the Jacobian matrix of the state-transition function with respect to the state variables.
This function needs to be called after [`transition!`](@ref).
"""
function jacobian(m::AbstractStateModel)
    throw(MethodError(jacobian, (m)))
end

# ——————————————————————————————————————————————————————————————————————————————————————————
# Observation models
# ------------------------------------------------------------------------------------------

"""
    AbstractObservationModel

Abstract supertype for observation / measurement models.
"""
abstract type AbstractObservationModel <: AbstractModel end

"""
    observation!(model::AbstractObservationModel, z, x; kwargs...)

Compute the expected measurement `z` from state `x` for time-constant models.
The result must be written in place into `z`.
"""
function observation!(m::AbstractObservationModel, z, x; kwargs...)
    throw(MethodError(observation!, (m, z, x)))
end

"""
    observation!(model::AbstractObservationModel, z, t, x, p = missing; kwargs...)

Time- or parameter-dependent counterpart of `observation!`.
Computes the expected measurement `z` at time `t` from state `x`.
"""
function observation!(m::AbstractObservationModel, z, t, x, p = missing; kwargs...)
    throw(MethodError(observation!, (m, z, t, x, p)))
end

"""
    jacobian(model::AbstractObservationModel)

Return the Jacobian of the observation function with respect to the state.
This function needs to be called after `observation!`.
"""
function jacobian(m::AbstractObservationModel)
    throw(MethodError(jacobian, (m)))
end

# ——————————————————————————————————————————————————————————————————————————————————————————
# Noise models
# ------------------------------------------------------------------------------------------

"""
    AbstractNoiseModel

Abstract supertype for process or measurement noise models.
"""
abstract type AbstractNoiseModel <: AbstractModel end

# ——— White noise ——————————————————————————————————————————————————————————————————————————

"""
    AbstractWhiteNoiseModel

Abstract supertype for process or measurement white noise models.
"""
abstract type AbstractWhiteNoiseModel <: AbstractNoiseModel end

"""
    covariance(model::AbstractWhiteNoiseModel)

Return the constant covariance matrix of the noise model.
"""
function covariance(m::AbstractWhiteNoiseModel)
    throw(MethodError(covariance, (m)))
end

"""
    cholesky(model::AbstractWhiteNoiseModel)

Return the Cholesky factor of the constant covariance matrix.
"""
function LinearAlgebra.cholesky(m::AbstractWhiteNoiseModel)
    throw(MethodError(cholesky, (m)))
end
