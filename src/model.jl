abstract type AbstractModel end

# State 

abstract type AbstractStateModel <: AbstractModel end

"""
    transition!(out, model, x)

Compute the next state `out` given current state `x`.
"""
function transition!(out, m::AbstractStateModel, x; kwargs...)
    throw(MethodError(transition!, (out, m, x)))
end

"""
    jacobian(model::AbstractStateModel)

Compute the state model jacobian matrix.
"""
function jacobian(m::AbstractStateModel)
    throw(MethodError(jacobian, (m)))
end

# Observation

abstract type AbstractObservationModel <: AbstractModel end

"""
    observation!(out, model, x)

Compute the expected measurement `out` from state `x`.
"""
function observation!(out, m::AbstractObservationModel, x; kwargs...)
    throw(MethodError(observation!, (out, m, x)))
end

"""
    jacobian(model::AbstractObservationModel)

Compute the observation model jacobian matrix.
"""
function jacobian(m::AbstractObservationModel)
    throw(MethodError(jacobian, (m)))
end

# Noise 

abstract type AbstractNoiseModel <: AbstractModel end
abstract type AbstractTimeConstantNoiseModel <: AbstractNoiseModel end

function covariance(m::AbstractTimeConstantNoiseModel)
    throw(MethodError(covariance, (m)))
end

abstract type AbstractTimeDependantNoiseModel <: AbstractNoiseModel end

function covariance(m::AbstractTimeDependantNoiseModel, t)
    throw(MethodError(covariance, (m, t)))
end