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

# ——— Time‑constant noise ——————————————————————————————————————————————————————————————————

"""
    AbstractTimeConstantNoiseModel

Noise model with time-invariant statistics.
"""
abstract type AbstractTimeConstantNoiseModel <: AbstractNoiseModel end

"""
    covariance(model::AbstractTimeConstantNoiseModel)

Return the constant covariance matrix of the noise model.
"""
function covariance(m::AbstractTimeConstantNoiseModel)
    throw(MethodError(covariance, (m)))
end

"""
    cholesky(model::AbstractTimeConstantNoiseModel)

Return the Cholesky factor of the constant covariance matrix.
"""
function LinearAlgebra.cholesky(m::AbstractTimeConstantNoiseModel)
    throw(MethodError(cholesky, (m)))
end

# ——— Time‑dependent noise —————————————————————————————————————————————————————————————————

"""
    AbstractTimeDependantNoiseModel

Noise model whose statistics vary with time. These processes do *not* need to be white:
they may exhibit auto-correlation or any other temporal structure.
"""
abstract type AbstractTimeDependantNoiseModel <: AbstractNoiseModel end

"""
    covariance(model::AbstractTimeDependantNoiseModel, t)

Return the instantaneous covariance matrix of the noise model at time `t`, i.e. `E[w(t)w(t)ᵀ]`.
"""
function covariance(m::AbstractTimeDependantNoiseModel, t)
    throw(MethodError(covariance, (m, t)))
end

"""
    cholesky(model::AbstractTimeDependantNoiseModel, t)

Return the Cholesky factor of the instantaneous covariance.
"""
function LinearAlgebra.cholesky(m::AbstractTimeDependantNoiseModel, t)
    throw(MethodError(cholesky, (m, t)))
end

"""
    autocovariance(model::AbstractTimeDependantNoiseModel, t, Δt)

Return the *autocovariance* matrix of the noise process between time `t` and time `t + Δt`,
i.e. `E[w(t)w(t + Δt)ᵀ]`.

If the noise is white this reduces to the instantaneous covariance for `Δt == 0` and `0` otherwise.
Implement this method for coloured / auto-correlated noise models.
"""
function autocovariance(m::AbstractTimeDependantNoiseModel, t, Δt)
    throw(MethodError(autocovariance, (m, t, Δt)))
end
