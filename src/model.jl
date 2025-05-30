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
    propagate!(model::AbstractStateModel, xn, x, t; Δt, u, θ, kwargs...)

Compute the next state `xn` at future time `t + Δt` given the state `x` at time `t` for
time- and parameter-dependent (`θ`) models. The result is written *in place* into `xn`.
"""
function propagate!(m::AbstractStateModel, xn, x, t; Δt, u, θ, kwargs...)
    throw(MethodError(propagate!, (m, xn, x, t, Δt, u, θ)))
end

"""
    stm(model::AbstractStateModel) -> Φ

Return the state transition matrix (STM).

The *STM* can be derived integrating, along with the state differential equations:

```math
\\begin{align*}
    \\dot{\\Phi}(t, t_0) &= \\frac{\\partial g}{\\partial x}\\, \\Phi(t, t_0) \\\\
                         &= \\mathcal{J}_x(t) \\,\\Phi(t, t_0) \\\\
    \\Phi(t_0, t_0)      &= \\mathbb{I}_n
\\end{align*}
```

!!! note

    This function needs to be called after [`propagate!`](@ref).
"""
function stm(m::AbstractStateModel)
    throw(MethodError(stm, (m)))
end

"""
    psm(model::AbstractStateModel) -> Ψ     

Return the parameters sensitivity matrix (PSM).

The ``i``th parameter sensitivity could be defined as:

```math
s_i = \\frac{\\partial x(t)}{\\partial \\theta_i}
```

and can be shown that they follow the following differential equation:

```math
\\begin{align*}
    \\dot{s}_i(t) &= \\frac{\\partial g}{\\partial x}\\, s_i(t) + \\frac{\\partial g}{\\partial \\theta_i} \\\\
                  &= \\mathcal{J}_x(t) \\,s_i(t) + \\nabla_{\\theta_i} g\\\\
      s_i(t_0)    &= s_{i, 0}
\\end{align*}
```

Then the *PSM* is composed by the sensitivities stacked by column as follows:

```math
\\Psi = \\begin{bmatrix}
    s_1 & s_2 & \\dots & s_p
\\end{bmatrix} \\in \\mathbb{R}^{n \\times p}
```

and follows the following differential equation:

```math
\\begin{align*}
    \\dot{\\Psi}(t) &= \\mathcal{J}_x(t) \\,\\Psi(t) + \\mathcal{J}_{\\theta}(t)\\\\
    \\Psi(t_0)      &= \\Psi_0
\\end{align*}
```

!!! note

    This function needs to be called after [`propagate!`](@ref).
"""
function psm(m::AbstractStateModel)
    throw(MethodError(ptm, (m)))
end

# ——— Time constant ————————————————————————————————————————————————————————————————————————

"""
    AbstractTimeConstantStateModel

Abstract supertype for time-independent state-transition models.
"""
abstract type AbstractTimeConstantStateModel <: AbstractStateModel end

"""
    propagate!(model::AbstractTimeConstantStateModel, xn, x; u, θ, kwargs...)

Compute the next state `xn` from the current state `x` for time-constant models.
The result is written *in place* into `xn`.
"""
function propagate!(m::AbstractTimeConstantStateModel, xn, x; u, θ, kwargs...)
    throw(MethodError(propagate!, (m, xn, x, u, θ)))
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
    observe!(model::AbstractObservationModel, z, x, t; kwargs...)

Time- or parameter-dependent counterpart measurement model.
Computes the expected measurement `z` at time `t` from state `x`.
"""
function observe!(m::AbstractObservationModel, z, x, t; kwargs...)
    throw(MethodError(observe!, (m, z, x, t)))
end

"""
    ojac(model::AbstractObservationModel)

Return the Jacobian of the observation function with respect to the state.

!!! note

    This function needs to be called after [`observe!`](@ref).
"""
function ojac(m::AbstractObservationModel)
    throw(MethodError(jacobian, (m)))
end

# ——— Time constant ————————————————————————————————————————————————————————————————————————

"""
    AbstractTimeConstantObservationModel

Abstract supertype for time-independent observation models.
"""
abstract type AbstractTimeConstantObservationModel <: AbstractObservationModel end

"""
    observe!(model::AbstractTimeConstantObservationModel, z, x; kwargs...)

Compute the expected measurement `z` from state `x` for time-independent models.
The result must be written in place into `z`.
"""
function observe!(m::AbstractTimeConstantObservationModel, z, x; kwargs...)
    throw(MethodError(observe!, (m, z, x)))
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
