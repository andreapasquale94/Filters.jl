# Model Interface

This section defines the core interfaces for dynamical and observation models, as well as 
their associated noise processes, used in filtering algorithms. All models are designed to 
be compatible with generic filter structures, including Kalman filters, particle filters, 
and beyond.

----

## State models

State models define the system's transition dynamics between time steps. 
Both time-dependent and constant formulations are supported.  

### Math

In order to have a clear view of the following models, let's introduct a generic non-linear 
model whose state vector is defined as ``x(t) \in \mathbb{R}^n``, that has (constant) parameters 
``\theta \in \mathbb{R}^p`` and (deterministic) inputs ``u(t) \in \mathbb{R}^c``.
Let's now assume that the state evolves with time based on a flow:

```math
\phi(t; x_0, t_0, \theta, u(t)) \, : \mathbb{R}  \to \mathbb{R}^n
```

and let the flow be the solution of the following differential equation:

```math
\begin{align*}
    \dot{x}(t) &= g(t, x(t), u(t), \theta) \\ 
    x(t_0) &= x_0
\end{align*}
```

Now, let us assume that at a given time ``t_k`` we are interested in the solution of such problem 
at a subsequent time-step ``t_{k+1} = t_k + \Delta t``. Using the above notation and writing 
``x_k = x(t_k)``:

```math
    x_{k+1} = f(t_k, x_k, \Delta t, \theta, u(t)) = \phi(\Delta t; x_k, t_k, \theta, u(t))
```

Note that here the deterministic control ``u(t)`` enters as function of time the discrete-time 
model. While this is the most accurate, there are approximations that could be used in case 
``\Delta t`` is small, so that:

```math
    x_{k+1} = f(t_k, x_k, u_k, \Delta t, \theta)
```

- **Zero-order hold**: it is assumed that ``u(t) = u(t_k)\,, \forall t \in [t_k, t_{k+1}]``.
- **Mid-point sampling**: it is assumed that ``u(t) = u(t_k + \Delta t /2) \,, \forall t \in [t_k, t_{k+1}]``.

### API 

```@docs
AbstractStateModel
AbstractTimeConstantStateModel
```

```@docs
propagate!(::AbstractStateModel, xn, x, t; Δt, u, θ, kwargs...)
propagate!(::AbstractTimeConstantStateModel, xn, x; u, θ, kwargs...)
stm(::AbstractStateModel)
psm(::AbstractStateModel)
```

The propagation could be combined to a specific [`AbstractStateEstimate`](@ref) via the 
interface, that is the actual interface *required* by all filters in this package to work.

```@docs
propagate!(::AbstractStateModel, ::AbstractStateEstimate; Δt, u, θ, kwargs...)
```

----

## Observation models

Observation models relate the hidden state to the measured observations. 
They mirror the structure of the state models and support both time-invariant and 
time-dependent definitions.

```@docs
AbstractObservationModel
```

```@docs
observe!(::AbstractObservationModel, z, x, t; kwargs...)
observe!(::AbstractTimeConstantObservationModel, z, x; kwargs...)
ojac(::AbstractObservationModel)
```

The measurement could be combined to a specific [`AbstractStateEstimate`](@ref) via the 
interface, that is the actual interface *required* by all filters in this package to work.

```@docs
observe!(::AbstractObservationModel, ::AbstractStateEstimate, z; kwargs...)
```

----

## Noise models

Noise models represent the uncertainty injected into the system and observation dynamics. 

```@docs
AbstractNoiseModel
```

### White noise

White noise models assume time-uncorrelated disturbances, typically with constant or 
piecewise-constant covariance.

```@docs
AbstractWhiteNoiseModel
```

```@docs
covariance(::AbstractWhiteNoiseModel)
cholesky(::AbstractWhiteNoiseModel)
```
