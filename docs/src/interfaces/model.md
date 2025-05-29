# Model Interface

This section defines the core interfaces for dynamical and observation models, as well as 
their associated noise processes, used in filtering algorithms. All models are designed to 
be compatible with generic filter structures, including Kalman filters, particle filters, 
and beyond.

## State models

State models define the system's transition dynamics between time steps. 
They support both time-invariant and time-dependent formulations.


```@docs
Filters.AbstractStateModel
```

```@docs
Filters.transition!(::AbstractStateModel, xn, x; kwargs...)
Filters.transition!(::AbstractStateModel, xn, t, Δt, x, p; kwargs...)
Filters.jacobian(::AbstractStateModel)
```

## Observation models

Observation models relate the hidden state to the measured observations. 
They mirror the structure of the state models and support both time-invariant and 
time-dependent definitions.

```@docs
AbstractObservationModel
```

```@docs
observation!(::AbstractObservationModel, z, x; kwargs...)
observation!(::AbstractObservationModel, z, t, x, p; kwargs...)
jacobian(::AbstractObservationModel)
```

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
