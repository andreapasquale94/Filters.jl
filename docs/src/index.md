# Welcome to `Filters.jl` 

`Filters.jl` is a library for state estimation using generic recursive filters.

It provides a common interface and modular infrastructure to implement and compose various 
filtering algorithms such as:

- Kalman Filter (KF)
- Extended Kalman Filter (EKF)
- Sigma-Point Kalman Filters (SPKF)
- Particle Filters (PF)

This library is designed to support general nonlinear, time-dependent state-space models:

```math
\begin{equation}
    \begin{split}
        x_k &= f(x_{k-1}, u_{k-1}, w_{k-1}, t_{k-1}, \Delta t) \\ 
        z_k &= h(x_k, u_k, v_k, t_k)
    \end{split}
\end{equation}
```

where:

- ``t_k`` is the current time step;
- ``\Delta t`` is the step-size;
- ``x_k`` is the latent state at time step $k$;
- ``u_k`` is the deterministic control input;
- ``z_k`` is the observation prediction;
- ``f`` is the (possibly non-linear) dynamics model;
- ``h`` is the (possibly non-linear) observation/measurement model;
- ``w_k`` is the process noise;
- ``v_k`` is the measurement noise.


