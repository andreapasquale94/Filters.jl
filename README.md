# Filters.jl

`Filters.jl` is a library for state estimation using recursive filters under the assumption 
of **additive noise models**. 

It provides a common interface and modular infrastructure to implement and compose various 
filtering algorithms such as:

- Kalman Filter (KF)
- Extended Kalman Filter (EKF)
- Sigma-Point Kalman Filters (SPKF)
- Bootstrap Particle Filters (BPF)

This library is designed to support general nonlinear state-space models with additive 
process and measurement noise of the form:

$x_k = f(x_{k-1}, t_{k-1},\Delta t) + w_{k-1}$

$z_k = h(x_k, t_k) + v_k$

where:

- $x_k$ is the latent state at time step $k$;
- $z_k$ is the observation prediction;
- $f$ is the (possibly non-linear) dynamics model;
- $h$ is the (possibly non-linear) observation/measurement model;
- $w_{k-1} \sim \mathcal{N}(0, Q_{k-1})$ is the additive (Gaussian) process noise;
- $v_k \sim \mathcal{N}(0, R_k)$ is the additive (Gaussian) measurement noise.


