using Filters
using LinearAlgebra
using Plots
using Random
Plots.gr()

Random.seed!(1)
T = Float64;

# Include models
include("models/_lti_harmonic_oscillator.jl");
include("models/_utils.jl");

# Initial conditions (mean and covariance)
x̄0 = T[1.0, 0.0];
P0 = T[0.1 0.0; 0.0 1e-3];

# Create workspace
work = harmonic_oscillator(T, x̄0; ω0 = 1, γ = 0.1, σₚ = 1e-2, σₘ = 1e-3, ΔT = 0.1, Tf = 50);

kf = SigmaPointsKalmanFilter(
    SigmaPointsKalmanState(x̄0, P0),
    work.model.state,
    work.model.obs,
    work.model.process_noise,
    work.model.obs_noise,
    CDKFSigmaPoints(sqrt(3)),
    work.nx,
    work.nz
);

# Run the filter
estimates = run(SigmaPointsKalmanState{T}, kf, work.z_sim);

# Collect and print results
p = plot_estimates(estimates, collect(work.dt), work.x_true, work.x_sim);
p