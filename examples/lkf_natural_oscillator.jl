using Filters
using LinearAlgebra
using Plots
using Random
Plots.gr()

Random.seed!(1)
T = Float64;

# Include models
include("models.jl");

# Initial conditions (mean and covariance)
x̄0 = T[1.0, 0.0];
P0 = T[0.1 0.0; 0.0 1e-3];

# Create workspace
work = harmonic_oscillator(T, x̄0; ω0 = 1, γ = 0.1, σₚ = 1e-2, σₘ = 1e-3, ΔT = 0.1, Tf = 50);

# Create filter 
kf = KalmanFilter{T}(
    KalmanState(x̄0, P0),
    KalmanFilterPrediction{T}(work.model.state, work.model.process_noise),
    KalmanFilterUpdate{T}(work.model.obs, work.model.obs_noise, work.nx, work.nz)
);

est = Vector{KalmanState{T}}(undef, work.nt);
est[1] = deepcopy(kf.est);

# Run filter
for i in 2:work.nt
    step!(kf, work.z_sim[:, i])
    est[i] = deepcopy(kf.est)
end

# Collect and print results
t = collect(work.dt);
x̂ = hcat([estimate(e) for e in est]...);
xt = work.x_true;
x = work.x_sim;
cb = hcat([confidence(e) for e in est]...);

# Position plot
p1 = plot(t, xt[1, :], label = "\$x_{g}(t)\$");
plot!(p1, t, x[1, :], label = "\$x(t)\$", xlabel = "\$t\$", ylabel = "\$x(t)\$");
plot!(p1, t, x̂[1, :], label = "\$\\hat{x}(t)\$", ribbon = cb[1, :], fillalpha = 0.15);
plot!(p1, t, x̂[1, :] + cb[1, :], label = nothing, linestyle = :dash, color = :gray);
plot!(p1, t, x̂[1, :] - cb[1, :], label = nothing, linestyle = :dash, color = :gray);

# Velocity plot
p2 = plot(t, xt[2, :], label = "\$x_{g}(t)\$");
plot!(p2, t, x[2, :], label = "\$x(t)\$", xlabel = "\$t\$", ylabel = "\$v(t)\$");
plot!(p2, t, x̂[2, :], label = "\$\\hat{x}(t)\$", ribbon = cb[2, :], fillalpha = 0.15);
plot!(p2, t, x̂[2, :] + cb[2, :], label = nothing, linestyle = :dash, color = :gray);
plot!(p2, t, x̂[2, :] - cb[2, :], label = nothing, linestyle = :dash, color = :gray);

plot(p1, p2, layout = (2, 1), dpi = 1600, size = (800, 600), framestyle = :box)