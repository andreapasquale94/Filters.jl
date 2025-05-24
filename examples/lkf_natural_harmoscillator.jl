using Filters
using LinearAlgebra
using Plots
using Random
Plots.gr()

Random.seed!(1)

include("models.jl")

# Create workspace
work = harmonic_oscillator(; ω0 = 1.0, γ = 0.1, σₚ = 1e-2, σₘ = 1e-3, ΔT = 0.1, Tf = 50.0);

# Create filter 
T = Float64;
kf = KalmanFilter(
    KalmanState(work.x_sim[:, 1], [0.1 0.0; 0.0 1e-3]),
    KalmanFilterPrediction{T}(work.models.state, work.models.process_noise),
    KalmanFilterUpdate{T}(work.models.obs, work.models.obs_noise, work.nx, work.nz)
);

est = Vector{KalmanState{T}}(undef, work.steps);
est[1] = deepcopy(kf.est);

for i in 2:work.steps
    step!(kf, work.z[:, i])
    est[i] = deepcopy(kf.est)
end

t = collect(work.dt);
x̂ = hcat([estimate(e) for e in est]...)
xt = work.x_true
x = work.x_sim
ξ = hcat([3sqrt.(diag(covariance(e))) for e in est]...)

# Position plot
p1 = plot(t, xt[1, :], label = "\$x_{g}(t)\$");
plot!(p1, t, x[1, 1:end-1], label = "\$x(t)\$", xlabel = "\$t\$", ylabel = "\$x(t)\$");
plot!(p1, t, x̂[1, :], label = "\$\\hat{x}(t)\$", ribbon = ξ[1, :], fillalpha = 0.15);
plot!(p1, t, x̂[1, :] + ξ[1, :], label = nothing, linestyle = :dash, color = :gray);
plot!(p1, t, x̂[1, :] - ξ[1, :], label = nothing, linestyle = :dash, color = :gray);

# Velocity plot
p2 = plot(t, xt[2, :], label = "\$x_{g}(t)\$");
plot!(p2, t, x[2, 1:end-1], label = "\$x(t)\$", xlabel = "\$t\$", ylabel = "\$v(t)\$");
plot!(p2, t, x̂[2, :], label = "\$\\hat{x}(t)\$", ribbon = ξ[2, :], fillalpha = 0.15);
plot!(p2, t, x̂[2, :] + ξ[2, :], label = nothing, linestyle = :dash, color = :gray);
plot!(p2, t, x̂[2, :] - ξ[2, :], label = nothing, linestyle = :dash, color = :gray);

plot(p1, p2, layout = (2, 1), dpi = 1600, size = (800, 600), framestyle = :box)