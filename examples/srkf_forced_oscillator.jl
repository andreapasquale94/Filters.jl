using Filters
using LinearAlgebra
using Plots
Plots.gr()

include("models/forced_oscillator.jl");

# ----
# Create and run the square root Kalman filter

srkf = SquareRootKalmanFilter{Float64}(
    nx, nz, x₀, cholesky(P₀).L, Fₖ, Bₖ, CQₖ, H, nothing, CRₖ
)
cache = KalmanFilterCache(srkf)
resize!(cache, nt)

for i in 2:nt
    predict!(cache; u=u[i])
    update!(cache, z[:, i])
end

# ----
# Collect results and plot
t = collect(dt)
x̂ = hcat(cache.x...) # Estimated state
confidence = hcat(cache.con...) # Confidence intervals

# Position plot
p1 = plot(t, xt[1, :], label="\$x_{g}(t)\$");
plot!(p1, t, x[1, 1:end-1], label="\$x(t)\$", xlabel="\$t\$", ylabel="\$x(t)\$");
plot!(p1, t, x̂[1, :], label="\$\\hat{x}(t)\$", ribbon=confidence[1, :], fillalpha=0.15);
plot!(p1, t, x̂[1, :] + confidence[1, :], label=nothing, linestyle=:dash, color=:gray);
plot!(p1, t, x̂[1, :] - confidence[1, :], label=nothing, linestyle=:dash, color=:gray);

# Velocity plot
p2 = plot(t, xt[2, :], label="\$x_{g}(t)\$");
plot!(p2, t, x[2, 1:end-1], label="\$x(t)\$", xlabel="\$t\$", ylabel="\$\\dot{x}(t)\$");
plot!(p2, t, x̂[2, :], label="\$\\hat{x}(t)\$", ribbon=confidence[2, :], fillalpha=0.15);
plot!(p2, t, x̂[2, :] + confidence[2, :], label=nothing, linestyle=:dash, color=:gray);
plot!(p2, t, x̂[2, :] - confidence[2, :], label=nothing, linestyle=:dash, color=:gray);

plot(p1, p2, layout=(2, 1), dpi=1600, size=(800,600),framestyle = :box)
