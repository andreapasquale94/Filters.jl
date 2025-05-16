using Filters
using LinearAlgebra
using Plots
Plots.gr()

# ==============================================================================
# Dumped harmonic oscillator, natural response 
# ẍ + 2*γ*ẋ + ω₀²*x = 0
# ------------------------------------------------------------------------------
# Model parameters
ω₀ = 1.0  # Natural frequency
γ = 0.1  # Damping ratio

# Noise parameters
Bₚ = [0, 1]
σₚ² = 1e-3     # Process noise covariance
σₘ² = 1e-4     # Measurement noise covariance

# Initial conditions
x₀ = [1.0, 0.0]             # Initial (true) state
P₀ = [0.1 0.0; 0.0 1e-3]    # Initial estimate covariance

# Time parameters
ΔT = 0.1
Tf = 50.0
# ==============================================================================

# Model
nx = length(x₀)
dt = 0.0:ΔT:Tf
nt = length(dt)
F = [0 1; -ω₀^2 -2*γ]    # System matrix
H = [1. 0]               # Measurement matrix
nz = size(H, 1)          # Number of measurements
# Disturbaces
Q = Matrix(σₚ² * I(nx))  # Process noise covariance
R = Matrix(σₘ² * I(nz))  # Measurement noise covariance

# Exact solution
exact(t) = exp(F * t) * x₀
xt = hcat([exact(t) for t in dt]...) # True state

# Discrete model
Z = [-F Bₚ*σₚ²*Bₚ'; zeros(nx, nx) F']
C = exp(Z * ΔT)
Fₖ = Matrix(C[nx+1:end, nx+1:end]')
Qₖ = Fₖ * C[1:nx, nx+1:end]
Rₖ = R / ΔT

# ----
# Prepare measurements 
x = zeros(nx, nt + 1)
x[:, 1] = x₀
z = zeros(nz, nt + 1)

CQₖ = cholesky(Hermitian(Qₖ)).L
CRₖ = cholesky(Hermitian(Rₖ)).L

for i in 1:nt
    x[:, i+1] .= Fₖ * x[:, i] .+ CQₖ * randn(nx)
    z[:, i] .= H * x[:, i] .+ CRₖ * randn(nz)
end

# ----
# Create and run the Kalman filter
kf = KalmanFilter{Float64}(nx, nz, x₀, P₀, Fₖ, nothing, Qₖ, H, nothing, Rₖ)
cache = KalmanFilterCache(kf)
resize!(cache, nt)

for i in 2:nt
    predict!(cache)
    update!(cache, z[:, i])
end

# ----
# Collect results and plot
t = collect(dt)
x̂ = hcat(cache.x...) # Estimated state
confidence = hcat(cache.con...) # Confidence intervals

# Position plot
p1 = plot(t, xt[1, :], label="True")
plot!(p1, t, x[1, 1:end-1], label="Simulated", xlabel="\$t\$", ylabel="\$x(t)\$")
plot!(p1, t, x̂[1, :], label="Estimate", ribbon=confidence[1, :], fillalpha=0.15)
plot!(p1, t, x̂[1, :] + confidence[1, :], label=nothing, linestyle=:dash, color=:gray)
plot!(p1, t, x̂[1, :] - confidence[1, :], label=nothing, linestyle=:dash, color=:gray)

# Velocity plot
p2 = plot(t, xt[2, :], label="True")
plot!(p2, t, x[2, 1:end-1], label="Simulated", xlabel="\$t\$", ylabel="\$\\dot{x}(t)\$")
plot!(p2, t, x̂[2, :], label="Estimated", ribbon=confidence[2, :], fillalpha=0.15)
plot!(p2, t, x̂[2, :] + confidence[2, :], label=nothing, linestyle=:dash, color=:gray)
plot!(p2, t, x̂[2, :] - confidence[2, :], label=nothing, linestyle=:dash, color=:gray)

plot(p1, p2, layout=(2, 1), size=(1000, 800))
