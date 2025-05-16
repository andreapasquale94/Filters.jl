using Filters
using LinearAlgebra
using Plots
Plots.gr()

# ==============================================================================
# Dumped harmonic oscillator, natural response 
# ẍ + 2*γ*ẋ + ω₀²*x = 0
# ------------------------------------------------------------------------------
# Model parameters
m = 50
c = 2
k = 4
ω₀ = sqrt(k / m)  # Natural frequency
γ = c / (2 * m)  # Damping ratio

# Noise parameters
Bₚ = [0, 1 / m]
σₚ² = 0.1     # Process noise covariance
σₘ² = 1e-4     # Measurement noise covariance

# Initial conditions
x₀ = [1.0, 0.0]             # Initial (true) state
P₀ = [0.1 0.0; 0.0 1e-4]    # Initial estimate covariance

# Time parameters
ΔT = 0.1
Tf = 300.0
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
x̂ = hcat(cache.x...) # Estimated state

p1 = plot(collect(dt), xt[1, :], label="True", xlabel="\$t\$", ylabel="\$x(t)\$");
plot!(p1, collect(dt), x[1, 1:end-1], label="Simulated");
p2 = plot(collect(dt), xt[2, :], label="True", xlabel="\$t\$", ylabel="\$\\dot{x}(t)\$");
plot!(p2, collect(dt), x[2, 1:end-1], label="Simulated");

plot!(p1, collect(dt), x̂[1, :], label="Estimated");
plot!(p2, collect(dt), x̂[2, :], label="Estimated");
plot(p1, p2, layout=(2, 1), size=(1000, 800))

