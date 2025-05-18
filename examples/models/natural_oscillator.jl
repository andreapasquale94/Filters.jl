using LinearAlgebra

# ==============================================================================
# Damped harmonic oscillator, natural response 
# ẍ + 2*γ*ẋ + ω₀²*x = 0
# ------------------------------------------------------------------------------
# Model parameters
ω₀ = 1.0  # Natural frequency
γ = 0.1  # Damping ratio

# Noise parameters
Bₚ = [0, 1]
σₚ² = 1e-2     # Process noise covariance
σₘ² = 1e-3     # Measurement noise covariance

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
