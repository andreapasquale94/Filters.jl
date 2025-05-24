using OrdinaryDiffEqVerner
using LinearAlgebra
using Filters
using Plots

# Model parameters
b = 0.1
g = 9.81
l = 2.0
ω₀ = sqrt(g / l)
ΔT = 1e-2
Tf = 10

# Noise parameters
σₚ² = 1e-3     # Process noise covariance
σₘ² = 1e-4     # Measurement noise covariance
P₀ = [0.1 0.0; 0.0 1e-3]    # Initial estimate covariance

# Initial state
x₀ = [deg2rad(30), 0]

# Disturbaces
Q = Matrix(σₚ² * I(2))  # Process noise covariance
R = Matrix(σₘ² * I(1))  # Measurement noise covariance

# Time parameters
t = 0:ΔT:Tf
nt = length(t)

function dynamics!(du, u, _, _)
    du[1] = u[2]
    du[2] = -ω₀^2 * sin(u[1]) - b * u[2]
    nothing
end

function jac(x)
    θ, _ = x
    J = zeros(2, 2)
    J[1, 1] = 0
    J[1, 2] = 1
    J[2, 1] = -ω₀^2 * cos(θ)
    J[2, 2] = -b
    return J
end

function state!(xn, J, x, _, _, _)
    probk = remake(prob; u0=x)
    sk = solve(probk, Vern9())

    xn .= sk.u[end]
    Jk = jac(x)
    J .= I(2) .+ Jk * ΔT .+ (Jk .^ 2) * ΔT^2 / 2
    nothing
end

function measure!(z, H, x, _, _)
    θ = x[1]
    z[1] = x[1]

    H[1, 1] = 1.0
    H[1, 2] = 0.0

    # z[1] = l * sin(θ)
    # z[2] = l * cos(θ)

    # Jacobian H = ∂h/∂x
    # H[1, 1] = l * cos(θ)
    # H[1, 2] = 0.0
    # H[2, 1] = -l * sin(θ)
    # H[2, 2] = 0.0
    nothing
end

# ----
# Prepare measurements
x = zeros(2, nt + 1)
x[:, 1] .= x₀
z = zeros(1, nt + 1)

Fₖ = zeros(2, 2)
CQₖ = cholesky(Q * ΔT).L
CRₖ = cholesky(R).L
for i in 1:nt
    J = jac(x[:, i])
    Fₖ .= I(2) .+ J * ΔT .+ (J .^ 2) * ΔT^2 / 2
    x[:, i+1] .= Fₖ * x[:, i] .+ CQₖ * randn(2)
    measure!(@views(z[:, i]), zeros(2,2), x[:, i], nothing, nothing)
end

# ----
# True solution
prob0 = ODEProblem(dynamics!, x₀, (0, Tf))
sol0 = solve(prob0, Vern9())
θt = hcat(sol0.(t)...)

# ----
# Problem definition
prob = ODEProblem(
    dynamics!, x₀, (0, ΔT);
    dense=false, save_everystep=false, save_start=false
)

ekf = ExtendedKalmanFilter{Float64}(
    2, 1, ΔT, 0.0, state!, measure!, x₀, P₀, Q * ΔT, R
)
# chace = KalmanFilterCache(ekf)
# resize!(cache, nt)

x̂ = zeros(2, nt)
x̂[:, 1] .= x₀
for i in 2:nt 
    predict!(ekf)
    update!(ekf, z[:, i])
    x̂[:, i] .= copy(ekf.x)
end

plot(t, x[1, 1:end-1])
plot!(t, x̂[1, :])