using LinearAlgebra
using Filters

function harmonic_oscillator(; ω0, γ, σₚ, σₘ, ΔT, Tf)
    # Initial state 
    x0 = [1.0, 0.0]

    # ---
    # Continous-time model

    # Damped harmonic oscillator, natural response 
    # ẍ + 2*γ*ẋ + ω₀²*x = 0
    nx = length(x0)
    dt = 0.0:ΔT:Tf
    nt = length(dt)
    F = [0 1; -ω0^2 -2*γ]    # System matrix
    H = Float64[1 0]         # Measurement matrix
    nz = size(H, 1)          # Number of measurements
    # Noise
    Bₚ = [0, 1]
    Q = Bₚ * σₚ * Bₚ'        # Process noise covariance
    R = Matrix(σₘ * I(nz))  # Measurement noise covariance

    # Exact solution
    exact(t) = exp(F * t) * x0
    xt = hcat([exact(t) for t in dt]...) # Exact solution

    # ---
    # Discrete time model
    Z = [-F Q; zeros(nx, nx) F']
    C = exp(Z * ΔT)
    Fₖ = Matrix(C[nx+1:end, nx+1:end]')
    Qₖ = Fₖ * C[1:nx, nx+1:end]
    Rₖ = R / ΔT

    # Simulated state and measurements
    x = zeros(nx, nt + 1)
    x[:, 1] = x0
    z = zeros(nz, nt + 1)
    CQₖ = cholesky(Hermitian(Qₖ)).L
    CRₖ = cholesky(Hermitian(Rₖ)).L
    for i in 1:nt
        x[:, i+1] .= Fₖ * x[:, i] .+ CQₖ * randn(nx)
        z[:, i] .= H * x[:, i] .+ CRₖ * randn(nz)
    end

    state = LinearStateModel(Fₖ, Float64[;;])
    obs = LinearObservationModel(H, Float64[;;])
    process_noise = ConstantGaussianNoise(Qₖ)
    obs_noise = ConstantGaussianNoise(Rₖ)

    return (
        steps = length(dt),
        nx = nx,
        nz = nz,
        nu = 0,
        dt = dt,
        x_true = xt,
        x_sim = x,
        z = z,
        models = (
            state = LinearStateModel(Fₖ, Float64[;;]),
            obs = LinearObservationModel(H, Float64[;;]),
            process_noise = ConstantGaussianNoise(Qₖ),
            obs_noise = ConstantGaussianNoise(Rₖ)
        )
    )
end