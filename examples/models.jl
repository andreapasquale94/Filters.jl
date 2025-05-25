using LinearAlgebra
using Filters

function harmonic_oscillator(
    ::Type{T},
    x0;
    ω0,
    γ,
    σₚ,
    σₘ,
    ΔT,
    Tf,
    ω = missing
) where {T <: Number}

    # Promote
    x0 = convert(Vector{T}, x0)
    ω0 = T(ω0)
    γ = T(γ)
    σₚ = T(σₚ)
    σₘ = T(σₘ)
    ΔT = T(ΔT)
    Tf = T(Tf)
    ω = ismissing(ω) ? missing : T(ω)

    # ----
    # Damped harmonic oscillator
    # ẍ + 2*γ*ẋ + ω₀²*x = cos(ωt)

    # Continous-time model    
    nx = length(x0)
    F = T[0 1; -ω0^2 -2*γ]    # System matrix
    H = T[1 0]                # Measurement matrix
    nz = size(H, 1)           # Number of measurements
    B = T[0 1]'               # Input matrix

    # Noise
    Bₚ = T[0, 1]
    Q = Bₚ * σₚ * Bₚ'          # Process noise covariance
    R = Matrix{T}(σₘ * I(nz))  # Measurement noise covariance

    # Discrete model
    Z = T[-F Q; zeros(nx, nx) F']
    C = exp(Z * ΔT)
    Fₖ = Matrix(C[nx+1:end, nx+1:end]')
    Qₖ = Fₖ * C[1:nx, nx+1:end]
    Rₖ = R / ΔT
    Bₖ = inv(F) * (Fₖ - I(nx)) * B

    state_model = LinearStateModel(Fₖ, Bₖ)
    obs = LinearObservationModel(H, Float64[;;])
    process_noise = ConstantGaussianNoise(Qₖ)
    obs_noise = ConstantGaussianNoise(Rₖ)

    # Exact solution 
    natural(t) = exp(F * t) * x0 # Natural response 
    solution = t -> natural(t)
    if !ismissing(ω)
        A_ω = 1 / sqrt((ω0^2 - ω^2) + (2γ * ω)^2) # Forced response amplitudes
        ϕ_ω = atan(2 * γ * ω / (ω0^2 - ω^2)) # Forced response phase
        forced(t) = A_ω * [cos(ω * t - ϕ_ω), -ω * sin(ω * t - ϕ_ω)]
        solution = t -> natural(t) + forced(t)
    end

    dt = 0.0:ΔT:Tf
    nt = length(dt)
    xt = hcat([solution(t) for t in dt]...)

    # Simulated state and measurements
    x = zeros(T, nx, nt)
    x[:, 1] = x0
    z = zeros(T, nz, nt)
    u = missing
    if !ismissing(ω)
        u = cos.(ω * dt)
    end
    CQₖ = cholesky(Hermitian(Qₖ)).L
    CRₖ = cholesky(Hermitian(Rₖ)).L

    for i in 2:nt
        x[:, i] .= Fₖ * x[:, i-1] .+ CQₖ * randn(nx)
        if !ismissing(ω)
            x[:, i] .+= Bₖ * u[i-1]
        end
        z[:, i] .= H * x[:, i] .+ CRₖ * randn(nz)
    end

    # return the workspace
    return (
        nx = nx,
        nz = nz,
        nu = ismissing(ω) ? 0 : 1,
        nt = nt,
        dt = dt,
        x_true = xt,
        x_sim = x,
        z_sim = z,
        u_sim = u,
        model = (
            state = state_model,
            obs = obs,
            process_noise = process_noise,
            obs_noise = obs_noise
        )
    )
end
