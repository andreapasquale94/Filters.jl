"""
    KalmanFilter{T}

Classic (linear) Kalman filter for estimating the state of a linear dynamical system
with Gaussian noise.

This filter maintains and updates the estimate of the hidden system state based on
control inputs and noisy observations. It models the system using standard time-invariant
Kalman filter equations:

- **State transition**:  `xₖ = F*xₖ₋₁ + B*uₖ + wₖ`,    where `wₖ ∼ 𝒩(0, Qₖ)`
- **Observation**:       `zₖ = H*xₖ + D*uₖ + vₖ`,    where `vₖ ∼ 𝒩(0, Rₖ)`

# Fields

## Dimensions
- `n` — Dimension of the state vector `x`.
- `m` — Dimension of the measurement vector `z`.

## State Estimate
- `x` — Current estimate of the system state.
- `P` — Current estimate of the error covariance of the state estimate.

## Model Matrices
- `F` — State transition matrix (maps state from previous to current step).
- `B` — Control input matrix (maps control `u` to state). May be `nothing` if unused.
- `Q` — Process noise covariance matrix.
- `H` — Observation matrix (maps state to measurement space).
- `D` — Control-to-measurement matrix. May be `nothing` if unused.
- `R` — Observation noise covariance matrix.

## Diagnostics
These fields store intermediate quantities from the *most recent measurement update* 
(useful for debugging or adaptive filtering):

- `z` — Predicted measurement.
- `y` — Innovation (residual): `y = z - ̂z`.
- `S` — Innovation covariance.
- `K` — Kalman gain.

"""
struct KalmanFilter{T} <: AbstractSequentialFilter{T}
    n::Int
    m::Int
    x::Vector{T}
    P::Matrix{T}
    F::Matrix{T}
    B::Union{Matrix{T},Nothing}
    Q::Matrix{T}
    H::Matrix{T}
    D::Union{Matrix{T},Nothing}
    R::Matrix{T}
    z::Vector{T}
    y::Vector{T}
    S::Matrix{T}
    K::Matrix{T}
end

function KalmanFilter{T}(nx::Int, m::Int, x0, P0, F0, B0, Q0, H0, D0, R0) where {T}
    z0 = zeros(T, m)
    y0 = zeros(T, m)
    S0 = Matrix{T}(I, m, m)
    K0 = zeros(T, nx, m)
    return KalmanFilter(nx, m, x0, P0, F0, B0, Q0, H0, D0, R0, z0, y0, S0, K0)
end

# ==========================================================================================================

@inline nx(filter::KalmanFilter) = filter.n
@inline nz(filter::KalmanFilter) = filter.m
@inline nu(filter::KalmanFilter) = filter.B === nothing ? 0 : size(filter.B, 2)
@inline islinear(filter::KalmanFilter) = true

# ==========================================================================================================

function predict!(kf::KalmanFilter{T}; u=nothing, F=nothing, Q=nothing, B=nothing) where {T}
    Fₖ = F === nothing ? kf.F : F
    Qₖ = Q === nothing ? kf.Q : Q
    Bₖ = B === nothing ? kf.B : B
    # 1. State prediction time update
    if Bₖ !== nothing && u !== nothing
        kf.x .= Fₖ * kf.x .+ Bₖ * u
    else
        kf.x .= Fₖ * kf.x
    end
    # 2. Covariance prediction time update
    kf.P .= Fₖ * kf.P * Fₖ' .+ Qₖ
    return nothing
end

function update!(kf::KalmanFilter{T}, z::AbstractVector{T};
    u=nothing, D=nothing, H=nothing, R=nothing) where {T}

    Hₖ = H === nothing ? kf.H : H
    Rₖ = R === nothing ? kf.R : R
    Dₖ = D === nothing ? kf.D : D

    # 3. Measurement prediction
    if Dₖ !== nothing && u !== nothing
        kf.z .= Hₖ * kf.x .+ Dₖ * u
    else
        kf.z .= Hₖ * kf.x
    end

    # Compute the innovation
    kf.y .= z .- kf.z
    # Compute the innovation covariance
    PHT = kf.P * Hₖ'
    kf.S .= Hₖ * PHT .+ Rₖ

    # 4. Compute the Kalman gain
    kf.K .= PHT / kf.S

    # 5. State update
    kf.x .+= kf.K * kf.y

    # 6. Covariance update
    IKH = I - kf.K * Hₖ
    kf.P .= IKH * kf.P * IKH' .+ kf.K * Rₖ * kf.K'
    return nothing
end

@inline estimate(kf::KalmanFilter) = kf.x
@inline covariance(kf::KalmanFilter) = kf.P

function loglikelihood(kf::KalmanFilter{T}) where T
    chol = cholesky(Hermitian(kf.S))
    logdetS = 2sum(log, diag(chol.U))
    yS = chol \ kf.y
    return -0.5 * (kf.n * log(2π) + logdetS + dot(yS, yS))
end