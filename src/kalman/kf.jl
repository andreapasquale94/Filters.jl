"""
    KalmanFilter{T}

Classic (linear) Kalman filter for estimating the state of a linear dynamical system
with Gaussian noise.

This filter maintains and updates the estimate of the hidden system state based on
control inputs and noisy observations. It models the system using standard time-invariant
Kalman filter equations:

- **State transition**:  `xₖ = F*xₖ₋₁ + B*uₖ + wₖ`,  where `wₖ ∼ 𝒩(0, Qₖ)`
- **Observation**:       `zₖ = H*xₖ + D*uₖ + vₖ`,    where `vₖ ∼ 𝒩(0, Rₖ)`

### Fields

#### Dimensions
- `n` — Dimension of the state vector `x`.
- `m` — Dimension of the measurement vector `z`.

#### State estimate
- `x` — Current estimate of the system state.
- `P` — Current estimate of the error covariance of the state estimate.

#### Model 
- `F` — State transition matrix (maps state from previous to current step).
- `B` — Control input matrix (maps control `u` to state). May be `nothing` if unused.
- `Q` — Process noise covariance matrix.
- `H` — Observation matrix (maps state to measurement space).
- `D` — Control-to-measurement matrix. May be `nothing` if unused.
- `R` — Observation noise covariance matrix.

#### Diagnostics
These fields store intermediate quantities from the *most recent measurement update* 
(useful for debugging or adaptive filtering):

- `z` — Predicted measurement.
- `y` — Innovation (residual): `y = z - ̂z`.
- `S` — Innovation covariance.
- `K` — Kalman gain.

"""
struct KalmanFilter{T} <: AbstractKalmanFilter{T}
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
    return KalmanFilter(
        nx, m, copy(x0), copy(P0), F0, B0, Q0, H0, D0, R0, z0, y0, S0, K0
    )
end

# ==========================================================================================================

@inline nx(filter::KalmanFilter) = filter.n
@inline nz(filter::KalmanFilter) = filter.m
@inline nu(filter::KalmanFilter) = filter.B === nothing ? 0 : size(filter.B, 2)
@inline islinear(filter::KalmanFilter) = true

function Base.show(io::IO, kf::KalmanFilter{T}) where T
    println(io, "KalmanFilter{$T}")
    println(io, " x̂: ", estimate(kf))
    println(io, " P: ", covariance(kf))
    return
end

# ==========================================================================================================

function predict!(kf::KalmanFilter{T}; u=nothing) where {T}
    # 1. State prediction time update
    if kf.B !== nothing && u !== nothing
        kf.x .= kf.F * kf.x .+ kf.B * u
    else
        kf.x .= kf.F * kf.x
    end
    # 2. Covariance prediction time update
    kf.P .= kf.F * kf.P * kf.F' .+ kf.Q
    return nothing
end

function update!(kf::KalmanFilter{T}, z::AbstractVector{T}; u=nothing) where {T}
    # 3. Measurement prediction
    if kf.D !== nothing && u !== nothing
        kf.z .= kf.H * kf.x .+ kf.D * u
    else
        kf.z .= kf.H * kf.x
    end

    # Compute the innovation
    kf.y .= z .- kf.z
    # Compute the innovation covariance
    PHT = kf.P * kf.H'
    kf.S .= kf.H * PHT .+ kf.R

    # 4. Compute the Kalman gain
    kf.K .= PHT / kf.S

    # 5. State update
    kf.x .+= kf.K * kf.y

    # 6. Covariance update
    IKH = I - kf.K * kf.H
    kf.P .= IKH * kf.P * IKH' .+ kf.K * kf.R * kf.K'
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