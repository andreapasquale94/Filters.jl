"""
    SquareRootKalmanFilter{T} <: AbstractSequentialFilter{T}

Square-root formulation of the classic linear Kalman filter using
Cholesky factors of covariance matrices.

This implementation improves numerical stability by maintaining only the
lower-triangular Cholesky factors of the covariance matrices.

The system is defined by:

- **State transition**:  `xₖ = F*xₖ₋₁ + B*uₖ + wₖ`,  where `wₖ ∼ 𝒩(0, Qₖ)`
- **Observation**:       `zₖ = H*xₖ + D*uₖ + vₖ`,    where `vₖ ∼ 𝒩(0, Rₖ)`

### Fields

#### Dimensions
- `n` — Dimension of the state vector.
- `m` — Dimension of the measurement vector.

#### State estimate
- `x` — Current estimate of the state.
- `sqrtP` — Cholesky factor of the state covariance.

#### System model
- `F` — State transition matrix.
- `B` — Control input matrix.
- `sqrtQ` — Cholesky factor of process noise covariance.
- `H` — Observation matrix.
- `D` — Control-to-observation matrix.
- `sqrtR` — Cholesky factor of observation noise covariance.

#### Diagnostics
These fields store intermediate quantities from the *most recent measurement update* 
(useful for debugging or adaptive filtering):

- `z` — Predicted measurement.
- `y` — Innovation (residual).
- `sqrtS` — Cholesky factor of innovation covariance.
- `K` — Kalman gain from the last update.
"""
struct SquareRootKalmanFilter{T} <: AbstractKalmanFilter{T}
    n::Int
    m::Int
    x::Vector{T}
    sqrtP::LowerTriangular{T}
    F::Matrix{T}
    B::Union{Matrix{T},Nothing}
    sqrtQ::LowerTriangular{T}
    H::Matrix{T}
    D::Union{Matrix{T},Nothing}
    sqrtR::LowerTriangular{T}
    z::Vector{T}
    y::Vector{T}
    sqrtS::LowerTriangular{T}
    K::Matrix{T}
    M::Matrix{T}
    U::Matrix{T}
end

function SquareRootKalmanFilter{T}(
    nx::Int, m::Int, x0, sqrtP0, F0, B0, sqrtQ0, H0, D0, sqrtR0) where {T}
    z0 = zeros(T, m)
    y0 = zeros(T, m)
    S0 = LowerTriangular(Matrix{T}(I, m, m))
    K0 = zeros(T, nx, m)

    M = zeros(T, max(nx, m), 2max(nx, m))
    U = zeros(T, nx, m)
    return SquareRootKalmanFilter(
        nx, m,
        copy(x0), LowerTriangular(copy(sqrtP0)),
        F0, B0, LowerTriangular(sqrtQ0),
        H0, D0, LowerTriangular(sqrtR0),
        z0, y0, S0, K0, M, U)
end

# ==========================================================================================================

@inline nx(filter::SquareRootKalmanFilter) = filter.n
@inline nz(filter::SquareRootKalmanFilter) = filter.m
@inline nu(filter::SquareRootKalmanFilter) = filter.B === nothing ? 0 : size(filter.B, 2)
@inline islinear(filter::SquareRootKalmanFilter) = true

function Base.show(io::IO, kf::SquareRootKalmanFilter{T}) where T
    println(io, "SquareRootKalmanFilter{$T}")
    println(io, " x̂: ", estimate(kf))
    println(io, " P: ", covariance(kf))
    return
end

# ==========================================================================================================

function predict!(kf::SquareRootKalmanFilter{T}; u=nothing) where {T}
    # Predict the state
    if kf.B !== nothing && u !== nothing
        kf.x .= kf.F * kf.x .+ kf.B * u
    else
        kf.x .= kf.F * kf.x
    end

    # Predict covariance via QR of [F * sqrtP  sqrtQ]
    mul!(@views(kf.M[1:kf.n, 1:kf.n]), kf.F, kf.sqrtP)
    copyto!(@views(kf.M[1:kf.n, kf.n+1:2kf.n]), kf.sqrtQ)
    _, R̃ = qr!(kf.M[1:kf.n, 1:2kf.n]')
    kf.sqrtP .= LowerTriangular(R̃')
    return nothing
end

function update!(kf::SquareRootKalmanFilter{T}, z::AbstractVector{T}; u=nothing) where {T}
    # Measurement prediction
    if kf.D !== nothing && u !== nothing
        kf.z .= kf.H * kf.x .+ kf.D * u
    else
        kf.z .= kf.H * kf.x
    end

    # Innovation
    kf.y .= z .- kf.z

    # Innovation covariance cholesky factor
    mul!(@views(kf.M[1:kf.m, 1:kf.n]), kf.H, kf.sqrtP)
    copyto!(@views(kf.M[1:kf.m, kf.n+1:(kf.n+kf.m)]), kf.sqrtR)
    _, R̃ = qr!(kf.M[1:kf.m, 1:(kf.n+kf.m)]')
    kf.sqrtS .= LowerTriangular(R̃')

    # Kalman gain
    kf.K .= ((kf.sqrtP * (kf.H * kf.sqrtP)') / kf.sqrtS') / kf.sqrtS

    # State update
    kf.x .+= kf.K * kf.y
    # Covariance cholesky factor update
    kf.U .= kf.K * kf.sqrtR'
    cholesky_downdate!(kf.sqrtP, kf.U)
    nothing
end

@inline estimate(kf::SquareRootKalmanFilter) = kf.x
@inline covariance(kf::SquareRootKalmanFilter) = kf.sqrtP * kf.sqrtP'
@inline loglikelihood(kf::SquareRootKalmanFilter) = -0.5 * (kf.m * log(2π) + 2 * sum(log.(diag(kf.sqrtS))))