"""
    KalmanFilter

Classic linear Kalman filter.

----

    KalmanFilter(nx::Int, m::Int, nu::Int=0)
    KalmanFilter{T}(nx::Int, m::Int, nu::Int=0)

Create a Kalman filter with state dimension `nx`, measurement dimension `m`, and optional control input dimension `nu`.
"""
struct KalmanFilter{T} <: AbstractSequentialFilter{T}
    n::Int # state dimension
    m::Int # measurement dimension

    x::Vector{T} # current state estimate 
    P::Matrix{T} # current error covariance

    # model
    F::Matrix{T}
    B::Union{Matrix{T},Nothing}
    Q::Matrix{T}
    H::Matrix{T}
    D::Union{Matrix{T},Nothing}
    R::Matrix{T}

    # diagnostics from the last update
    z::Vector{T} # measurement prediction
    y::Vector{T} # residual
    S::Matrix{T} # innovation covariance
    K::Matrix{T} # Kalman gain
end

function KalmanFilter{T}(nx::Int, m::Int, x0, P0, F0, B0, Q0, H0, D0, R0) where {T}
    z0 = zeros(T, m)
    y0 = zeros(T, m)
    S0 = Matrix{T}(I, m, m)
    K0 = zeros(T, nx, m)
    return KalmanFilter(nx, m, x0, P0, F0, B0, Q0, H0, D0, R0, z0, y0, S0, K0)
end

function KalmanFilter{T}(nx::Int, m::Int, nu::Int) where {T}
    x0 = zeros(T, nx)
    P0 = Matrix{T}(I, nx, nx)
    F0 = Matrix{T}(I, nx, nx)
    H0 = zeros(T, m, nx)
    Q0 = Matrix{T}(I, nx, nx)
    R0 = Matrix{T}(I, m, m)
    B0 = nu > 0 ? zeros(T, nx, nu) : nothing
    D0 = nu > 0 ? zeros(T, m, nu) : nothing
    return KalmanFilter{T}(nx, m, x0, P0, F0, B0, Q0, H0, D0, R0)
end

@inline KalmanFilter(nx::Int, m::Int, nu::Int=0) = KalmanFilter{Float64}(nx, m, nu)

# ==========================================================================================================
# Methods
# ==========================================================================================================

@inline nx(filter::KalmanFilter) = filter.n
@inline nz(filter::KalmanFilter) = filter.m
@inline nu(filter::KalmanFilter) = filter.B === nothing ? 0 : size(filter.B, 2)
@inline islinear(filter::KalmanFilter) = true

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