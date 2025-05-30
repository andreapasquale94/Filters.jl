"""
    KalmanState{T}

Kalman state estimate, storing the state estimate `x` and its error covariance `P`.
"""
struct KalmanState{T <: Number} <: AbstractTimeConstantStateEstimate
    x::Vector{T}
    P::Matrix{T}
end

@inline estimate(s::KalmanState) = s.x
@inline estimate!(out, s::KalmanState) = out .= s.x
@inline covariance(s::KalmanState) = s.P
@inline covariance!(out, s::KalmanState) = out .= s.P
@inline variance(s::KalmanState) = diag(s.P)
@inline variance!(out, s::KalmanState) = out .= diag(s.P)
@inline skewness(s::KalmanState{T}) where T = zeros(T, length(s.x))
@inline skewness!(out, ::KalmanState) = fill!(out, 0)
@inline kurtosis(s::KalmanState{T}) where T = 3ones(T, length(s.x))
@inline kurtosis!(out, ::KalmanState) = fill!(out, 3)

# ------------------------------------------------------------------------------------------

"""
    SquareRootKalmanState{T}

Kalman state estimate for a square-root filter, storing the state estimate `x` and the
error covariance lower triangular Cholesky factor, `L`.
"""
struct SquareRootKalmanState{T <: Number} <: AbstractTimeConstantStateEstimate
    x::Vector{T}
    L::LowerTriangular{T}
end

@inline estimate(s::SquareRootKalmanState) = s.x
@inline estimate!(out, s::SquareRootKalmanState) = out .= s.x
@inline covariance(s::SquareRootKalmanState) = s.L * s.L'
@inline covariance!(out, s::SquareRootKalmanState) = mul!(out, s.L, s.L')
@inline variance(s::SquareRootKalmanState) = diag(covariance(s))
@inline variance!(out, s::SquareRootKalmanState) = out .= diag(covariance(s))
@inline skewness(s::SquareRootKalmanState{T}) where T = zeros(T, length(s.x))
@inline skewness!(out, ::SquareRootKalmanState) = fill!(out, 0)
@inline kurtosis(s::SquareRootKalmanState{T}) where T = 3ones(T, length(s.x))
@inline kurtosis!(out, ::SquareRootKalmanState) = fill!(out, 3)


# ------------------------------------------------------------------------------------------

"""
    SigmaPointsKalmanState{T}

Kalman state estimate for a sigma-point filter, storing the sigma points `X`, the weights
for the mean `Wm` and the ones for the covariance `Wc` as well as the latest state `x` and
covariance `P`.
"""
struct SigmaPointsKalmanState{T <: Number} <: AbstractTimeConstantStateEstimate
    X::Matrix{T}
    Wm::Vector{T}
    Wc::Vector{T}
    x::Vector{T}
    P::Matrix{T}
end

function SigmaPointsKalmanState(x0::AbstractVector{T}, P0::AbstractMatrix{T}) where {T}
    n_states = length(x0)
    return SigmaPointsKalmanState(
        zeros(T, n_states, 2n_states + 1),
        zeros(T, 2n_states + 1),
        zeros(T, 2n_states + 1),
        x0,
        P0
    )
end

@inline estimate(s::SigmaPointsKalmanState) = s.x
@inline estimate!(out, s::SigmaPointsKalmanState) = out .= s.x
@inline covariance(s::SigmaPointsKalmanState) = s.P
@inline covariance!(out, s::SigmaPointsKalmanState) = out .= s.P
@inline variance(s::SigmaPointsKalmanState) = diag(s.P)
@inline variance!(out, s::SigmaPointsKalmanState) = out .= diag(s.P)

function skewness!(out, s::SigmaPointsKalmanState{T}) where T
    n, N = size(s.X)
    dx = zeros(T, N)
    for i in 1:n
        xᵢ = @views(s.X[i, :])
        μᵢ = dot(s.Wm, xᵢ)
        dx .= (xᵢ .- μᵢ) .^ 2
        σᵢ = sqrt(dot(s.Wc, dx))
        dx .*= xᵢ .- μᵢ
        out[i] = dot(s.Wm, dx) / (σᵢ)^3
    end
    nothing
end

function skewness(s::SigmaPointsKalmanState{T}) where T
    out = zeros(T, length(s.x))
    skewness!(out, s)
    return out
end

function kurtosis!(out, s::SigmaPointsKalmanState{T}) where T
    n, N = size(s.X)
    dx = zeros(T, N)
    for i in 1:n
        xᵢ = @views(s.X[i, :])
        μᵢ = dot(s.Wm, xᵢ)
        dx .= (xᵢ .- μᵢ) .^ 2
        σᵢ² = dot(s.Wc, dx)
        dx .*= dx
        out[i] = dot(s.Wm, dx) / (σᵢ²)^2
    end
    nothing
end

function kurtosis(s::SigmaPointsKalmanState{T}) where T
    out = zeros(T, length(s.x))
    kurtosis!(out, s)
    return out
end
