# ——————————————————————————————————————————————————————————————————————————————————————————
# State estimate models
# ------------------------------------------------------------------------------------------

"""
    KalmanState{T}

Kalman state estimate, storing the state estimate `x` and its error covariance `P`.
"""
struct KalmanState{T <: Number} <: AbstractStateEstimate
    x::Vector{T}
    P::Matrix{T}
end

@inline estimate(s::KalmanState) = s.x

@inline estimate!(out, s::KalmanState) = out .= s.x

@inline covariance(s::KalmanState) = s.P

@inline covariance!(out, s::KalmanState) = out .= s.P

# ------------------------------------------------------------------------------------------

"""
    SquareRootKalmanState{T}

Kalman state estimate for a square-root filter, storing the state estimate `x` and the
error covariance lower triangular Cholesky factor, `L`.
"""
struct SquareRootKalmanState{T <: Number} <: AbstractStateEstimate
    x::Vector{T}
    L::LowerTriangular{T}
end

@inline estimate(s::SquareRootKalmanState) = s.x

@inline estimate!(out, s::SquareRootKalmanState) = out .= s.x

@inline covariance(s::SquareRootKalmanState) = s.L * s.L'

@inline covariance!(out, s::SquareRootKalmanState) = mul!(out, s.L, s.L')

# ------------------------------------------------------------------------------------------

"""
    SigmaPointKalmanState{T}

Kalman state estimate for a sigma-point filter, storing the sigma points `X`, the weights
for the mean `Wm` and the ones for the covariance `Wc` as well as the latest state and
covariance.
"""
struct SigmaPointKalmanState{T <: Number} <: AbstractStateEstimate
    X::Matrix{T}
    Wm::Vector{T}
    Wc::Vector{T}
    x::Vector{T}
    P::Matrix{T}
end

function SigmaPointKalmanState(x0::AbstractVector{T}, P0::AbstractMatrix{T}) where {T}
    n_states = length(x0)
    return SigmaPointKalmanState(
        zeros(T, n_states, 2n_states + 1),
        zeros(T, 2n_states + 1),
        zeros(T, 2n_states + 1),
        x0,
        P0
    )
end

@inline estimate(s::SigmaPointKalmanState{T}) where {T} = s.x
@inline covariance(s::SigmaPointKalmanState{T}) where {T} = s.P

# ——————————————————————————————————————————————————————————————————————————————————————————
# State transition models
# ------------------------------------------------------------------------------------------

"""
    LinearStateModel{T}

Linear time-invariant state model storing the system matrix `F` and the input matrix `B`.
"""
struct LinearStateModel{T <: Number} <: AbstractStateModel
    F::Matrix{T}
    B::Matrix{T}
end

function transition!(m::LinearStateModel, xn, x; u = missing, kwargs...)
    @inbounds begin
        xn .= m.F * x
        if !ismissing(u)
            xn .+= m.B * u
        end
    end
    nothing
end

function jacobian(m::LinearStateModel)
    return m.F
end

# ——————————————————————————————————————————————————————————————————————————————————————————
# Observation models
# ------------------------------------------------------------------------------------------

"""
    LinearObservationModel{T}

Linear time-invariant observation model storing the output matrix `H` and the feed-forward
matrix `D`.
"""
struct LinearObservationModel{T <: Number} <: AbstractObservationModel
    H::Matrix{T}
    D::Matrix{T}
end

function observation!(m::LinearObservationModel, z, x; u = missing, kwargs...)
    @inbounds begin
        z .= m.H * x
        if !ismissing(u)
            z .+= m.D * u
        end
    end
    nothing
end

function jacobian(m::LinearObservationModel)
    return m.H
end

# ——————————————————————————————————————————————————————————————————————————————————————————
# Noise models
# ------------------------------------------------------------------------------------------

"""
    GaussianWhiteNoise{T}

Gaussian white noise with covariance `M`.
"""
struct GaussianWhiteNoise{T <: Number} <: AbstractWhiteNoiseModel
    M::Matrix{T}
end

@inline covariance(noise::GaussianWhiteNoise) = noise.M

@inline LinearAlgebra.cholesky(noise::GaussianWhiteNoise) = cholesky(noise.M).L
