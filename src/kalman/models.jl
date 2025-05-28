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
@inline covariance(s::KalmanState) = s.P

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
@inline covariance(s::SquareRootKalmanState) = s.L * s.L'

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
    ConstantGaussianNoise{T}

Constant Gaussian noise with covariance `M`.
"""
struct ConstantGaussianNoise{T <: Number} <: AbstractTimeConstantNoiseModel
    M::Matrix{T}
end

@inline covariance(noise::ConstantGaussianNoise) = noise.M

@inline LinearAlgebra.cholesky(noise::ConstantGaussianNoise) = cholesky(noise.M).L
