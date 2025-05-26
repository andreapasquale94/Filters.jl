# ——————————————————————————————————————————————————————————————————————————————————————————
# State estimate models
# ------------------------------------------------------------------------------------------

struct KalmanState{T <: Number} <: AbstractStateEstimate
    x::Vector{T}
    P::Matrix{T}
end

@inline estimate(s::KalmanState) = s.x
@inline covariance(s::KalmanState) = s.P

struct SquareRootKalmanState{T <: Number} <: AbstractStateEstimate
    x::Vector{T}
    L::LowerTriangular{T}
end

@inline estimate(s::SquareRootKalmanState) = s.x
@inline covariance(s::SquareRootKalmanState) = s.L * s.L'

# ——————————————————————————————————————————————————————————————————————————————————————————
# State transition models
# ------------------------------------------------------------------------------------------

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
# Observiation models
# ------------------------------------------------------------------------------------------

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

struct ConstantGaussianNoise{T <: Number} <: AbstractTimeConstantNoiseModel
    M::Matrix{T}
end

@inline covariance(noise::ConstantGaussianNoise) = noise.M

@inline LinearAlgebra.cholesky(noise::ConstantGaussianNoise) = cholesky(noise.M).L

struct ConstantLGaussianNoise{T <: Number} <: AbstractTimeConstantNoiseModel
    L::LowerTriangular{T}
end

@inline covariance(noise::ConstantLGaussianNoise) = noise.L * noise.L'

@inline LinearAlgebra.cholesky(noise::ConstantLGaussianNoise) = noise.L
