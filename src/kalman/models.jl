# State 

struct KalmanState{T <: Number} <: AbstractStateEstimate
    x::Vector{T}
    P::Matrix{T}
end

@inline estimate(s::KalmanState) = s.x
@inline covariance(s::KalmanState) = s.P

# ----

struct LinearStateModel{T <: Number} <: AbstractStateModel
    F::Matrix{T}
    B::Matrix{T}
end

function transition!(out, m::LinearStateModel, x; u = missing, kwargs...)
    @inbounds begin
        out .= m.F * x
        if !ismissing(u)
            out .+= m.B * u
        end
    end
    nothing
end

function jacobian(m::LinearStateModel)
    return m.F
end

# Observation

struct LinearObservationModel{T <: Number} <: AbstractObservationModel
    H::Matrix{T}
    D::Matrix{T}
end

function observation!(out, m::LinearObservationModel, x; u = missing, kwargs...)
    @inbounds begin
        out .= m.H * x
        if !ismissing(u)
            out .+= m.D * u
        end
    end
    nothing
end

function jacobian(m::LinearObservationModel)
    return m.H
end

# Noise

struct ConstantGaussianNoise{T <: Number} <: AbstractTimeConstantNoiseModel
    M::Matrix{T}
end

@inline covariance(noise::ConstantGaussianNoise) = noise.M
