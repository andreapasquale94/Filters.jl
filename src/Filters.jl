module Filters

using LinearAlgebra

# ——————————————————————————————————————————————————————————————————————————————————————————
# Interface
# ------------------------------------------------------------------------------------------

export AbstractStateEstimate, estimate, covariance, confidence
include("state.jl")

export AbstractFilter
export AbstractSequentialFilter, predict!, update!, step!
export AbstractFilterPrediction, AbstractFilterUpdate
include("filter.jl")

export AbstractModel
export AbstractStateModel, AbstractObservationModel, transition!, observation!, jacobian
export AbstractNoiseModel, AbstractTimeConstantNoiseModel, AbstractTimeDependantNoiseModel
export cholesky
include("model.jl")

# ——————————————————————————————————————————————————————————————————————————————————————————
# Kalman filters API
# ------------------------------------------------------------------------------------------

abstract type AbstractKalmanFilter{T} <: AbstractSequentialFilter end

include("kalman/utils.jl")

export KalmanState, SquareRootKalmanState, SigmaPointKalmanState
export LinearStateModel, LinearObservationModel, ConstantGaussianNoise
include("kalman/models.jl")

export KalmanFilter, KalmanFilterPrediction, KalmanFilterUpdate
include("kalman/kf.jl")

export SquareRootKalmanFilter,
    SquareRootKalmanFilterPrediction, SquareRootKalmanFilterUpdate
include("kalman/srkf.jl")

export UKFSigmaPoints, CDKFSigmaPoints, compute!
include("kalman/sigma.jl")

export SigmaPointsKalmanFilter,
    SigmaPointsKalmanFilterPrediction, SigmaPointsKalmanFilterUpdate
include("kalman/spkf.jl")

# ------------------------------------------------------------------------------------------
# Information filters API
# ------------------------------------------------------------------------------------------

export InformationState
include("information/models.jl")

export InformationFilter, InformationFilterPrediction, InformationFilterUpdate
include("information/if.jl")

# ——————————————————————————————————————————————————————————————————————————————————————————
# Particle filters API
# ------------------------------------------------------------------------------------------

abstract type AbstractParticleFilter{T} <: AbstractSequentialFilter end

export ParticleState, normalize!, length, effective_samples
export AbstractLikelihoodModel, likelihood
include("particle/models.jl")

export Resampling, resample!, trigger
export EffectiveSamplesPolicy
export NoResamplingAlgorithm, SystematicResamplingAlgorithm, MultinomialResamplingAlgorithm
include("particle/resampling.jl")

export BootstrapParticleFilter,
    BootstrapParticleFilterPrediction, BootstrapParticleFilterUpdate
include("particle/bootstrap.jl")

end