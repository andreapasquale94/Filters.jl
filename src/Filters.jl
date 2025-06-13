module Filters

using LinearAlgebra

# ——————————————————————————————————————————————————————————————————————————————————————————
# Interface
# ------------------------------------------------------------------------------------------

export AbstractStateEstimate
export estimate, covariance, confidence
export variance, skewness, kurtosis
include("state.jl")

export AbstractFilter, init!
export AbstractSequentialFilter, predict!, update!, step!
export AbstractFilterPrediction, AbstractFilterUpdate
include("filter.jl")

export AbstractModel
export AbstractStateModel, transition!, transition_matrix
export AbstractObservationModel, observation!, jacobian
export AbstractNoiseModel
export AbstractWhiteNoiseModel, covariance, cholesky
include("model.jl")

# ——————————————————————————————————————————————————————————————————————————————————————————
# Kalman filters API
# ------------------------------------------------------------------------------------------

abstract type AbstractKalmanFilter{T} <: AbstractSequentialFilter end

include("kalman/utils.jl")

export KalmanState, SquareRootKalmanState, SigmaPointKalmanState
export LinearStateModel, LinearObservationModel, GaussianWhiteNoise
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

export IteratedKalmanFilter, IteratedKalmanFilterUpdate
include("kalman/ikf.jl")

export FadingKalmanFilter, FadingKalmanFilterPrediction
include("kalman/fading.jl")

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