module Filters

using LinearAlgebra

# ——————————————————————————————————————————————————————————————————————————————————————————
# Interface
# ------------------------------------------------------------------------------------------

export AbstractModel
export AbstractStateModel, AbstractTimeConstantStateModel, propagate!, stm, psm
export AbstractObservationModel, AbstractTimeConstantObservationModel, observe!, ojac
export AbstractNoiseModel
export AbstractWhiteNoiseModel, covariance, cholesky
include("model.jl")

export AbstractStateEstimate, AbstractTimeConstantStateEstimate, StateEstimate
export estimate, covariance, confidence, variance, skewness, kurtosis
include("state.jl")

export AbstractFilter, init!
export AbstractSequentialFilter, predict!, update!, step!
export AbstractFilterPrediction, AbstractFilterUpdate
include("filter.jl")

# ——————————————————————————————————————————————————————————————————————————————————————————
# Utils (internal)
# ------------------------------------------------------------------------------------------

include("utils/cholesky.jl")
export CDKFSigmaPoints, UKFSigmaPoints
include("utils/sigma.jl")

# ——————————————————————————————————————————————————————————————————————————————————————————
# Models
# ------------------------------------------------------------------------------------------

export LTIStateModel, LTIObservationModel
include("models/lti.jl")

export GaussianWhiteNoise
include("models/noise.jl")

# ——————————————————————————————————————————————————————————————————————————————————————————
# Kalman filters API
# ------------------------------------------------------------------------------------------

export AbstractKalmanStateEstimate
export KalmanState, SquareRootKalmanState, SigmaPointsKalmanState
include("kalman/state.jl")
include("kalman/filter.jl")

export KalmanFilter
include("kalman/kf.jl")

export SquareRootKalmanFilter
include("kalman/srkf.jl")

export SigmaPointsKalmanFilter
include("kalman/spkf.jl")

export IteratedKalmanFilter
include("kalman/ikf.jl")

export FadingMemoryKalmanFilter
include("kalman/fading.jl")

# ——————————————————————————————————————————————————————————————————————————————————————————
# Particle filters API
# ------------------------------------------------------------------------------------------

export ParticleState, normalize!, neffective
include("particle/state.jl")

export AbstractLikelihoodModel, likelihood
include("particle/likelihood.jl")

export AbstractParticleFilter
include("particle/filter.jl")

export AbstractParticleResampling, AbstractResamplingAlgorithm, AbstractResamplingPolicy
export Resampling,
    resample!,
    trigger,
    EffectiveSamplesPolicy,
    NoResamplingAlgorithm,
    SystematicResamplingAlgorithm,
    MultinomialResamplingAlgorithm,
    StratifiedResamplingAlgorithm
#    ResidualResamplingAlgorithm, 
include("particle/resample.jl")

export BootstrapParticleFilter
include("particle/bootstrap.jl")

# ——————————————————————————————————————————————————————————————————————————————————————————
# Information filters API
# ------------------------------------------------------------------------------------------

export InformationState
include("information/state.jl")

export InformationFilter
include("information/if.jl")

end