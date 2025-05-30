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

"""
    AbstractKalmanFilter{T}

Abstract type for all Kalman-base sequential filters.
"""
abstract type AbstractKalmanFilter{T} <: AbstractSequentialFilter end

export KalmanState, SquareRootKalmanState, SigmaPointsKalmanState
include("kalman/state.jl")
include("kalman/filter.jl")

end