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

end