module Filters

using LinearAlgebra

# ------------------------------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------------------------------

export AbstractStateEstimate, estimate, covariance
include("state.jl")

export AbstractFilter
export AbstractSequentialFilter, predict!, update!, step!
export AbstractFilterPrediction, AbstractFilterUpdate
include("filter.jl")

export AbstractModel
export AbstractStateModel, AbstractObservationModel, transition!, observation!, jacobian
export AbstractNoiseModel, AbstractTimeConstantNoiseModel, AbstractTimeDependantNoiseModel
include("model.jl")

# ------------------------------------------------------------------------------------------
# Kalman filters 
# ------------------------------------------------------------------------------------------

export KalmanState
export LinearStateModel, LinearObservationModel, ConstantGaussianNoise
include("kalman/models.jl")

export KalmanFilterPrediction, KalmanFilterUpdate, KalmanFilter
include("kalman/kf.jl")

end