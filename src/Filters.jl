module Filters

using LinearAlgebra

# ------------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------------
# Kalman filters 
# ------------------------------------------------------------------------------------------

include("kalman/utils.jl")

export KalmanState, SquareRootKalmanState
export LinearStateModel, LinearObservationModel, ConstantGaussianNoise
include("kalman/models.jl")

export KalmanFilter, KalmanFilterPrediction, KalmanFilterUpdate
include("kalman/kf.jl")

export SquareRootKalmanFilter,
    SquareRootKalmanFilterPrediction, SquareRootKalmanFilterUpdate
include("kalman/srkf.jl")
end