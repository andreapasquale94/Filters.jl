module Filters

using LinearAlgebra
using StaticArrays
using FunctionWrappers: FunctionWrapper

include("utils/linalg.jl")

export AbstractFilter, AbstractSequentialFilter, AbstractBatchFilter, AbstractSmoother,
    predict!, update!, estimate, predict, update,
    covariance, loglikelihood,
    AbstractFilterCache, empty!, resize!
include("interface.jl")
include("cache.jl")

# ==========================================================================================
# Kalman filters

include("kalman/interface.jl")

export KalmanFilterCache, KalmanFilterSCache
include("kalman/cache.jl")

export KalmanFilter
include("kalman/kf.jl")

export SquareRootKalmanFilter
include("kalman/srkf.jl")

export ExtendedKalmanFilter
include("kalman/ekf.jl")

end