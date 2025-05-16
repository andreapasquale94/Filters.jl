module Filters

using LinearAlgebra
using StaticArrays

export AbstractFilter, AbstractSequentialFilter, AbstractBatchFilter, AbstractSmoother,
    predict!, update!, estimate, predict, update,
    covariance, loglikelihood,
    AbstractFilterCache
include("interface.jl")
include("cache.jl")

# ==========================================================================================================
# Kalman filters

export KalmanFilterCache, KalmanFilterSCache
include("kalman/cache.jl")

export KalmanFilter
include("kalman/kf.jl")

end