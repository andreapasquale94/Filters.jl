"""
    AbstractStateEstimate

Basic type for all state estimates types.
"""
abstract type AbstractStateEstimate end

"""
    estimate(est::AbstractStateEstimate)

Returns the current state estimate.
"""
function estimate(est::AbstractStateEstimate)
    throw(MethodError(estimate, (est,)))
end

"""
    covariance(est::AbstractStateEstimate)

Returns the current state covariance matrix.
"""
function covariance(est::AbstractStateEstimate)
    throw(MethodError(covariance, (est,)))
end