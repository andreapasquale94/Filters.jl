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
    estimate!(out::AbstractVector, est::AbstractStateEstimate)

Compute the current state estimate in-place in `out`.
"""
function estimate!(out, est::AbstractStateEstimate)
    throw(MethodError(estimate!, (out, est)))
end

"""
    covariance(est::AbstractStateEstimate)

Returns the current state covariance matrix.
"""
function covariance(est::AbstractStateEstimate)
    throw(MethodError(covariance, (est,)))
end

"""
    covariance!(out::AbstractMatrix, est::AbstractStateEstimate)

Compute the current state estimate estimation covariance in-place in `out`.
"""
function covariance!(out, est::AbstractStateEstimate)
    throw(MethodError(covariance!, (out, est)))
end

"""
    confidence(est::AbstractStateEstimate)

Returns the current state estimate confidence (3σ).
"""
function confidence(est::AbstractStateEstimate)
    return 3sqrt.(diag(covariance(est)))
end

"""
    confidence!(out::AbstractVector, est::AbstractStateEstimate)

Compute the current state estimate confidence in-place, in `out`.
"""
function confidence!(out, est::AbstractStateEstimate)
    throw(MethodError(confidence!, (out, est)))
end
