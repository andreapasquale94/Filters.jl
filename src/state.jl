"""
    AbstractStateEstimate

Basic type for all state estimates types.
"""
abstract type AbstractStateEstimate end

"""
    estimate(est::AbstractStateEstimate)

Returns the current state estimate.

The state estimate is the best guess of the current state of the system, given all available
information and is computed as:

```math
\\hat{x} = \\mathbb{E}[x | \\mathcal{Z}]
```

where ``x`` is the state vector and ``\\mathcal{Z}`` is the set of all available data.
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

The state covariance is the uncertainty associated with the state estimate and is computed as:

```math
\\Sigma = \\mathbb{E}[(x - \\hat{x})(x - \\hat{x})^T | \\mathcal{Z}]
```

where ``x`` is the state vector, ``\\hat{x}`` is the state estimate, and ``\\mathcal{Z}`` is
the set of all available data.
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

"""
    variance(est::AbstractStateEstimate)

Returns the current state variance vector.

The state variance is the diagonal of the state covariance matrix and is computed as:

```math
\\sigma^2 = \\mathbb{E}[(x - \\hat{x})^2 | \\mathcal{Z}]
```

where ``x`` is the state vector, ``\\hat{x}`` is the state estimate, and ``\\mathcal{Z}`` is
the set of all available data.
"""
function variance(est::AbstractStateEstimate)
    throw(MethodError(variance, (est,)))
end

"""
    variance!(out, est::AbstractStateEstimate)

Returns the current state variance vector in-place, in `out`.
"""
function variance!(out, est::AbstractStateEstimate)
    throw(MethodError(variance!, (out, est)))
end

"""
    skewness(est::AbstractStateEstimate)

Returns the current state skewness vector.

The state skewness is a measure of the asymmetry of the probability distribution of the
state estimate and is computed as:

```math
\\gamma = \\frac{\\mathbb{E}[(x - \\hat{x})^3 | \\mathcal{Z}]}{\\sigma^3}
```

where ``x`` is the state vector, ``\\hat{x}`` is the state estimate, and ``\\mathcal{Z}`` is
the set of all available data.
"""
function skewness(est::AbstractStateEstimate)
    throw(MethodError(skewness, (est,)))
end

"""
    skewness!(out, est::AbstractStateEstimate)

Returns the current state skewness vector in-place, in `out`.
"""
function skewness!(out, est::AbstractStateEstimate)
    throw(MethodError(skewness!, (out, est)))
end

"""
    kurtosis(est::AbstractStateEstimate)

Returns the current state kurtosis vector.

The state kurtosis is a measure of the *tailedness* of the probability distribution of the
state estimate and is computed as:

```math
\\kappa = \\frac{\\mathbb{E}[(x - \\hat{x})^4 | \\mathcal{Z}]}{\\sigma^4}
```

where ``x`` is the state vector, ``\\hat{x}`` is the state estimate, and ``\\mathcal{Z}`` is
the set of all available data.
"""
function kurtosis(est::AbstractStateEstimate)
    throw(MethodError(kurtosis, (est,)))
end

"""
    kurtosis(out, est::AbstractStateEstimate)

Returns the current state kurtosis vector in-place, in `out`.
"""
function kurtosis!(out, est::AbstractStateEstimate)
    throw(MethodError(kurtosis!, (out, est)))
end