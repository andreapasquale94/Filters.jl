"""
    AbstractFilter{T}

This is an abstract type that serves as a base for all filter types in the package.
"""
abstract type AbstractFilter{T} end

"""
    AbstractSequentialFilter{T}

This is an abstract type that serves as a base for all sequential filter types in the package.
"""
abstract type AbstractSequentialFilter{T} <: AbstractFilter{T} end

"""
    AbstractBatchFilter{T}

This is an abstract type that serves as a base for all batch filter types in the package.
"""
abstract type AbstractBatchFilter{T} <: AbstractFilter{T} end

"""
    AbstractSmoother{T}

This is an abstract type that serves as a base for all smoother types in the package.
"""
abstract type AbstractSmoother{T} <: AbstractFilter{T} end

"""
    ftype(filter::AbstractFilter)

Return the type of the filter's state.
"""
@inline ftype(::AbstractFilter{T}) where T = T

"""
    nstates(filter::AbstractFilter)::Int

Return the state dimension of the filter.
"""
@inline nstates(filter::AbstractFilter) = throw(MethodError(nstates, (filter,)))

"""
    nobs(filter::AbstractFilter)::Int

Return the observation dimension of the filter.
"""
@inline nobs(filter::AbstractFilter) = throw(MethodError(nobs, (filter,)))

"""
    ncontrol(filter::AbstractFilter)::Int

Return the control input dimension of the filter. 
"""
@inline ncontrol(filter::AbstractFilter) = throw(MethodError(ncontrol, (filter,)))

"""
    islinear(filter::AbstractFilter)::Bool 

Return true if the filter is linear, false otherwise.
"""
@inline islinear(filter::AbstractFilter) = throw(MethodError(islinear, (filter,)))

# ==========================================================================================================
# Methods
# ==========================================================================================================

"""
    predict!(filter::AbstractFilter; kwargs...)

Perform the time-update step (*prior*).
"""
function predict!(filter::AbstractFilter; kwargs...)
    throw(MethodError(predict!, (filter,)))
end

"""
    update!(filter::AbstractFilter, z; kwargs...)

Perform the measurement-update step (*posterior*).
"""
function update!(filter::AbstractFilter, z; kwargs...)
    throw(MethodError(update!, (filter, z)))
end

"""
    update!(filter::AbstractFilter, i::Int, z::Real; kwargs...)

Perform the i-th measurement-update step (*posterior*).
"""
function update!(filter::AbstractFilter, i, z; kwargs...)
    throw(MethodError(update!, (filter, z)))
end

"""
    estimate(filter::AbstractFilter)

Return the current best estimate of the state.
"""
function estimate(filter::AbstractFilter)
    throw(MethodError(estimate, (filter,)))
end

"""
    predict(filter::AbstractFilter; kwargs...)

Perform the prediction step and return the predicted state (prior). 
"""
@inline function predict(filter::AbstractFilter; kwargs...)
    predict!(filter; kwargs...)
    return estimate(filter)
end

"""
    update(filter::AbstractFilter, z; kwargs...)

Perform the measurement-update step and return the predicted state (posterior). 
"""
@inline function update(filter::AbstractFilter, z; kwargs...)
    update(filter, z; kwargs...)
    return estimate(filter)
end

# ==========================================================================================================
# Optional methods 
# ==========================================================================================================

"""
    covariance(filter::AbstractFilter)

Return the posterior covariance (if defined).
"""
@inline covariance(::AbstractFilter) = nothing

"""
    loglikelihood(filter::AbstractFilter)

Return the (incremental) log-likelihood of the last update.
"""
@inline loglikelihood(::AbstractFilter) = nothing