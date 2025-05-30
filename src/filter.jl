"""
    AbstractFilter 

Abstract base type for all filters.
"""
abstract type AbstractFilter end

"""
    init!(filter::AbstractFilter)

Perform initialization step of the filter.
"""
function init!(filter::AbstractFilter)
    throw(MethodError(init!, (filter)))
end

# ——————————————————————————————————————————————————————————————————————————————————————————
# Sequential filter API
# ------------------------------------------------------------------------------------------

"""
    AbstractSequentialFilter

Abstract base type for all sequential filters.
"""
abstract type AbstractSequentialFilter <: AbstractFilter end

"""
    predict!(f::AbstractSequentialFilter; kwargs...)

Perform the prediction step of a filter (prior).
"""
function predict!(f::AbstractSequentialFilter; kwargs...)
    throw(MethodError(predict!, (f)))
end

"""
    update!(f::AbstractFilter, obs; kwargs...)

Perform the update step of a filter (posterior), given the observation `obs`.
"""
function update!(f::AbstractSequentialFilter, obs; kwargs...)
    throw(MethodError(update!, (f, obs)))
end

"""
    step!(f::AbstractSequentialFilter, obs; kwargs...)

Perform a full filtering step (predict + update).
"""
function step!(f::AbstractSequentialFilter, obs; kwargs...)
    throw(MethodError(step!, (f, obs)))
end

"""
    estimate(f::AbstractSequentialFilter) -> s

Return the current state estimate, as a child of [`AbstractStateEstimate`](@ref).
"""
function estimate(f::AbstractSequentialFilter)
    throw(MethodError(estimate, (f,)))
end

# ------------------------------------------------------------------------------------------
# Sequential filter prediction API
# ------------------------------------------------------------------------------------------

"""
    AbstractFilterPrediction

Abstract base type for all sequential filters prediction steps.
"""
abstract type AbstractFilterPrediction end

"""
    predict!(p::AbstractFilterPrediction, est::AbstractStateEstimate; kwargs...)

Perform the prediction step of a filter (prior) and update the estimate accordingly.
Prediction performed in-place in `est`.
"""
function predict!(p::AbstractFilterPrediction, est::AbstractStateEstimate; kwargs...)
    throw(MethodError(predict!, (est, p)))
end

# ------------------------------------------------------------------------------------------
# Sequential filter update API
# ------------------------------------------------------------------------------------------

"""
    AbstractFilterUpdate

Abstract base type for all sequential filters update steps.
"""
abstract type AbstractFilterUpdate end

"""
    update!(u::AbstractFilterUpdate, est::AbstractStateEstimate,  z; kwargs...)

Perform the update step of a filter (posterior), given the observation `z`.
Update performed in-place in `est`.
"""
function update!(u::AbstractFilterUpdate, est::AbstractStateEstimate, obs; kwargs...)
    throw(MethodError(update!, (u, est, obs)))
end
