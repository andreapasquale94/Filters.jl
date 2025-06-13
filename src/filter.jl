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
    predict!(p::AbstractSequentialFilter; kwargs...)

Perform the prediction step of a filter (prior).
"""
function predict!(p::AbstractSequentialFilter; kwargs...)
    throw(MethodError(predict!, (p)))
end

"""
    update!(u::AbstractFilter, obs; kwargs...)

Perform the update step of a filter (posterior), given the observation `obs`.
"""
function update!(u::AbstractSequentialFilter, obs; kwargs...)
    throw(MethodError(update!, (u, obs)))
end

"""
    step!(f::AbstractSequentialFilter, obs; kwargs...)

Perform a full filtering step (predict + update).
"""
function step!(node::AbstractSequentialFilter, obs; kwargs...)
    throw(MethodError(step!, (node, obs)))
end

"""
    estimate(f::AbstractSequentialFilter)

Return the current state estimate, as a child of [`AbstractStateEstimate`](@ref).
"""
function estimate(node::AbstractSequentialFilter)
    throw(MethodError(estimate, (node,)))
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
    predict!(est::AbstractStateEstimate, p::AbstractFilterPrediction; kwargs...)

Perform the prediction step of a filter (prior) and update the estimate accordingly.
"""
function predict!(est::AbstractStateEstimate, p::AbstractFilterPrediction; kwargs...)
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
    update!(est::AbstractStateEstimate, u::AbstractFilterUpdate, obs; kwargs...)

Perform the update step of a filter (posterior), given the observation `obs`.
"""
function update!(est::AbstractStateEstimate, u::AbstractFilterUpdate, obs; kwargs...)
    throw(MethodError(update!, (est, u, obs)))
end
