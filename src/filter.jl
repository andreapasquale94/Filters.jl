abstract type AbstractFilter end

# ——————————————————————————————————————————————————————————————————————————————————————————
# Sequential filter API
# ------------------------------------------------------------------------------------------

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
    step!(node::AbstractSequentialFilter, obs; kwargs...)

Perform a full filtering step (predict + update).
"""
function step!(node::AbstractSequentialFilter, obs; kwargs...)
    throw(MethodError(step!, (node, obs)))
end

"""
    estimate(filter::AbstractSequentialFilter)

Return the current best estimate.
"""
function estimate(node::AbstractSequentialFilter)
    throw(MethodError(estimate, (node,)))
end

# ------------------------------------------------------------------------------------------
# Sequential filter prediction API
# ------------------------------------------------------------------------------------------

abstract type AbstractFilterPrediction <: AbstractSequentialFilter end

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

abstract type AbstractFilterUpdate <: AbstractSequentialFilter end

"""
    update!(est::AbstractStateEstimate, u::AbstractFilterUpdate, obs; kwargs...)

Perform the update step of a filter (posterior), given the observation `obs`.
"""
function update!(est::AbstractStateEstimate, u::AbstractFilterUpdate, obs; kwargs...)
    throw(MethodError(update!, (est, u, obs)))
end
