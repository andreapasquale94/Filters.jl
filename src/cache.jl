"""
    AbstractFilterCache

Abstract type for filter caches.
"""
abstract type AbstractFilterCache end

"""
    predict!(cf::AbstractFilterCache; kwargs...) 

Perform the time-update step (prior) and store the result in the cache.
"""
function predict!(cf::AbstractFilterCache; kwargs...)
    throw(MethodError(predict!, (cf,)))
end

"""
    update!(cf::AbstractFilterCache, z; kwargs...)

Perform the measurement-update step (posterior) and store the result in the cache.
"""
function update!(cf::AbstractFilterCache, z; kwargs...)
    throw(MethodError(update!, (filter, z)))
end
