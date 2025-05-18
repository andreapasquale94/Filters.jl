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
    update!(cf::AbstractFilterCache, i, z; kwargs...)

Perform the measurement-update step (posterior) and store the result in the cache.
"""
function update!(cf::AbstractFilterCache, z; kwargs...)
    throw(MethodError(update!, (filter, z)))
end

function update!(cf::AbstractFilterCache, i, z; kwargs...)
    throw(MethodError(update!, (filter, i, z)))
end

"""
    empty!(cf::AbstractFilterCache) 

Empty the cache.
"""
function Base.empty!(cf::AbstractFilterCache)
    throw(MethodError(Base.empty!, (cf,)))
end

"""
    resize!(cf::AbstractFilterCache, n)

Resize the cache to hold `n` elements.
"""
function Base.resize!(cf::AbstractFilterCache, n)
    throw(MethodError(Base.resize!, (cf, n)))
end