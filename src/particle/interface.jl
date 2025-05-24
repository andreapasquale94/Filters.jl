"""
    AbstractParticleFilter{T}

Abstract type for all particle filters implementations.
"""
abstract type AbstractParticleFilter{T} <: AbstractSequentialFilter{T} end

"""
    nparticles(filter::AbstractParticleFilter)::Int

Return the number of particles.
"""
@inline nparticles(filter::AbstractParticleFilter) = throw(MethodError(nparticles, (filter,)))

@inline islinear(::AbstractParticleFilter) = false

# ------------------------------------------------------------------------------------------
# Resampling API 
# ------------------------------------------------------------------------------------------

"""
    AbstractResamplingLogic

Abstract type for particle filters resampling logics.
"""
abstract type AbstractResampling end

"""
    AbstractResamplingAlgorithm{T}

Abstract type for particle filters resampling methods.
"""
abstract type AbstractResamplingAlgorithm{T} end

"""
    resample!(filter, algo; kwargs...)

Re-sample according to the given algorithm.
"""
function resample!(filter::AbstractParticleFilter, algo::AbstractResamplingAlgorithm; kwargs...)
    throw(MethodError(resample!, (filter, algo)))
end

"""
    AbstractResamplingPolicy

Abstract type for particle filters resampling policy/triggers.
"""
abstract type AbstractResamplingPolicy end

"""
    trigger(filter, policy; kwargs...)

Trigger resampling given a the current policy.
"""
function trigger(filter::AbstractParticleFilter, policy::AbstractResamplingPolicy; kwargs...)
    throw(MethodError(trigger, (filter, policy)))
end

# ----

"""
    Resampling{R, P}

Basic type to store a resampling algorithm and a policy.
"""
struct Resampling{R<:AbstractResamplingAlgorithm,P<:AbstractResamplingPolicy} <: AbstractResampling
    algorithm::R
    policy::P
end

"""
    resample!(filter, logic::Resampling; kwargs...)

Resampling logic implementation, triggered by a given policy and executed accoring to the 
given algorithm.
"""
function resample!(filter::AbstractParticleFilter, r::Resampling; kwargs...)
    if trigger(filter, r.policy; kwargs...)
        resample!(filter, r.algorithm; kwargs...)
    end
end

