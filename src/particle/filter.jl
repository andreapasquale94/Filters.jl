
struct ParticleFilter{T,S,O,R} <: AbstractParticleFilter{T}
    particles::Matrix{T}
    weights::Vector{T}
    resampling::R
    state::S
    likelihood::O
    N::Int
end

@inline nparticles(filter::ParticleFilter) = size(filter.particles)[1]
@inline nstates(filter::ParticleFilter) = size(filter.particles)[2]

function predict!(pf::ParticleFilter; kwargs...)
    for i in 1:nparticles(pf)
        pf.particles[i, :] .= pf.state(pf.particles[i, :]; kwargs...)
    end
    nothing
end

function normalize!(pf::ParticleFilter)
    wt = sum(pf.weights)
    if wt == 0.0
        # Avoid division by zero: reset to uniform weights
        fill!(pf.weights, 1 / pf.N)
    else
        pf.weights ./= wt
    end
    nothing
end

function update!(pf::ParticleFilter, z; kwargs...)
    for i in 1:nparticles(pf)
        pf.weights[i] = pf.likelihood(pf.particles[i, :], z; kwargs...)
    end
    normalize!(pf)
end

function resample!(pf::ParticleFilter)
    resample!(pf, pf.resampling)
    nothing
end

"""
    neffective(pf::ParticleFilter)

Compute the current effective sample size.
"""
@inline neffective(pf::ParticleFilter) = 1 / sum(pf.weights .^ 2)
