# ——————————————————————————————————————————————————————————————————————————————————————————
# Resampling API
# ------------------------------------------------------------------------------------------

"""
    AbstractParticleResampling

Abstract type for particle filters resampling logics.
"""
abstract type AbstractParticleResampling <: AbstractFilterUpdate end

"""
    AbstractResamplingAlgorithm

Abstract type for particle filters resampling methods.
"""
abstract type AbstractResamplingAlgorithm end

"""
    resample!(state, algo; kwargs...)

Re-sample according to the given algorithm.
"""
function resample!(state::ParticleState, algo::AbstractResamplingAlgorithm; kwargs...)
    throw(MethodError(resample!, (state, algo)))
end

"""
    AbstractResamplingPolicy

Abstract type for particle filters resampling policy/triggers.
"""
abstract type AbstractResamplingPolicy end

"""
    trigger(state, policy; kwargs...)

Trigger resampling given a the current policy.
"""
function trigger(state::ParticleState, policy::AbstractResamplingPolicy; kwargs...)
    throw(MethodError(trigger, (state, policy)))
end

# ——— Concrete resampling type  ————————————————————————————————————————————————————————————

"""
    Resampling{R, P}

Basic type to store a resampling algorithm and a policy.
"""
struct Resampling{R <: AbstractResamplingAlgorithm, P <: AbstractResamplingPolicy} <:
       AbstractParticleResampling
    algorithm::R
    policy::P
    function Resampling(algorithm::R, policy::P) where {R, P}
        return new{R, P}(algorithm, policy)
    end
end

"""
    resample!(state, logic::Resampling; kwargs...)

Resampling logic implementation, triggered by a given policy and executed accoring to the
given algorithm. Returns a boolean depending associated to the resampling triggering.
"""
function resample!(state::ParticleState, r::Resampling; kwargs...)
    if trigger(state, r.policy; kwargs...)
        resample!(state, r.algorithm; kwargs...)
        return true
    end
    return false
end

# ——— Algorithms  ——————————————————————————————————————————————————————————————————————————

"""
    NoResamplingAlgorithm

Type representing a particle filter resampling strategy that performs no resampling.
"""
struct NoResamplingAlgorithm <: AbstractResamplingAlgorithm end

"""
    resample!(::ParticleState, ::NoResamplingAlgorithm)

Does nothing. Implements the resampling interface but performs no operations.
"""
function resample!(::ParticleState, ::NoResamplingAlgorithm; kwargs...)
    nothing
end

"""
    SystematicResamplingAlgorithm

Type representing *systematic resampling*, a low-variance resampling method.

Systematic resampling uses a single random offset and evenly spaced intervals to select
particles proportionally to their weights. It is more deterministic than multinomial resampling
and typically exhibits lower variance.

## Math

Let the normalized weight of a particle \$i\$ be \$w_i\$.
The resampling process is done as follows:

  - Generate a random offset \$\\xi \\sim \\mathcal{U}(0, 1/N)\$.
  - Compute the positions:

\$\$p_j = \\xi + \\frac{j-1}{N}, \\quad j = 1, 2, \\dots, N\$\$

  - For each position \$\\p_j\$, find the particle index \$i\$ such that:

\$\$p_j \\in [c_{i-1}, c_i)\$\$

where \$c_i\$ is the cumulative sum of the weights:

\$\$c_i = \\sum_{k=1}^i w_k\$\$

Assign the particles accordingly, then reset the weights to uniform (\$1/N\$).
"""
struct SystematicResamplingAlgorithm <: AbstractResamplingAlgorithm end

"""
    resample!(state::ParticleState, ::SystematicResamplingAlgorithm)

Performs systematic resampling on the particles `state`.
"""
function resample!(state::ParticleState, ::SystematicResamplingAlgorithm; kwargs...)
    N = length(state)
    pos = range(rand() / N, step = 1 / N, length = N)
    cum = cumsum!(state.w, state.w)

    i = 1
    @inbounds for j in 1:N
        while pos[j] > cum[i]
            i += 1
        end
        state.p[j, :] .= state.p[i, :]
    end
    fill!(state.w, 1 / N)
    nothing
end

"""
    MultinomialResamplingAlgorithm{T}

Type representing *multinomial resampling* with a preallocated buffer.

This resampling strategy samples particles independently according to their normalized
weights using inverse transform sampling.

## Math

Let the normalized weight of a particle \$i\$ be \$w_i\$.
For each particle \$j\$, sample \$u_j \\in \\mathcal{U}(0, 1)\$, then find the index \$i\$
such that:

\$\$c_{i-1} = \\sum_{k=1}^{i-1} w_k < u_j \\leq \\sum_{k=1}^{i} w_k = c_i\$\$

This corresponds to the resampled particle index \$i\$ for the particle \$j\$.
The resampling process is done as follows:

  - Compute the cumulative sum of normalized weights, \$\\boldsymbol{c}\$.
  - For each particle, draw a random number \$u_j \\in \\mathcal{U}(0, 1)\$.
  - Find the corresponding index \$i\$ using the inverse CDF:

\$\$i = \\text{searchsortedfirst}(\\boldsymbol{c}, u_j)\$\$

Assign the particle \$j\$ to particle \$i\$.
When the particles resampling is finished, reset the weights to uniform (\$1/N\$).
"""
struct MultinomialResamplingAlgorithm{T} <: AbstractResamplingAlgorithm
    cache::Matrix{T}
end

"""
    resample!(state::ParticleState{T}, rs::MultinomialResamplingAlgorithm{T})

Performs multinomial resampling on the particle `state`.
"""
function resample!(
    state::ParticleState{T},
    rs::MultinomialResamplingAlgorithm{T};
    kwargs...
) where {T}
    N = length(state)
    cum = cumsum!(state.w, state.w)

    @inbounds for j in 1:N
        u = rand()
        i = searchsortedfirst(cum, u)
        rs.cache[j, :] .= state.p[i, :]
    end

    state.p .= rs.cache
    fill!(pf.weights, 1 / N)
    nothing
end

# ——— Triggers  ————————————————————————————————————————————————————————————————————————————

"""
    EffectiveSamplesPolicy

Resampling trigger based on a desired effective samples size.
"""
struct EffectiveSamplesPolicy <: AbstractResamplingPolicy
    N::Int
end

function trigger(state::ParticleState, policy::EffectiveSamplesPolicy)
    return neffective(state) < policy.N
end