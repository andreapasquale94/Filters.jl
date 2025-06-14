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
    resample!(algo, state; kwargs...)

Re-sample according to the given algorithm.
"""
function resample!(algo::AbstractResamplingAlgorithm, state::ParticleState; kwargs...)
    throw(MethodError(resample!, (algo, state)))
end

"""
    AbstractResamplingPolicy

Abstract type for particle filters resampling policy/triggers.
"""
abstract type AbstractResamplingPolicy end

"""
    trigger(policy, state; kwargs...)

Trigger resampling given a the current policy.
"""
function trigger(policy::AbstractResamplingPolicy, state::ParticleState; kwargs...)
    throw(MethodError(trigger, (policy, state)))
end

# ——— Concrete resampling type  ————————————————————————————————————————————————————————————

"""
    Resampling{A, P}

Basic type to store a resampling algorithm and a policy.
"""
struct Resampling{A <: AbstractResamplingAlgorithm, P <: AbstractResamplingPolicy} <:
       AbstractParticleResampling
    algorithm::A
    policy::P
    function Resampling(algorithm::A, policy::P) where {A, P}
        return new{A, P}(algorithm, policy)
    end
end

"""
    resample!(r::Resampling, state::ParticleState; kwargs...)

Resampling logic implementation, triggered by a given policy and executed accoring to the
given algorithm. Returns a boolean depending associated to the resampling triggering.
"""
function resample!(r::Resampling, state::ParticleState; kwargs...)
    if trigger(r.policy, state; kwargs...)
        resample!(r.algorithm, state; kwargs...)
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

function resample!(::NoResamplingAlgorithm, ::ParticleState; kwargs...)
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

function resample!(::SystematicResamplingAlgorithm, state::ParticleState; kwargs...)
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

function resample!(
    rs::MultinomialResamplingAlgorithm{T},
    state::ParticleState{T};
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


"""
    StratifiedResampling <: AbstractResamplingStrategy

Type representing *stratified resampling*.

This strategy reduces variance by stratifying the unit interval [0, 1] into equal-width
subintervals and drawing one sample from each. It ensures that the resampled indices
are more uniformly distributed compared to standard multinomial resampling.

## Mathematical Description

Let the normalized weights be ``w_i`` and define cumulative sums ``c``.
Let ``N`` be the number of particles. Define ``N`` stratified bins of width ``1/N``:

```math
u_j \\sim \\mathcal{U}[(j - 1)/N, j/N)]
```
Then for each ``u_j``, compute the ancestor index. 
The j-th resampled particle becomes a copy of particle ``i``. After resampling, weights are 
reset to the uniform distribution ``1/N``.
"""
struct StratifiedResamplingAlgorithm{T} <: AbstractResamplingAlgorithm
    idx::Vector{Int}
end

function resample!(
    r::StratifiedResamplingAlgorithm{T},
    state::ParticleState{T};
    kwargs...
) where {T}
    N = length(state)
    pos = range(rand() / N, step = 1 / N, length = N)
    cum = cumsum!(state.w, state.w)

    @inbounds for j in 1:N
        u = pos[j]
        i = searchsortedfirst(cum, u)
        r.idx[j] = i
    end

    @inbounds for j in 1:N
        state.p[j, :] .= state.p[r.idx[j], :]
    end
    fill!(state.w, 1 / N)
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

function trigger(policy::EffectiveSamplesPolicy, state::ParticleState; kwargs...)
    return neffective(state) < policy.N
end