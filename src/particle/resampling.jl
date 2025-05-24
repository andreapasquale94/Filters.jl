
# ------------------------------------------------------------------------------------------
# Algorithms
# ------------------------------------------------------------------------------------------

"""
    NoResamplingAlgorithm

Type representing a particle filter resampling strategy that performs no resampling.

This is useful in cases where resampling is disabled or deferred, such as for debugging,
evaluating filter performance without resampling noise, or implementing custom resampling criteria.

In this case, no change is made to the particle set or the weights. 
"""
struct NoResamplingAlgorithm <: AbstractResamplingAlgorithm{Nothing} end

"""
    resample!(pf::AbstractParticleFilter, ::NoResamplingAlgorithm)

Does nothing. Implements the resampling interface but performs no operations.

The particle set and weights remain unchanged.
"""
function resample!(::AbstractParticleFilter, ::NoResamplingAlgorithm; kwargs...)
    nothing
end

"""
    SystematicResamplingAlgorithm

Type representing *systematic resampling*, a low-variance resampling method
commonly used in particle filters.

Systematic resampling uses a single random offset and evenly spaced intervals
to select particles proportionally to their weights. It is more deterministic
than multinomial resampling and typically exhibits lower variance.

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
struct SystematicResamplingAlgorithm <: AbstractResamplingAlgorithm{Nothing} end

"""
    resample!(pf::ParticleFilter, ::SystematicResamplingAlgorithm)

Performs systematic resampling on the particle filter `pf`.
"""
function resample!(pf::ParticleFilter, ::SystematicResamplingAlgorithm; kwargs...)
    N = nparticles(pf)
    pos = range(rand() / N, step=1 / N, length=N)
    cum = cumsum!(pf.weights, pf.weights)

    i = 1
    for j in 1:N
        while pos[j] > cum[i]
            i += 1
        end
        pf.particles[j, :] .= pf.particles[i, :]
    end
    fill!(pf.weights, 1 / N)
    nothing
end

"""
    MultinomialResamplingAlgorithm{T}

Type representing *multinomial resampling* with a preallocated buffer.

This resampling strategy samples particles independently according to
their normalized weights using inverse transform sampling. It uses a buffer
matrix to avoid in-place overwrites and minimize allocations.

## Math

Let the normalized weight of a particle \$i\$ be \$w_i\$.
For each particle \$j\$, sample \$u_j \\in \\mathcal{U}(0, 1)\$, then find the index \$i\$ such that:

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
struct MultinomialResamplingAlgorithm{T} <: AbstractResamplingAlgorithm{T}
    buffer::Matrix{T}
end

"""
    resample!(pf::ParticleFilter{T}, rs::MultinomialResamplingAlgorithm{T})

Performs multinomial resampling on the particle filter `pf` using the buffer in `rs`.
"""
function resample!(pf::ParticleFilter{T}, rs::MultinomialResamplingAlgorithm{T}; kwargs...) where T
    N = nparticles(pf)
    cum = cumsum!(pf.weights, pf.weights)

    for j in 1:N
        u = rand()
        i = searchsortedfirst(cum, u)
        rs.buffer[j, :] .= pf.particles[i, :]
    end

    pf.particles .= rs.buffer
    fill!(pf.weights, 1 / N)
    nothing
end

# ------------------------------------------------------------------------------------------
# Triggers
# ------------------------------------------------------------------------------------------

"""
    EffectiveSamplePolicy

Resampling trigger based on the effective samples size.
"""
struct EffectiveSamplePolicy <: AbstractResamplingPolicy
    Neff::Int
end

function trigger(filter::ParticleFilter, p::EffectiveSamplePolicy)
    return neffective(filter) < p.Neff
end