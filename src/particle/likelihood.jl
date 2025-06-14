# ——————————————————————————————————————————————————————————————————————————————————————————
# Likelihood model API
# ------------------------------------------------------------------------------------------

"""
    AbstractLikelihoodModel 

Abstract type for likelihood models.
"""
abstract type AbstractLikelihoodModel <: AbstractModel end

"""
    likelihood(m::AbstractLikelihoodModel, x, z)

Compute the likelihood of the observation `z` given the state `x` under model `m`.
"""
function likelihood(m::AbstractLikelihoodModel, x, z; kwargs...)
    throw(MethodError(likelihood, (m, x, z)))
end