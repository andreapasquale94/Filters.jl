
"""
    AbstractSigmaPointsGenerator

Abstract type for sigma-points generators.
"""
abstract type AbstractSigmaPointsGenerator end

"""
    compute!(sp::SigmaPointKalmanState, gen::AbstractSigmaPointsGenerator)

Compute Sigma-Points based on a particular generator `gen`.
"""
function compute!(gen::AbstractSigmaPointsGenerator, sp::SigmaPointKalmanState{T}) where {T}
    throw(MethodError(compute!, (gen, sp)))
end

# ——————————————————————————————————————————————————————————————————————————————————————————
# UKF sigma points
# ------------------------------------------------------------------------------------------

struct UKFSigmaPoints{T} <: AbstractSigmaPointsGenerator
    α::T
    β::T
    κ::T
end

function compute!(gen::UKFSigmaPoints{T}, sp::SigmaPointKalmanState{T}) where {T}
    n = length(sp.x)
    λ = gen.α^2 * (n + gen.κ) - n
    γ = sqrt(n + λ)
    S = cholesky(sp.P).L

    @views begin
        sp.X[:, 1] .= sp.x
        for i in 1:n
            sp.X[:, i+1] .= sp.x .+ γ * S[:, i]
            sp.X[:, i+1+n] .= sp.x .- γ * S[:, i]
        end

        tmp = 1 / (2(n + λ))
        fill!(sp.Wm, tmp)
        fill!(sp.Wc, tmp)
        tmp = λ / (n + λ)
        sp.Wm[1] = tmp
        sp.Wc[1] = tmp + (1 - gen.α^2 + gen.β)
    end
end

# ——————————————————————————————————————————————————————————————————————————————————————————
# CDKF sigma points
# ------------------------------------------------------------------------------------------

struct CDKFSigmaPoints{T} <: AbstractSigmaPointsGenerator
    h::T
end

function compute!(gen::CDKFSigmaPoints{T}, sp::SigmaPointKalmanState{T}) where {T}
    n = length(sp.x)
    γ = gen.h
    S = cholesky(sp.P).L

    @views begin
        sp.X[:, 1] .= sp.x
        for i in 1:n
            sp.X[:, i+1] .= sp.x .+ γ * S[:, i]
            sp.X[:, i+1+n] .= sp.x .- γ * S[:, i]
        end

        tmp = 1 / (2n)
        fill!(sp.Wm, tmp)
        fill!(sp.Wc, tmp)
        sp.Wm[1] = 0
        sp.Wc[1] = 1
    end
end