"""
    AbstractSigmaPointsGenerator

Abstract type for sigma-points generators.
"""
abstract type AbstractSigmaPointsGenerator end

"""
    compute!(gen::AbstractSigmaPointsGenerator, X, Wm, Wc, x, P)

Compute Sigma-Points based on a particular generator `gen`, based on the mean `μ` and covariance
matrix `P` and store them in `X` together with the weights `Wm` and `Wc`.
"""
function compute!(
    gen::AbstractSigmaPointsGenerator,
    X,
    Wm,
    Wc,
    x,
    P::LowerTriangular{<:Number}
)
    throw(MethodError(compute!, (gen, X, Wm, Wc, x, P)))
end

function compute!(
    gen::AbstractSigmaPointsGenerator,
    X,
    Wm,
    Wc,
    x,
    P::AbstractMatrix{<:Number}
)
    compute!(gen, X, Wm, Wc, x, cholesky(Hermitian(P)).L)
    nothing
end

# ——————————————————————————————————————————————————————————————————————————————————————————
# UKF sigma points
# ------------------------------------------------------------------------------------------

struct UKFSigmaPoints{T} <: AbstractSigmaPointsGenerator
    α::T
    β::T
    κ::T
end

function compute!(
    gen::UKFSigmaPoints{T},
    X::AbstractMatrix{N},
    Wm::AbstractVector{N},
    Wc::AbstractVector{N},
    x::AbstractVector{N},
    S::LowerTriangular{N}
) where {T <: Number, N <: Number}
    n = length(x)
    λ = gen.α^2 * (n + gen.κ) - n
    γ = sqrt(n + λ)

    @views begin
        X[:, 1] .= x
        for i in 1:n
            X[:, i+1] .= x .+ γ * S[:, i]
            X[:, i+1+n] .= x .- γ * S[:, i]
        end

        tmp = 1 / (2(n + λ))
        fill!(Wm, tmp)
        fill!(Wc, tmp)
        tmp = λ / (n + λ)
        Wm[1] = tmp
        Wc[1] = tmp + (1 - gen.α^2 + gen.β)
    end
    nothing
end

# ——————————————————————————————————————————————————————————————————————————————————————————
# CDKF sigma points
# ------------------------------------------------------------------------------------------

struct CDKFSigmaPoints{T} <: AbstractSigmaPointsGenerator
    h::T
end

function compute!(
    gen::CDKFSigmaPoints{T},
    X::AbstractMatrix{N},
    Wm::AbstractVector{N},
    Wc::AbstractVector{N},
    x::AbstractVector{N},
    S::LowerTriangular{N}
) where {T <: Number, N <: Number}
    n = length(x)
    γ = gen.h

    @views begin
        X[:, 1] .= x
        for i in 1:n
            X[:, i+1] .= x .+ γ * S[:, i]
            X[:, i+1+n] .= x .- γ * S[:, i]
        end

        tmp = 1 / (2n)
        fill!(Wm, tmp)
        fill!(Wc, tmp)
        Wm[1] = 0
        Wc[1] = 1
    end
    nothing
end