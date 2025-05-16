struct _KalmanFilterCache{F<:AbstractFilter,X,P,L,C} <: AbstractFilterCache
    f::F
    x::X # state estimate
    cov::P # covariance
    log::L # log-likelihood
    con::C # confidence
end

function predict!(cache::_KalmanFilterCache; kwargs...)
    predict!(cache.f; kwargs...)
    return nothing
end

"""
    KalmanFilterCache{F, T}

A cache for Kalman filters. 
The cache stores the state estimate, covariance, log-likelihood, and confidence intervals for each update step.
"""
const KalmanFilterCache{F,T} = _KalmanFilterCache{F,Vector{Vector{T}},Vector{Matrix{T}},Vector{T},Vector{Vector{T}}}

function KalmanFilterCache{T}(filter::F) where {F,T}
    x = Vector{Vector{T}}(undef, 0)
    cov = Vector{Matrix{T}}(undef, 0)
    log = Vector{T}(undef, 0)
    con = Vector{Vector{T}}(undef, 0)
    return KalmanFilterCache{F,T}(filter, x, cov, log, con)
end

"""
    KalmanFilterSCache{F, N, T}

A cache for Kalman filters with static arrays.
The cache stores the state estimate, covariance, log-likelihood, and confidence intervals for each update step.
"""
const KalmanFilterSCache{F,N,T} = _KalmanFilterCache{F,Vector{SVector{N,T}},Vector{SMatrix{N,N,T}},Vector{T},Vector{SVector{N,T}}}

function KalmanFilterSCache{N,T}(f::F) where {F,N,T}
    x = Vector{SVector{N,T}}(undef, 0)
    cov = Vector{SMatrix{N,N,T}}(undef, 0)
    log = Vector{T}(undef, 0)
    con = Vector{SVector{N,T}}(undef, 0)
    return KalmanFilterSCache{F,N,T}(f, x, cov, log, con)
end

function update!(cache::KalmanFilterCache{F,T}, z; kwargs...) where {F,T}
    update!(cache.f, z; kwargs...)
    x̂ = estimate(cache.f)
    P = covariance(cache.f)
    push!(cache.x, x̂)
    push!(cache.cov, P)
    push!(cache.log, loglikelihood(cache.f))
    push!(cache.con, 3sqrt(diag(P)))
    return nothing
end

function update!(cache::KalmanFilterSCache{F,N,T}, z; kwargs...) where {F,N,T}
    update!(cache.f, z; kwargs...)
    x̂ = estimate(cache.f)
    P = covariance(cache.f)
    push!(cache.x, SVector{N,T}(x̂))
    push!(cache.cov, SMatrix{N,N,T}(P))
    push!(cache.log, loglikelihood(cache.f))
    push!(cache.con, SVector{N,T}(3sqrt.(diag(P))))
    return nothing
end