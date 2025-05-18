mutable struct KFCache{F<:AbstractFilter,X,P,L} <: AbstractFilterCache
    f::F
    x::X # state estimate
    cov::P # covariance
    log::L # log-likelihood
    con::X # confidence
    k::Int # current index
    function KFCache(f::F, x::X, cov::P, log::L, con::X) where {F,X,P,L}
        return new{F,X,P,L}(f, x, cov, log, con, 1)
    end
end

function Base.empty!(cache::KFCache)
    empty!(cache.x)
    empty!(cache.cov)
    empty!(cache.log)
    empty!(cache.con)
    return nothing
end

function Base.resize!(cache::KFCache, n)
    resize!(cache.x, n)
    resize!(cache.cov, n)
    resize!(cache.log, n)
    resize!(cache.con, n)
    return nothing
end

function predict!(cache::KFCache; kwargs...)
    predict!(cache.f; kwargs...)
    return nothing
end

"""
    KalmanFilterCache{F, T}

A cache for Kalman filters. 
The cache stores the state estimate, covariance, log-likelihood, and confidence intervals for each update step.
"""
const KalmanFilterCache{F,T} = KFCache{F,Vector{Vector{T}},Vector{Matrix{T}},Vector{T}}

function KalmanFilterCache(filter::F) where F
    T = ftype(filter)
    x = Vector{Vector{T}}(undef, 0)
    cov = Vector{Matrix{T}}(undef, 0)
    log = Vector{T}(undef, 0)
    con = Vector{Vector{T}}(undef, 0)

    push!(x, estimate(filter))
    push!(cov, covariance(filter))
    push!(log, loglikelihood(filter))
    push!(con, 3sqrt.(diag(cov[end])))
    return KFCache(filter, x, cov, log, con)
end

function Base.show(io::IO, c::KalmanFilterCache{F, T}) where {F, T}
    println(io, "KalmanFilterCache{$T} for $F with size $(length(c.x))")
    nothing
end

"""
    KalmanFilterSCache{F, N, T}

A cache for Kalman filters with static arrays.
The cache stores the state estimate, covariance, log-likelihood, and confidence intervals for each update step.
"""
const KalmanFilterSCache{F,N,T} = KFCache{F,Vector{SVector{N,T}},Vector{SMatrix{N,N,T}},Vector{T}}

function KalmanFilterSCache(filter::F) where F
    T = ftype(filter)
    N = nx(filter)
    x = Vector{SVector{N,T}}(undef, 0)
    cov = Vector{SMatrix{N,N,T}}(undef, 0)
    log = Vector{T}(undef, 0)
    con = Vector{SVector{N,T}}(undef, 0)
    return KFCache(filter, x, cov, log, con)
end

function update!(cache::KalmanFilterCache{F,T}, z; kwargs...) where {F,T}
    update!(cache.f, z; kwargs...)
    x̂ = estimate(cache.f)
    P = covariance(cache.f)

    cache.k += 1 # update the index
    if cache.k > length(cache.x)
        # If the cache is full, resize it
        push!(cache.x, copy(x̂))
        push!(cache.cov, copy(P))
        push!(cache.log, loglikelihood(cache.f))
        push!(cache.con, 3sqrt.(diag(P)))
    else
        # If the cache is not full, just update the existing elements
        cache.x[cache.k] = copy(x̂)
        cache.cov[cache.k] = copy(P)
        cache.log[cache.k] = loglikelihood(cache.f)
        cache.con[cache.k] = 3sqrt.(diag(P))
    end
    return nothing
end

function update!(cache::KalmanFilterSCache{F,N,T}, z; kwargs...) where {F,N,T}
    update!(cache.f, z; kwargs...)
    x̂ = estimate(cache.f)
    P = covariance(cache.f)

    cache.k += 1 # update the index
    if cache.k > length(cache.x)
        # If the cache is full, resize it
        push!(cache.x, SVector{N,T}(x̂))
        push!(cache.cov, SMatrix{N,N,T}(P))
        push!(cache.log, loglikelihood(cache.f))
        push!(cache.con, SVector{N,T}(3sqrt.(diag(P))))
    else
        # If the cache is not full, just update the existing elements
        cache.x[cache.k] = SVector{N,T}(x̂)
        cache.cov[cache.k] = SMatrix{N,N,T}(P)
        cache.log[cache.k] = loglikelihood(cache.f)
        cache.con[cache.k] = SVector{N,T}(3sqrt.(diag(P)))
    end
    return nothing
end