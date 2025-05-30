"""
    cholesky_downdate_rank1!(L::LowerTriangular, u)

Downdates the Cholesky factor `L` of a positive definite matrix `A` by a rank-1 update `A - uu'`.

This function uses Givens rotations to perform the downdate.
The matrix `u` is assumed to be of size `n`, where `n` is the size of the matrix `A`.
"""
function cholesky_downdate_rank1!(L::LowerTriangular{T}, u::Vector{T}) where {T}
    n = length(u)
    for k in 1:n
        r = sqrt(L[k, k]^2 - u[k]^2)
        if !(isreal(r) && r > 0)
            throw(ErrorException("matrix not positive definite after downdate"))
        end
        c = r / L[k, k]
        s = u[k] / L[k, k]
        L[k, k] = r
        for j in k+1:n
            L[j, k] = (L[j, k] - s * u[j]) / c
            u[j] = c * u[j] - s * L[j, k]
        end
    end
    return L
end

"""
    cholesky_downdate!(L::LowerTriangular, U; buffer)

Downdates the Cholesky factor `L` of a positive definite matrix `A` by a rank-`k` update `A - UU'`.
The matrix `U` is assumed to be of size `n x k`, where `n` is the size of the matrix `A`.
The function modifies `L` in place.
"""
function cholesky_downdate!(
    L::LowerTriangular{T},
    U::Matrix{T};
    buffer::Vector{T} = zeros(T, size(U, 1))
) where {T <: AbstractFloat}
    n, k = size(U)
    @assert size(L, 1) == n && size(L, 2) == n
    @assert length(buffer) == n
    for j in 1:k
        @inbounds copyto!(buffer, view(U, :, j))
        cholesky_downdate_rank1!(L, buffer)
    end
    return nothing
end

"""
    cholesky_update_rank1!(L::LowerTriangular, u)

Updates the Cholesky factor `L` of a positive definite matrix `A` by a rank-1 update `A + u*u'`.

This function uses Givens rotations to perform the update.
The matrix `u` is assumed to be of size `n`, where `n` is the size of the matrix `A`.
"""
function cholesky_update_rank1!(L::LowerTriangular{T}, u::Vector{T}) where {T}
    n = length(u)
    for k in 1:n
        r = hypot(L[k, k], u[k])
        c = L[k, k] / r
        s = u[k] / r
        L[k, k] = r
        for j in k+1:n
            L[j, k], u[j] = c * L[j, k] + s * u[j], -s * L[j, k] + c * u[j]
        end
    end
    return L
end

"""
    cholesky_update!(L::LowerTriangular, U::Matrix; buffer)

Performs a rank-k update on the Cholesky factor `L`, corresponding to updating
`A_new = A + U*U'`.

Each column of `U` is applied sequentially as a rank-1 update.
Modifies `L` in-place.
"""
function cholesky_update!(
    L::LowerTriangular{T},
    U::Matrix{T};
    buffer::Vector{T} = zeros(T, size(U, 1))
) where {T <: AbstractFloat}
    n, k = size(U)
    @assert size(L, 1) == n && size(L, 2) == n
    for j in 1:k
        @inbounds copyto!(buffer, view(U, :, j))
        cholesky_update_rank1!(L, buffer)
    end
    return nothing
end