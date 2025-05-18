using LinearAlgebra
using Test
using Filters

 # Helper function: create SPD matrix and its Cholesky lower factor
function generate_spd_matrix(n::Int)
    A = randn(n, n)
    P = A * A' + n * I
    return P, cholesky(P).L
end

@testset "Cholesky Downdate" verbose=true begin
    @testset "Rank-1" begin
        for trial in 1:100
            n = rand(3:10)
            P, L = generate_spd_matrix(n)

            u = randn(n)
            P_new = P - u * u'

            if isposdef(P_new)
                L_downdate = LowerTriangular(copy(L))
                Filters.cholesky_downdate_rank1!(L_downdate, copy(u))
                P_downdated = L_downdate * L_downdate'
                @test isapprox(P_downdated, P_new; atol=1e-8, rtol=1e-8)
            end
        end
    end;

    @testset "Rank-k" begin
        for trial in 1:100
            n = rand(3:10)
            P, L = generate_spd_matrix(n)

            k = rand(1:n)
            U = randn(n, k)
            P_new = P - U * U'

            if isposdef(P_new)
                L_downdate = LowerTriangular(copy(L))
                Filters.cholesky_downdate!(L_downdate, copy(U))
                P_downdated = L_downdate * L_downdate'
                @test isapprox(P_downdated, P_new; atol=1e-8, rtol=1e-8)
            end
        end
    end;
end;

@testset "Cholesky Update" verbose=true begin
    @testset "Rank-1" begin
        for trial in 1:100
            n = rand(3:10)
            P, L = generate_spd_matrix(n)

            u = randn(n)
            P_new = P + u * u'

            if isposdef(P_new)
                L_update = LowerTriangular(copy(L))
                Filters.cholesky_update_rank1!(L_update, copy(u))
                P_updated = L_update * L_update'
                @test isapprox(P_updated, P_new; atol=1e-8, rtol=1e-8)
            end
        end
    end;

    @testset "Rank-k" begin
        for trial in 1:100
            n = rand(3:10)
            P, L = generate_spd_matrix(n)

            k = rand(1:n)
            U = randn(n, k)
            P_new = P + U * U'

            if isposdef(P_new)
                L_update = LowerTriangular(copy(L))
                Filters.cholesky_update!(L_update, copy(U))
                P_updated = L_update * L_update'
                @test isapprox(P_updated, P_new; atol=1e-8, rtol=1e-8)
            end
        end
    end;
end;