using Test
using Filters
using LinearAlgebra

@testset "Constructor" begin
    for T in (Float16, Float32, Float64)
        n = 4   # State dimension
        m = 2   # Measurement dimension

        F = Matrix{T}(I, n, n)
        H = randn(T, m, n)
        Q = Matrix{T}(1e-3I, n, n)
        R = Matrix{T}(1e-2I, m, m)
        B = randn(T, n, 1)
        D = randn(T, m, 1)

        x0 = randn(T, n)
        P0 = Matrix{T}(I, n, n)

        kf = KalmanFilter{T}(n, m, x0, P0, F, B, Q, H, D, R)

        @test isa(kf, KalmanFilter{T})
        @test size(kf.x) == (n,)
        @test size(kf.P) == (n, n)
        @test size(kf.z) == (m,)
        @test size(kf.K) == (n, m)
    end
end;

@testset "Execution" begin
    T = Float64
    n = 4   # State dimension
    m = 2   # Measurement dimension

    F = Matrix{T}(I, n, n)
    H = randn(T, m, n)
    Q = Matrix{T}(1e-3I, n, n)
    R = Matrix{T}(1e-2I, m, m)
    B = randn(T, n, 1)
    D = randn(T, m, 1)

    x0 = randn(T, n)
    P0 = Matrix{T}(I, n, n)

    kf = KalmanFilter{T}(n, m, x0, P0, F, B, Q, H, D, R)

    # Control and observation
    u = randn(1)
    z = H * x0 + D * u + randn(m) * sqrt(R[1, 1])

    @testset "Prediction" begin
        x_prev = copy(kf.x)
        P_prev = copy(kf.P)
        predict!(kf, u=u)
        @test kf.x ≈ F * x_prev + B * u atol = 1e-10
        @test kf.P ≈ F * P_prev * F' + Q atol = 1e-10
    end

    @testset "Update" begin
        update!(kf, z, u=u)
        @test size(kf.y) == (m,)
        @test size(kf.S) == (m, m)
        @test isposdef(kf.S)

        # Innovation check
        @test kf.y ≈ z - kf.z atol=1e-10

        # Kalman gain should reduce covariance
        @test isapprox(kf.P, kf.P', atol=1e-12)  # Symmetry
        @test all(eigvals(kf.P) .>= 0)           # Positive semi-definite
    end
end;