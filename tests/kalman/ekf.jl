using Test 
using LinearAlgebra
using Filters

function dynamics!(xnew, F, x, u, t, tn)
    # Simple constant velocity model: x = [position; velocity]
    dt = tn - t
    A = [1.0  dt;
         0.0  1.0]
    xnew .= A * x
    F .= A
end

function observation!(z, H, x, u, t)
    # Observe position only
    Htmp = [1.0 0.0]
    z .= Htmp * x
    H .= Htmp
end

@testset "ExtendedKalmanFilter Tests" begin
    T = Float64
    n = 2  # state size
    m = 1  # measurement size
    Δt = 1.0
    t0 = 0.0

    # Initial state and covariance
    x0 = [0.0, 1.0]
    P0 = Matrix{T}(I, n, n)
    Q = 0.01I(n)
    R = 0.1I(m)

    # Create EKF instance
    ekf = ExtendedKalmanFilter{T}(n, m, Δt, t0, dynamics!, observation!, x0, P0, Q, R)

    @test size(ekf.x) == (2,)
    @test size(ekf.P) == (2, 2)
    @test ekf.t[1] == t0

    # === Prediction test ===
    predict!(ekf)
    @test isapprox(ekf.x, [1.0, 1.0], atol=1e-8)
    @test ekf.t[1] == t0 + Δt

    # === Update test ===
    z = [1.2]  # new measurement
    update!(ekf, z)

    # After update, state should move toward measurement
    @test abs(ekf.x[1] - 1.2) < abs(1.0 - 1.2)
    @test isapprox(ekf.P, ekf.P', atol=1e-10)  # covariance should remain symmetric
end