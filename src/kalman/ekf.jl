
# f!( xn, Φ, x, u, t, tn )
const FilterStateFunction{T} = FunctionWrapper{Nothing,Tuple{Vector{T},Matrix{T},Vector{T},Vector{T},T, T}};

# f!( z, H, x, u, t )
const FilterObservationFunction{T} = FunctionWrapper{Nothing,Tuple{Vector{T},Matrix{T},Vector{T},Vector{T},T}};

struct ExtendedKalmanFilter{T, FT, JT} <: AbstractKalmanFilter{T}
    n::Int
    m::Int
    Δt::T
    f::FT
    h::JT

    # Cache
    x::Vector{T}
    P::Matrix{T}
    F::Matrix{T}
    H::Matrix{T}
    Q::Matrix{T}
    R::Matrix{T}

    z::Vector{T}
    y::Vector{T}
    S::Matrix{T}
    K::Matrix{T}
    t::MVector{2, T}
end

function ExtendedKalmanFilter{T}(n, m, Δt, t0, f, h, x0, P0, Q, R) where T
    F0 = zeros(T, n, n)
    H0 = zeros(T, m, n)
    z0 = zeros(T, m)
    y0 = zeros(T, m)
    S0 = zeros(T, m, m)
    K0 = zeros(T, n, m)
    
    fw = FilterStateFunction{T}(f)
    hw = FilterObservationFunction{T}(h)
    return ExtendedKalmanFilter{T, typeof(fw), typeof(hw)}(
        n, m, Δt, fw, hw, x0, P0, F0, H0, Q, R, z0, y0, S0, K0, [t0, t0+Δt]
    )
end

function predict!(kf::ExtendedKalmanFilter{T}; u=nothing) where {T}
    # 1. State prediction time update
    if u !== nothing
        kf.f(kf.x, kf.F, copy(kf.x), u, kf.t[1], kf.t[2])
    else
        kf.f(kf.x, kf.F, copy(kf.x), T[], kf.t[1], kf.t[2])
    end
    # 2. Covariance prediction time update
    kf.P .= kf.F * kf.P * kf.F' .+ kf.Q
    kf.t .+= kf.Δt
    return nothing
end

function update!(kf::ExtendedKalmanFilter{T}, z::AbstractVector{T}; u=nothing) where {T}
    # 3. Measurement prediction
    if u !== nothing
        kf.h(kf.z, kf.H, kf.x, u, kf.t[1])
    else
        kf.h(kf.z, kf.H, kf.x, T[], kf.t[1])
    end

    # Compute the innovation
    kf.y .= z .- kf.z
    # Compute the innovation covariance
    PHT = kf.P * kf.H'
    kf.S .= kf.H * PHT .+ kf.R

    # 4. Compute the Kalman gain
    kf.K .= PHT / kf.S

    # 5. State update
    kf.x .+= kf.K * kf.y

    # 6. Covariance update
    IKH = I - kf.K * kf.H
    kf.P .= IKH * kf.P * IKH' .+ kf.K * kf.R * kf.K'
    return nothing
end