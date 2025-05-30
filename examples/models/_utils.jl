using Filters

function run(
    ::Type{S},
    f::Filters.BaseKalmanFilter,
    z;
    Δt = missing,
    u = missing,
    θ = missing
) where {S <: AbstractKalmanStateEstimate}
    nt = length(z)
    est = Vector{S}(undef, nt)
    est[1] = deepcopy(f.est)

    for i in 2:nt
        if ismissing(u)
            step!(f, z[:, i]; Δt = Δt, θ = θ)
        else
            step!(f, z[:, i]; Δt = Δt, θ = θ, u₋ = u[i-1])
        end
        est[i] = deepcopy(estimate(f))
    end
    return est
end

function plot_estimates(estimates, t, xt, x)
    x̂ = hcat([estimate(e) for e in estimates]...)
    cb = hcat([confidence(e) for e in estimates]...)

    # Position plot
    p1 = plot(t, xt[1, :], label = "\$x_{g}(t)\$")
    plot!(p1, t, x[1, :], label = "\$x(t)\$", xlabel = "\$t\$", ylabel = "\$x(t)\$")
    plot!(p1, t, x̂[1, :], label = "\$\\hat{x}(t)\$", ribbon = cb[1, :], fillalpha = 0.15)
    plot!(
        p1,
        t,
        x̂[1, :] + cb[1, :],
        label = "\$3\\sigma\$",
        linestyle = :dash,
        color = :green
    )
    plot!(p1, t, x̂[1, :] - cb[1, :], label = nothing, linestyle = :dash, color = :green)

    # Velocity plot
    p2 = plot(t, xt[2, :], label = "\$x_{g}(t)\$")
    plot!(p2, t, x[2, :], label = "\$x(t)\$", xlabel = "\$t\$", ylabel = "\$v(t)\$")
    plot!(p2, t, x̂[2, :], label = "\$\\hat{x}(t)\$", ribbon = cb[2, :], fillalpha = 0.15)
    plot!(
        p2,
        t,
        x̂[2, :] + cb[2, :],
        label = "\$3\\sigma\$",
        linestyle = :dash,
        color = :green
    )
    plot!(p2, t, x̂[2, :] - cb[2, :], label = nothing, linestyle = :dash, color = :green)

    p = plot(p1, p2, layout = (2, 1), dpi = 1600, size = (800, 600), framestyle = :box)
    return p
end