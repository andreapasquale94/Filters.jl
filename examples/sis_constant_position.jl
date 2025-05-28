using LinearAlgebra
using Distributions
using Filters
using Plots
using Random
Plots.gr()

# ----
# Models

struct SimpleStateTransition{T} <: AbstractStateModel
    P::T
    b::T
end

function Filters.transition!(m::SimpleStateTransition, xn, x)
    @. xn = x + m.b + randn() * m.P
    nothing
end

struct SimpleLikelyhoodModel{T} <: AbstractLikelihoodModel
    P::T
end

function Filters.likelihood(m::SimpleLikelyhoodModel, x, z)
    z_pred = atan((r * sin(x[1]) + y0), (r * cos(x[1]) + x0))
    return pdf(Normal(z_pred, m.P), z)[1]
end

# ---- 
# Setup

Random.seed!(1)

μx0 = deg2rad(50);
Σx0 = deg2rad(10);
Σw = deg2rad(10);
Σv = deg2rad(5);

x0, y0 = 500, 500;
r = 200;
ω = deg2rad(2);

nIter = 200;
x = zeros(nIter + 1, 1);
z = zeros(nIter, 1);
w = randn(nIter, 1) * Σw;
v = randn(nIter, 1) * Σv;

for k in 1:nIter
    x[k+1] = x[k] + ω + w[k]
    z[k] = atan(r * sin(x[k]) + y0, r * cos(x[k]) + x0) + v[k]
end;
x = x[1:nIter];

# ----
# Filter 

Np = 1000;
p0 = rand(Normal(μx0, Σx0), Np, 1);
w0 = fill(1.0 / Np, Np);

T = Float64;
pf = BootstrapParticleFilter{T}(
    ParticleState(p0, w0),
    BootstrapParticleFilterPrediction{T}(SimpleStateTransition(Σw, ω)),
    BootstrapParticleFilterUpdate{T}(SimpleLikelyhoodModel(Σv)),
    Resampling(SystematicResamplingAlgorithm(), EffectiveSamplesPolicy(800))
);

# Cache 
xp = zeros(Np, nIter);
wp = zeros(Np, nIter);
xHat = zeros(nIter, 1);
xp[:, 1] .= p0;
wp[:, 1] .= w0;
xHat[1, :] .= Filters.estimate(pf.est);

for k in 2:nIter
    step!(pf, z[k, :])
    xHat[k, :] .= Filters.estimate(pf.est)
    xp[:, k] .= deepcopy(pf.est.p[:, 1])
    wp[:, k] .= deepcopy(pf.est.w)
end;

p = plot(dpi = 1600, size = (800, 600), framestyle = :box);
for i in 1:Np
    plot!(p, 1:nIter, rad2deg.(xp[i, :]), color = :grey, label = nothing, linewidth = 0.1)
end
plot!(p, 1:nIter, rad2deg.(x), label = "True state", color = :blue, linewidth = 1);
plot!(p, 1:nIter, rad2deg.(z), label = "Measurement", color = :red, linewidth = 1);
plot!(
    p,
    1:nIter,
    rad2deg.(xHat),
    label = "PF estimate",
    color = :black,
    linewidth = 1,
    linestyle = :dash
);
xlabel!(p, "\$k\$");
ylabel!(p, "\$\\theta\$ (deg)");
p

p2 = plot(dpi = 1600, size(800, 600), framestyle = :box);
plot!(p2, 1:nIter, [1 / sum(wp[:, i] .^ 2) for i in 1:nIter]);
xlabel!(p2, "\$k\$");
ylabel!(p2, "\$N_{eff}\$");
p2