using LinearAlgebra
using Distributions
using Filters
using Plots
using Random

Random.seed!(1)

μx0 = deg2rad(50)
Σx0 = deg2rad(10)
Σw = deg2rad(10)
Σv = deg2rad(5)

x0, y0 = 500, 500
r = 200
ω = deg2rad(2)

nIter = 300

x = zeros(nIter + 1, 1)
z = zeros(nIter, 1)
w = randn(nIter, 1) * Σw
v = randn(nIter, 1) * Σv

for k in 1:nIter
    x[k+1] = x[k] + ω + w[k]
    z[k] = atan(r * sin(x[k]) + y0, r * cos(x[k]) + x0) + v[k]
end
x = x[1:nIter]

# Particle filter 
Np = 1000

# Model
p0 = rand(Normal(μx0, Σx0), Np, 1)
w0 = fill(1.0 / Np, Np)
state_transition = (x; P=Σw, b=ω) -> x .+ b .+ randn() * P

likelihood = (x, z; P=Σv) -> begin
    z_pred = atan((r * sin(x[1]) + y0), (r * cos(x[1]) + x0))
    return pdf(Normal(z_pred, P), z)[1]
end

# Build resampling 
policy = EffectiveSamplePolicy(750)
# algo = NoResamplingAlgorithm()
algo = SystematicResamplingAlgorithm()
# algo = MultinomialResamplingAlgorithm( zeros(Np, 1) )
resampling = Resampling(algo, policy)
filter = ParticleFilter(p0, w0, resampling, state_transition, likelihood, Np)

# Cache 
xp = zeros(Np, nIter)
wp = zeros(Np, nIter)
xHat = zeros(nIter, 1)
xp[:, 1] .= p0
wp[:, 1] .= w0
xHat[1] = sum(filter.particles[:, 1] .* filter.weights)

for k in 2:nIter
    predict!(filter)
    update!(filter, z[k, :])
    resample!(filter)
    xHat[k] = sum(filter.particles[:, 1] .* filter.weights)
    xp[:, k] .= filter.particles[:, 1]
    wp[:, k] .= filter.weights
end

p = plot(dpi=1600, size=(800,600), framestyle = :box);
for i in 1:Np
    plot!(p, 1:nIter, rad2deg.(xp[i, :]), color=:grey, label=nothing, linewidth=0.1);
end
plot!(p, 1:nIter, rad2deg.(x), label="True state", color=:blue, linewidth=1);
plot!(p, 1:nIter, rad2deg.(z), label="Measurement", color=:red, linewidth=1);
plot!(p, 1:nIter, rad2deg.(xHat), label="PF estimate", color=:black, linewidth=1, linestyle=:dash);
xlabel!(p, "\$k\$");
ylabel!(p, "\$\\theta\$ (deg)");
p

p2 = plot(dpi=1600, size(800, 600), framestyle=:box);
plot!(p2, 1:nIter, [1/sum(wp[:, i].^2) for i in 1:nIter]);
xlabel!(p2, "\$k\$");
ylabel!(p2, "\$N_{eff}\$");
p2