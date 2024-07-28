#--------------------------------------------------
# Code for Labor Economics Homework 2 
# Version: v"1.10.4"
#--------------------------------------------------

# Load Packages
using CairoMakie
using Random, Interpolations, NLsolve
using Statistics, BenchmarkTools, Parameters, Optim


# For reproducibility
Random.seed!(1234)
# Define Parameters
function OptimalGrowth(;β = 0.96, α = 0.8, γ = 1.0, μ = 0.0, σ = 0.3, 
                        y_grids = collect(range(1e-10, 10.0, length = 100)), 
                        z = exp.(μ .+ σ .* randn(300)))
    u(c) = γ == 1.0 ? log(c) : c^(1-γ)/(1-γ)
    f(k) = k ≥ 0 ? k^α : 0.0
    return (;β, α, γ, μ, σ, y_grids, z, u, f) 
end
OG = OptimalGrowth()

# Define the Bellman Operator
function T(v;OG = OG, tol = 1e-10)
    @unpack β, α, γ, μ, σ, y_grids, z, u, f = OG
    v_new = similar(v)
    policy = similar(v)
    v_func = LinearInterpolation(y_grids, v, extrapolation_bc = Line())
    for (i, y) in enumerate(y_grids)
        objective(c) = u(c) + β * mean(v_func.(z .* f(y - c)))
        res = maximize(objective, tol, y)
        v_new[i] = Optim.maximum(res)
        policy[i] = Optim.maximizer(res)
    end
    return (; v = v_new, policy = policy)
end

# Plot the first 50 iterations 
# The hotter color means the later iteration
begin 
    n = 50
    v_0 = zeros(length(OG.y_grids))
    v_1 = similar(v_0)
    fig = Figure()
    ax = Axis(fig[1, 1])
    for i in 1:n
        v_1 = T(v_0).v
        lines!(ax, OG.y_grids, v_1, color = RGBAf(0.0 + i/n, 0.2, 0.5), linewidth = 2.0)
        v_0 .= v_1
    end
    fig
end

# Solve the model using value function iteration 
function VFI(v; OG = OG, tol = 1e-10, max_iter = 1000)
    res = fixedpoint(v -> T(v).v, v; iterations = max_iter, xtol = tol)
    return (;v = res.zero, policy = T(res.zero).policy, iter = res.iterations)
end

sol = VFI(zeros(length(OG.y_grids)))

# Plot the policy function 
begin
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, OG.y_grids, sol.policy, color = :blue, linewidth = 2.0)
    lines!(ax, OG.y_grids, OG.y_grids*(1-OG.β*OG.α), color = :red, linewidth = 2.0)
    fig
end

# Set σ = 0.1, 0.15, ..., 0.6
σ_vals = 0.1:0.05:0.3
begin 
    fig = Figure()
    ax = Axis(fig[1, 1])
    for (i, σ) in enumerate(σ_vals)
        Random.seed!(9527)
        OG = OptimalGrowth(;σ = σ, γ = 1.5)
        sol = VFI(zeros(length(OG.y_grids)))
        lines!(ax, OG.y_grids, sol.policy, color = RGBAf(0.0 + i/length(σ_vals), 0.2, 0.5), linewidth = 2.0)
    end
    fig
end
