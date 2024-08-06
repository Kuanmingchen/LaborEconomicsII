#--------------------------------------------------
# Code for Labor Economics Homework 2 
# Version: v"1.10.4"
#--------------------------------------------------

# Load Packages
using CairoMakie
using Random, Interpolations, NLsolve, Roots
using Statistics, BenchmarkTools, Parameters, Optim


# For reproducibility
Random.seed!(1234)
# Define Parameters
function OptimalGrowth(;β = 0.96, α = 0.8, γ = 1.0, μ = 0.0, σ = 0.3, 
                        y_grids = collect(range(1e-5, 10.0, length = 100)), 
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
    ax = Axis(fig[1, 1], title = "Convergence of VFI", xlabel = "y", ylabel = "v")
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

sol_vfi = VFI(zeros(length(OG.y_grids))).policy

# For policy function iteration
# Define T_policy
function T_policy(v; OG = OG, policy = policy)
    @unpack β, α, γ, μ, σ, y_grids, z, u, f = OG
    v_new = similar(policy)
    v_func = LinearInterpolation(y_grids, v, extrapolation_bc = Line())
    for (i, y) in enumerate(y_grids)
        v_new[i] = u(policy[i]) + β * mean(v_func.(z .* f(y - policy[i])))
    end
    return v_new
end

# Solve the model using policy function iteration 
function compute_v_policy(policy; OG = OG, tol = 1e-10, max_iter = 1000)
    @unpack β, α, γ, μ, σ, y_grids, z, u, f = OG
    res = fixedpoint(v -> T_policy(v; policy = policy), zeros(length(y_grids)); iterations = max_iter, xtol = tol) 
    return res.zero
end

function compute_v_policy_opt(policy; OG = OG, tol = 1e-10, max_iter = 1000)
    @unpack β, α, γ, μ, σ, y_grids, z, u, f = OG
    v_policy = compute_v_policy(policy)
    policy_new = similar(v_policy)
    v_func = LinearInterpolation(y_grids, v_policy, extrapolation_bc = Line())
    for (i, y) in enumerate(y_grids)
        objective(c) = u(c) + β * mean(v_func.(z .* f(y - c)))
        res = maximize(objective, tol, y)
        policy_new[i] = Optim.maximizer(res)
    end
    return policy_new
end

function PFI(policy; OG = OG, tol = 1e-10, max_iter = 1000)
    @unpack β, α, γ, μ, σ, y_grids, z, u, f = OG
    res = fixedpoint(policy -> compute_v_policy_opt(policy), policy; iterations = max_iter, xtol = tol)
    return res.zero
end

sol_pfi = PFI(0.5*OG.y_grids)

# Solve the model using the envelope condition methods
function K(policy; OG = OG)
    @unpack β, α, γ, μ, σ, y_grids, z, u, f = OG
    policy_new = similar(policy)
    policy_func = LinearInterpolation(y_grids, policy, extrapolation_bc = Line())
    # Define the derivatives. One can try autodiff here.
    Du(c) = c^(-γ)
    Df(k) = k ≥ 0 ? α * k^(α-1) : 0.0
    for (i, y) in enumerate(y_grids)
        obj(c) = Du(c) - β * mean(Du.(policy_func.(z .* f(y - c))) .* z .* Df(y - c))
        policy_new[i] = find_zero(obj, (1e-10, y - 1e-10))
    end
    return policy_new
end

function ECM(policy; OG = OG, tol = 1e-10, max_iter = 1000)
    @unpack β, α, γ, μ, σ, y_grids, z, u, f = OG
    res = fixedpoint(policy -> K(policy), policy; iterations = max_iter, xtol = tol, m = 1)
    return res.zero
end

sol_ecm = ECM(0.5*OG.y_grids)

# Solve the model using the endogenous grid method
function K_EGM(policy; OG = OG, k_grids = collect(range(1e-6, 10.0, length = 100)))
    @unpack β, α, γ, μ, σ, y_grids, z, u, f = OG
    c = similar(policy)
    policy_new = similar(policy)
    policy_func = LinearInterpolation(y_grids, policy, extrapolation_bc = Line())
    # Define the derivatives. One can try autodiff here.
    Du(c) = c^(-γ)
    Df(k) = α * k^(α-1)
    Du_inv(x) = x^(-1/γ) 
    for (i, k) in enumerate(k_grids)
        c[i] = Du_inv(β * mean(Du.(policy_func.(z .* f(k))) .* z .* Df(k)))
    end
    y = k_grids + c
    c_func = LinearInterpolation(y, c, extrapolation_bc = Line())
    policy_new = c_func.(y_grids)
    return policy_new
end

function EGM(policy; OG = OG, tol = 1e-10, max_iter = 1000)
    @unpack β, α, γ, μ, σ, y_grids, z, u, f = OG
    res = fixedpoint(policy -> K_EGM(policy), policy; iterations = max_iter, xtol = tol, m = 1)
    return res.zero
end

sol_egm = EGM(0.5*OG.y_grids)

# Compare the speed of different methods
@benchmark VFI(zeros(length(OG.y_grids))) # 363.003 ms
@benchmark PFI(0.5*OG.y_grids) # 49.664 s
@benchmark ECM(0.5*OG.y_grids) # 295.850 ms
@benchmark EGM(0.5*OG.y_grids) # 9.494 ms

# One may see that the endogenous grid method is much faster than 
# the other methods. PFI seems to be the slowest one; however, 
# this is largely due to our problem setup. Since the expectation 
# operator here cannot be represented by a matrix, we have to 
# use contraction mapping theorem to solve the problem. For 
# problems with finite state space, PFI can be much faster, which 
# we would see in future exercises.

# Compare the solutions
# Plot the policy functions 
begin
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Policy Functions", xlabel = "y", ylabel = "c")
    lines!(ax, OG.y_grids, sol_vfi, color = :blue, linewidth = 2.0, label = "VFI")
    lines!(ax, OG.y_grids, sol_pfi, color = :red, linewidth = 2.0, label = "PFI")
    lines!(ax, OG.y_grids, sol_ecm, color = :green, linewidth = 2.0, label = "ECM")
    lines!(ax, OG.y_grids, sol_egm, color = :orange, linewidth = 2.0, label = "EGM")
    lines!(ax, OG.y_grids, OG.y_grids * (1 - OG.α*OG.β), color = :purple, linewidth = 2.0, label = "Analytic", linestyle = :dash)
    Legend(fig[1, 2], ax)
    fig
end

# Set σ = 0.1, 0.15, ..., 0.3
σ_vals = 0.1:0.05:0.3
begin 
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Policy Functions with Different σ", xlabel = "y", ylabel = "c")
    for (i, σ) in enumerate(σ_vals)
        Random.seed!(9527)
        OG = OptimalGrowth(;σ = σ, γ = 1.5)
        sol = VFI(zeros(length(OG.y_grids)))
        lines!(ax, OG.y_grids, sol.policy, color = RGBAf(0.0 + i/length(σ_vals), 0.2, 0.5), linewidth = 2.0, label = "σ = $σ")
    end
    Legend(fig[1, 2], ax)
    fig
end

# As we can see, the more volatile the income is, the more precautionary 
# saving the agent would have. This is due to the fact that the agent with 
# CRRA utility has a positive third order derivative for γ > 0. Such 
# preference is called prudence. 

# Extra: Simulation of γ
γ_vals = 0.5:0.1:1.5
begin 
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Policy Functions with Different γ", xlabel = "y", ylabel = "c")
    for (i, γ) in enumerate(γ_vals)
        Random.seed!(9527)
        OG = OptimalGrowth(;γ = γ)
        sol = VFI(zeros(length(OG.y_grids)))
        lines!(ax, OG.y_grids, sol.policy, color = RGBAf(0.0 + i/length(γ_vals), 0.2, 0.5), linewidth = 2.0, label = "γ = $γ")
    end
    Legend(fig[1, 2], ax)
    fig
end

# Extra: Simulation of β
β_vals = 0.7:0.05:0.95
begin 
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Policy Functions with Different β", xlabel = "y", ylabel = "c")
    for (i, β) in enumerate(β_vals)
        Random.seed!(9527)
        OG = OptimalGrowth(;β = β)
        sol = VFI(zeros(length(OG.y_grids)))
        lines!(ax, OG.y_grids, sol.policy, color = RGBAf(0.0 + i/length(β_vals), 0.2, 0.5), linewidth = 2.0, label = "β = $β")
    end
    Legend(fig[1, 2], ax)
    fig
end