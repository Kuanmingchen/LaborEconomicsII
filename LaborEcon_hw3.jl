#--------------------------------------------------
# Code for Labor Economics Homework 3 
# Version: v"1.10.4"
#--------------------------------------------------

# Load Packages
using CairoMakie, Distributions, DataFrames
using Random, Interpolations, NLsolve, Roots
using Statistics, BenchmarkTools, Parameters, Optim
using GLM

# For reproducibility
rng = Xoshiro(9527)
# Define Parameters
function McCall(;ρ = 0.9, c̲ = 0.1, α = 2.0, β = 5.0, rng = rng)
    return (; ρ, c̲, α, β, ϵ = rand(rng, Beta(α, β), 3000), w_grids = collect(0.0:0.01:1.0))
end

McCall_model = McCall()

# Define the Bellman Operator 
function T(v; model = McCall_model)
    @unpack ρ, c̲, α, β, ϵ, w_grids = model
    Tv = similar(v)
    v_func = LinearInterpolation(w_grids, v)
    Tv = max.(w_grids/(1-ρ), c̲ + ρ * mean(v_func.(ϵ)))
    return Tv
end

function VFI(v; model = McCall_model, tol = 1e-10, max_iter = 1000)
    @unpack ρ, c̲, α, β, ϵ, w_grids = model
    res = fixedpoint(v -> T(v), v; iterations = max_iter, xtol = tol)
    v_func = LinearInterpolation(w_grids, res.zero)
    v_accept = w_grids/(1-ρ)
    v_reject = (c̲ + ρ * mean(v_func.(ϵ)))*ones(length(w_grids))

    return (;v = res.zero, policy = v_accept .> v_reject, iter = res.iterations)
end

# Solve the model using value function iteration
sol = VFI(zeros(length(McCall_model.w_grids)))

function compute_res_wage(sol;model = McCall_model)
    @unpack ρ, c̲, α, β, ϵ, w_grids = model 
    v_func = LinearInterpolation(w_grids, sol.v)
    w̄ = (c̲ + ρ * mean(v_func.(ϵ))) * (1-ρ)
    return w̄
end

function simulate_McCall(w̄; model = McCall_model, n = 1000)
    @unpack ρ, c̲, α, β, ϵ, w_grids = model
    w_star = rand(Beta(α, β), n)
    s = w_star .> w̄ 
    w = s .* w_star 
    return DataFrame(w = w, s = s)
end

Random.seed!(2024)
df_1 = simulate_McCall(compute_res_wage(sol))

# Naive Mean approach
mean_naive = mean(filter(!iszero, df_1.w)) #0.5467
real_mean_wage = McCall_model.α/(McCall_model.α + McCall_model.β) #0.2857

# MLE 
# θ = (log(α), log(β))
# transformation to make sure the domain satisfies the requirement. 
# α, β ∈ (0, ∞)
# min w is a consistent estimator of w̄ 
function log_likelihood(θ; df = df_1)
    min_w = minimum(filter(!iszero, df.w))
    α, β = exp(θ[1]), exp(θ[2])
    return sum(df.s .* logpdf.(truncated(Beta(α, β); lower = min_w), df.w))
end

res = optimize(θ -> -log_likelihood(θ), zeros(2))
begin
    fig1 = Figure()
    ax1 = Axis(fig1[1,1], title = "Wage Distribution", xlabel = "w")
    lines!(ax1, 0.0:0.001:1.0, pdf(Beta(exp.(res.minimizer)...), 0.0:0.001:1.0), color = :blue, label = "MLE")
    lines!(ax1, 0.0:0.001:1.0, pdf(Beta(McCall_model.α, McCall_model.β), 0.0:0.001:1.0), color = :red, label = "True")
    Legend(fig1[1,2], ax1)
    fig1
end

# Better estimator for uncondotional mean wage
exp(res.minimizer[1])/(sum(exp.(res.minimizer)))

# Estimate ρ
function ρ_moment(θ; df = df_1, α = exp(res.minimizer[1]), β = exp(res.minimizer[2]))
    ρ = (atan(θ[1]) + π/2)/π
    min_w = minimum(filter(!iszero, df.w))
    McCall_model = McCall(α = α, β = β, ρ = ρ)
    sol = VFI(zeros(length(McCall_model.w_grids)), model = McCall_model)
    w̄ = compute_res_wage(sol, model = McCall_model)
    return abs2(min_w - w̄)
end

res_ρ = optimize(ρ -> ρ_moment(ρ), [1.0])

# ρ estimate
(atan(res_ρ.minimizer[1]) + π/2)/π

# Simulation of Crusoe's Problem
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
    res = fixedpoint(policy -> K_EGM(policy; OG = OG), policy; iterations = max_iter, xtol = tol, m = 1)
    return res.zero
end

sol_egm = EGM(0.5*OG.y_grids)

function simulate_OG(sol; OG = OG, T = 200)
    @unpack β, α, γ, μ, σ, y_grids, z, u, f = OG 
    k = ones(T+1)
    c = zeros(T)
    y = zeros(T)
    policy_func = LinearInterpolation(y_grids, sol, extrapolation_bc = Line())
    for t in 1:T
        y[t] = exp(μ + σ * randn())*f(k[t])
        c[t] = policy_func(y[t])
        k[t+1] = y[t] - c[t]
    end
    return DataFrame(k = k[1:T], c = c, y = y)
end

Random.seed!(2024)
df_2 = simulate_OG(sol_egm)

# Estimate the production Parameters
# log(y) = μ + α * log(k) + ϵ
reg = lm(@formula(log(y) ~ log(k)), df_2)
μ, α = coef(reg)
ϵ = residuals(reg)
σ = std(ϵ)

# Estimate β and γ
function βγ_moment(θ; df = df_2, μ = μ, α = α, σ = σ)
    β, γ = (atan(θ[1]) + π/2)/π, exp(θ[2])
    Random.seed!(2024)
    OG_sim = OptimalGrowth(β = β, γ = γ, μ = μ, σ = σ, α = α)
    sol = EGM(0.5*OG.y_grids; OG = OG_sim)
    policy_func = LinearInterpolation(OG.y_grids, sol, extrapolation_bc = Line())
    simulated_c = policy_func.(df.y)
    return sum((df.c - simulated_c).^2)
end

res_βγ = optimize(θ -> βγ_moment(θ), zeros(2))

β, γ = (atan(res_βγ.minimizer[1]) + π/2)/π, exp(res_βγ.minimizer[2])

Random.seed!(1234)
OG = OptimalGrowth(β = β, γ = γ, μ = μ, σ = σ, α = α)

sol = EGM(0.5*OG.y_grids; OG = OG)

df_2[!,:c_hat] = LinearInterpolation(OG.y_grids, sol, extrapolation_bc = Line()).(df_2.y)

# Compare the estimated consumption
begin
    fig3 = Figure()
    ax3 = Axis(fig3[1,1], title = "Simulation of Crusoe's Problem", xlabel = "t")
    lines!(ax3, 1:200, df_2.c, color = :blue, label = "c")
    lines!(ax3, 1:200, df_2.c_hat, color = :red, label = "ĉ")
    lines!(ax3, 1:200, df_2.y, color = :green, label = "y")
    lines!(ax3, 1:200, df_2.k, color = :orange, label = "k")
    Legend(fig3[1,2], ax3)
    fig3
end

# Compare the policy function
begin
    fig4 = Figure()
    ax4 = Axis(fig4[1,1], title = "Policy Function", xlabel = "y")
    lines!(ax4, OG.y_grids, sol, color = :blue, label = "ĉ(y)")
    lines!(ax4, OG.y_grids, sol_egm, color = :green, label = "c*(y)")
    Legend(fig4[1,2], ax4)
    fig4
end