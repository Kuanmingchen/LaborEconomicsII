#--------------------------------------------------
# Code for Labor Economics Homework 1
# Version: v"1.10.4"
#--------------------------------------------------

# Load Packages
using CairoMakie
using Random, Interpolations, Distributions
using Parameters, NLsolve

# For reproducibility
rng = Xoshiro(9527)
# Define Parameters
function McCall(;ρ = 0.9, c̲ = 0.1, α = 2.0, β = 5.0, rng = rng)
    return (; ρ, c̲, α, β, ϵ = rand(rng, Beta(α, β), 3000), w_grids = collect(0.0:0.01:1.0))
end

model = McCall()

# Define the Bellman Operator 
function T(v; model = model)
    @unpack ρ, c̲, α, β, ϵ, w_grids = model
    Tv = similar(v)
    v_func = LinearInterpolation(w_grids, v)
    Tv = max.(w_grids/(1-ρ), c̲ + ρ * mean(v_func.(ϵ)))
    return Tv
end

function VFI(v; model = model, tol = 1e-10, max_iter = 1000)
    @unpack ρ, c̲, α, β, ϵ, w_grids = model
    res = fixedpoint(v -> T(v), v; iterations = max_iter, xtol = tol)
    v_func = LinearInterpolation(w_grids, res.zero)
    v_accept = w_grids/(1-ρ)
    v_reject = (c̲ + ρ * mean(v_func.(ϵ)))*ones(length(w_grids))

    return (;v = res.zero, policy = v_accept .> v_reject, iter = res.iterations)
end

# Solve the model using value function iteration
sol = VFI(zeros(length(model.w_grids)))

# Plot the value function
begin
    f1 = Figure()
    ax1 = Axis(f1[1, 1], xlabel = "w", ylabel = "v", title = "Value")
    lines!(ax1, model.w_grids, sol.v, color = :blue, linewidth = 2.0)
    f1
end

# Plot the policy function
begin
    f2 = Figure()
    ax2 = Axis(f2[1, 1], xlabel = "w", ylabel = "Accept", title = "Policy")
    lines!(ax2, model.w_grids, sol.policy, color = :red, linewidth = 2.0)
    f2
end

# Compute the reservation wage 
function compute_res_wage(sol;model = model)
    @unpack ρ, c̲, α, β, ϵ, w_grids = model 
    v_func = LinearInterpolation(w_grids, sol.v)
    w̄ = (c̲ + ρ * mean(v_func.(ϵ))) * (1-ρ)
    return w̄
end

w̄ = compute_res_wage(sol)

# Simulate the time to find a job 
function simulate(w̄; n = 1000, model = model)
    @unpack ρ, c̲, α, β, ϵ, w_grids = model
    s = zeros(n) 
    s_new = similar(s)
    job_finding_time = zeros(n)
    w = similar(s)
    t = 1
    while prod(s) == 0
        w = rand(Beta(α, β), n)
        s_new = (1 .- s) .* (w .> w̄)
        job_finding_time += s_new * t
        s = s .+ s_new
        t += 1
    end
    return job_finding_time
end

Random.seed!(2024)
job_finding_time = simulate(w̄)

# Plot the histogram of the job finding time
begin
    f3 = Figure()
    ax3 = Axis(f3[1, 1], xlabel = "Time", ylabel = "Frequency", title = "Job Finding Time")
    hist!(ax3, job_finding_time, bins = 20, color = :green, strokewidth = 0, normalization = :pdf)
    f3
end

# Set different c̲
c̲_vals = collect(0.1:0.1:0.6)
res_wages = zeros(length(c̲_vals))

for (i, c̲) in enumerate(c̲_vals)
    model = McCall(c̲ = c̲)
    sol = VFI(zeros(length(model.w_grids)))
    res_wages[i] = compute_res_wage(sol)
end

# Plot the reservation wage as a function of c̲
begin
    f4 = Figure()
    ax4 = Axis(f4[1, 1], xlabel = "c̲", ylabel = "w̄", title = "Reservation Wage")
    lines!(ax4, c̲_vals, res_wages, color = :purple, linewidth = 2.0)
    f4
end

# Simulate and plot the job finding time for different c̲
begin
    f5 = Figure()
    for (i, w̄) in enumerate(res_wages)
        Random.seed!(2024)
        job_finding_time = simulate(w̄)
        ax = Axis(f5[cld(i, 2), 2 - (i % 2)])
        xlims!(ax, 1, 100)
        ylims!(ax, 0, 0.15)
        hist!(ax, job_finding_time, color = :green, strokewidth = 0, normalization = :pdf)
        ax.title = "c̲ = $(c̲_vals[i])"
    end 
    f5
end

# We see that the reservation wage increases with c̲. As c̲ increases, the worker becomes 
# more patient and is willing to wait for a higher wage.

# The above is the code for the alternative approach to solve the 
# McCall model using value function iteration. 
# Reset the model
model = McCall()
function U(w̄; model = model)
    @unpack ρ, c̲, α, β, ϵ, w_grids = model
    return [(1-ρ)*c̲ + ρ*mean(max.(ϵ, w̄[1]))] 
end

function VFI_2(w̄; model = model, tol = 1e-10, max_iter = 1000)
    @unpack ρ, c̲, α, β, ϵ, w_grids = model
    res = fixedpoint(w̄ -> U(w̄), [w̄]; iterations = max_iter, xtol = tol)
    return (res_wage = res.zero, iter = res.iterations)
end

w̄ = VFI_2(0.5)

# As long as the reservation wage is obtained, the policy function
# is then clear. The worker will accept the wage offer if it is 
# higher than the reservation wage, and reject otherwise. 