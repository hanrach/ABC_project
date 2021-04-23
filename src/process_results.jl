using JLD
using DataFrames
using Dates
using LinearAlgebra
using Distributions
using Random
using CSV
using Printf
using CPUTime

df_pre = DataFrame(CSV.File("data/covid19_bc_pre_vaccine.csv"))
df_post = DataFrame(CSV.File("data/covid19_bc_post_vaccine.csv"))

# """ Pre-vaccine analysis """
tot_days = size(df_pre)[1]
time_interval = 7

y_pre = hcat([df_pre[:S],df_pre[:I],df_pre[:R]]...)[1:7:tot_days,:]
y_pre = y_pre./1e4
y_dim = size(y_pre)[1]

df_weekly = df_pre[1:7:tot_days,:]
scale_SIR(SIR) = SIR./1e4
transform!(df_weekly,[:S] => ByRow(scale_SIR) => :S)
transform!(df_weekly,[:I] => ByRow(scale_SIR) => :I)
transform!(df_weekly,[:R] => ByRow(scale_SIR) => :R)

CSV.write("data/covid19_bc_pre_vaccine_weekly.csv",df_pre[1:7:tot_days,:])

mcmc = load("results/abc_mcmc_solution_500.jld")["abc_mcmc_solutions"]