using JLD, DataFrames,Dates
using LinearAlgebra, DifferentialEquations, Distributions,Random
using Printf
include("utils.jl")
include("sir_ode.jl")
include("abc.jl")
include("abcMCMC.jl")
include("BayesianCalibration.jl")

master_data = load("../data/covid_bc.jld")

df_pre = master_data["pre"]
df_post = master_data["post"]

""" Pre-vaccine analysis """
days =30
y_pre = hcat([df_pre[!,"S"],df_pre[!,"I"],df_pre[!,"R"]]...)[1:days,:]
# test ABC
function data_generator(p)
    initial_state = y_pre[1,:]; time_window=(0,Float64(days)-1)
    solve_ode(initial_state,time_window,p)
end

algo_parameters = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 1e6,
eta= identity_mapping, d= distanceFunction)

output_abc=ABC(y_pre', data_generator, algo_parameters, 0, 20)
density(output[:,1], xlabel="Paramter value β")
density(output[:,2], xlabel="Paramter value γ")

p_output = (mode(output_abc[:,1]), 1.644)
solution = solve_ode(y_pre[1,:], (0,Float64(days)), p_output )

plot(solution, yaxis=:log, label = ["s_model" "i_model" "r_model"]); 
plot!(y_pre, yaxis=:log, label=["s" "i" "r"], seriestype=:scatter)
display(gcf())