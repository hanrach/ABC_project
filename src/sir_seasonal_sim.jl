using JLD
using DataFrames
using Dates
using LinearAlgebra
using DifferentialEquations
using Distributions
using Random
using CSV
using Printf
using RCall
using CPUTime
using StatsPlots
include("utils.jl")
include("sir_ode.jl")
include("abc.jl")
include("abcMCMC.jl")
include("abcSMC.jl")

df_tot = DataFrame(CSV.File("../data/covid_19_bc_all.csv"))


 """ Post-vaccine analysis """
 tot_days = size(df_tot)[1]
 time_interval = 1
 
 y_tot = hcat([df_tot[:S],df_tot[:I],df_tot[:R]]...)[1:7:tot_days,:]
 y_tot = y_tot./1e4
 y_dim = size(y_tot)[1]

 # L: duration of immunity,  beta0: infection rate , beta1: amplitude of seasonal focring, theta: peak of pandemic, gamma: recovery rate
 function data_generator(p)
     initial_state = y_tot[1,:]; time_window=(1,Float64(tot_days)/time_interval)
     solve_ode_seasonal_forcing(initial_state,time_window,p)
 end
 
# p = (36., 0.6, 0.8, 5., 0.5)
# sol = data_generator(p)


 algo_parameters_abc = (prior = (Truncated(Normal(52,10),0,Inf), Truncated(Normal(0.1,1),0,Inf),Truncated(Normal(0.1,0.05),0,Inf), Truncated(Normal(3,1),0,Inf), Truncated(Normal(0.5,0.1),0,Inf)),
 epsilon = 15,
 eta= identity_mapping, d= distanceFunction)
 output_abc=ABC(y_tot', data_generator, algo_parameters_abc, 500)
 
 abc_solutions=zeros(3,y_dim,500)

 for i=1:500
     abc_solutions[:,:,i] = data_generator(output_abc[1][i,:])[1:3,:]
    
 end


 alpha_level=1
 L_plot = density(output_abc[1][:,1],alpha=alpha_level, label="ABC",  title="Predicted L Posterior", xlabel="L", ylabel="Density")
 beta0_plot = density(output_abc[1][:,2],alpha=alpha_level, label="ABC",  title="Predicted β0 Posterior", xlabel="β0", ylabel="Density")
 forcing_plot = density(output_abc[1][:,3],alpha=alpha_level, label="ABC",  title="Predicted forcing Posterior", xlabel="δ", ylabel="Density")
 peak_plot = density(output_abc[1][:,4],alpha=alpha_level, label="ABC",  title="Predicted peak Posterior", xlabel="θ", ylabel="Density")
 gamma_plot = density(output_abc[1][:,5],alpha=alpha_level, label="ABC",  title="Predicted gamma Posterior", xlabel="γ", ylabel="Density")

 p_predicted=(50.,0.4,0.1,3.,0.5)
 sol_hat = data_generator(p_predicted)
 plot(abc_solutions[:,:,1]'[:,2:3],label = ["s_model" "i_model" "r_model"])
 plot!(y_tot[:,2:3],label=["s" "i" "r"], seriestype=:scatter)
 # plot(solution_day[2:end], yaxis=:log, label = ["s_model" "i_model" "r_model"], title="Daily:β=0.042, γ=0.07"); 
# plot!(1:7:250, y_pre, yaxis=:log, label=["s" "i" "r"], seriestype=:scatter)


#  # ess
# ess_output = zeros(3)
# ess_output_time = copy(ess_output)
# ess_output[1]  = try rcopy(R"mean(mcmcse::ess($(output_abc[1])))") catch; 0 end
# ess_output_time[1]  = ess_output[1] / output_abc[2]
