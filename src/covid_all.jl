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
 time_interval = 7
 
 y_tot = hcat([df_tot[:S],df_tot[:I],df_tot[:R]]...)[1:7:tot_days,:]
 y_tot = y_tot./1e4
 y_dim = size(y_tot)[1]

 function data_generator_week(p)
     initial_state = y_tot[1,:]; time_window=(1,Float64(tot_days)/time_interval)
     solve_ode(initial_state,time_window,p)
 end
 
 algo_parameters_abc = (prior = (Truncated(Normal(0.1,0.5),0,Inf),Truncated(Normal(0.2,0.2),0,Inf)),epsilon = 2,
 eta= identity_mapping, d= distanceFunction)
 output_abc=ABC(y_tot', data_generator_week, algo_parameters_abc, 500)
 
 algo_parameter_mcmc = (prior = (Truncated(Normal(0.1,0.5),0,Inf),Truncated(Normal(0.2,0.2),0,Inf)),epsilon = 5, eta = identity_mapping,
                         d= compute_norm, proposal = proposal_Normal,
                         proposalRatio = proposalRatio_Normal,sd = (0.5,0.5),
                         thinning = 10000, burn_in=100, verbose=true)
 
 output_mcmc = ABC_MCMC(y_tot', data_generator_week, algo_parameter_mcmc, 500)
 
 
 algo_parameters_smc = (prior = (Truncated(Normal(0.1,0.5),0,Inf),Truncated(Normal(0.2,0.2),0,Inf)), time_final=5, eps_list = [20, 15, 10, 5, 2],
 eta= identity_mapping, d= distanceFunction, kernel=proposal_Normal, 
 kernel_density=proposal_Normal_density,
 sd=(0.5,0.5), resample_method=systematic_resample, verbose=true)
 output_smc = ABC_SMC(y_tot', data_generator_week, algo_parameters_smc, 500)
 
 
 abc_solutions=zeros(3,y_dim,500)
 abc_mcmc_solutions=zeros(3,y_dim,500)
 abc_smc_solutions=zeros(3,y_dim,500)
 for i=1:500
     abc_solutions[:,:,i] = data_generator_week(output_abc[1][i,:])[1:3,:]
     abc_mcmc_solutions[:,:,i] = data_generator_week(output_mcmc[1][i,:])[1:3,:]
     abc_smc_solutions[:,:,i] = data_generator_week(output_smc[1][i,:])[1:3,:]
 end
 
 save("../results/abc_solution_500_tot.jld","abc_solutions",abc_solutions)
 save("../results/abc_mcmc_solution_500_tot.jld","abc_mcmc_solutions",abc_mcmc_solutions)
 save("../results/abc_smc_solution_500_tot.jld","abc_smc_solutions",abc_smc_solutions)
 
 alpha_level=1
 beta_plot = density(output_abc[1][:,1],alpha=alpha_level, label="ABC",  title="Predicted β Posterior", xlabel="β", ylabel="Density")
 density!(output_mcmc[1][:,1],alpha=alpha_level, label="ABC-MCMC")
 density!(output_smc[1][:,1],alpha=alpha_level, label="ABC-SMC")
 savefig(beta_plot,"../figs/beta_plot_tot.png")
 
 gamma_plot = density(output_abc[1][:,2],alpha=alpha_level, label="ABC",  title="Predicted γ Posterior", xlabel="γ", ylabel="Density")
 density!(output_mcmc[1][:,2],alpha=alpha_level, label="ABC-MCMC")
 density!(output_smc[1][:,2],alpha=alpha_level, label="ABC-SMC")
 savefig(gamma_plot,"../figs/gamma_plot_tot.png")
 
 # ess
ess_output = zeros(3)
ess_output_time = copy(ess_output)
ess_output[1]  = try rcopy(R"mean(mcmcse::ess($(output_abc[1])))") catch; 0 end
ess_output_time[1]  = ess_output[1] / output_abc[2]
ess_output[2]  = try rcopy(R"mean(mcmcse::ess($(output_mcmc[1])))") catch; 0 end
ess_output_time[2]  = ess_output[2] / output_mcmc[2]
ess_output[3]  = try rcopy(R"mean(mcmcse::ess($(output_smc[1])))") catch; 0 end
ess_output_time[3]  = ess_output[3] / output_smc[2]

save("../results/covid/ess_tot.jld","ess_output",ess_output)
save("../results/covid/ess_cputime_tot.jld","ess_output_time",ess_output_time)


