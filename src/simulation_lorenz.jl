using LinearAlgebra
using DifferentialEquations
using Distributions
using Random
using Printf
using Plots
using RCall
using CPUTime
# using StatsPlots
include("utils.jl")
include("lorenz_ode.jl")
include("abc.jl")
include("abcMCMC.jl")
include("abcSMC.jl")
include("BayesianCalibration.jl")


# sigma, rho, beta, x0



sigma_prior = Truncated(Normal(10., 5), 0, Inf)
rho_prior = Truncated(Normal(25, 4), 0, Inf)
beta_prior = Truncated(Normal(8/3., 1), 0, Inf)
x0_prior = Truncated(Normal(10., 5), 0, Inf)
prior_p = map(x -> rand(x),(sigma_prior, rho_prior, beta_prior, x0_prior))

sd = 0.1
sigma_true = Truncated(Normal(10., sd), 0, 11)
rho_true = Truncated(Normal(26., sd), 0, 27)
beta_true = Truncated(Normal(8/3., sd), 2, 3)
x0_true = Truncated(Normal(10., sd), 9, 11)

algo_parameter_ABC = (prior = ( sigma_prior, rho_prior, beta_prior, x0_prior),epsilon = 13,
eta= identity_mapping, d= compute_full_norm)


true_prior=(sigma_true, rho_true, beta_true, x0_true)


algo_parameter_smc = (prior = ( sigma_prior, rho_prior, beta_prior, x0_prior), time_final=5, eps_list = [30, 20, 25, 20, 13],
eta= identity_mapping, d= compute_full_norm, kernel=proposal_Normal,
kernel_density=proposal_Normal_density,
sd=(0.5,0.5, 0.5, 0.5), resample_method=systematic_resample, verbose=false)


# test ABC MCMC
algo_parameter_mcmc = (prior = ( sigma_prior, rho_prior, beta_prior, x0_prior) ,epsilon = 13, eta = identity_mapping,
                        d= compute_full_norm, proposal = proposal_Normal,
                        proposalRatio = proposalRatio_Normal,sd = (0.5,0.5, 0.5, 0.5),
                        thinning = 10, burn_in = 100,verbose=false)

# Bayesian Calibration
algo_list = (ABC=ABC, ABC_MCMC=ABC_MCMC, ABC_SMC=ABC_SMC)

algo_param_list = (ABC = algo_parameter_ABC,
                   ABC_MCMC = algo_parameter_mcmc,
                   ABC_SMC = algo_parameter_smc)

simulation = BayesianCalibration(Int(50),Int(250),0.10,algo_list,algo_param_list,
ode_model = solve_lorenz,
initial_state = [10., 10., 10.],
time_window=(0,25.0),
add_noise = false,
true_p_dist=true_prior,
model = "lorenz",save_param_output=true)



# calibration
show_calibration(simulation)

results_dir = "results/simulation_lorenz/"

save(results_dir*"output.jld","output",simulation)

using DataFrames
simulation = load(results_dir*"output.jld")["output"]

function to_dataframe(x)
    tmp = DataFrame(x)
    rename!(tmp,[:ABC,:MCMC,:SMC])
end

CSV.write(results_dir*"calibration.csv",to_dataframe(simulation[2]))
CSV.write(results_dir*"cpu.csv",to_dataframe(simulation[4]))
CSV.write(results_dir*"ess.csv",to_dataframe(simulation[5]))
CSV.write(results_dir*"ess_cpu.csv",to_dataframe(simulation[6]))
