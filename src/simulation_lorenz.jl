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
include("BayesianCalibration.jl")
include("abcSMC.jl")

# test ABC
function data_generator_lorenz(p, u0)
    time_window=(0,100.0)
    solve_lorenz(u0,time_window,p)
end

# sigma, rho, beta
p=(10., 28., 3)
solution1 = data_generator_lorenz(p, [1., 1., 1.])
solution2 = data_generator_lorenz(p, [1.1, 1.1,1.1])

plot(solution1, vars=(1,2,3))
plot!(solution2, vars=(1,2,3))

algo_parameters = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 25,
eta= identity_mapping, d= distanceFunction)

true_p_dist=(Gamma(2,1),Gamma(1,1))
true_p = map(x -> rand(x),true_p_dist)
y = data_generator(true_p)
output,t_end=ABC(y, data_generator, algo_parameters, 20)

using StatsPlots
density(output[:,1], xlabel="Paramter value β")
density(output[:,2], xlabel="Paramter value γ")

# test ABC MCMC
algo_parameter_mcmc = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 25, eta = identity_mapping,
                        d= compute_norm, proposal = proposal_Normal,
                        proposalRatio = proposalRatio_Normal,sd = (0.25,0.25),
                        thinning = 100, burn_in = 100,verbose=true)

true_p_dist=(Gamma(2,1),Gamma(1,1))
true_p = map(x -> rand(x),true_p_dist)
y = data_generator(true_p)

output, t_end = ABC_MCMC(y, data_generator, algo_parameter_mcmc,100)


# test ABC_SMC

algo_parameters_smc = (prior = (Gamma(2,1),Gamma(1,1)), time_final=5, eps_list = [30, 25, 20, 15, 10],
eta= identity_mapping, d= distanceFunction, kernel=proposal_Normal, 
kernel_density=proposal_Normal_density,
sd=(0.5,0.5), resample_method=systematic_resample)

output = ABC_SMC(y, data_generator, algo_parameters_smc, 100)


# Bayesian Calibration
algo_list = (ABC=ABC, ABC_MCMC=ABC_MCMC, ABC_SMC=ABC_SMC)
algo_parameter_ABC = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 10,eta = identity_mapping, d = compute_norm)
algo_parameter_mcmc = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 10, eta = identity_mapping,
                        d= compute_norm, proposal = proposal_Normal,
                        proposalRatio = proposalRatio_Normal,sd = (0.25,0.25),
                        thinning = 100, verbose=true,
                        burn_in = 100)

algo_parameters_smc = (prior = (Gamma(2,1),Gamma(1,1)), time_final=5, eps_list = [30, 25, 20, 15, 10],
                        eta= identity_mapping, d= distanceFunction, kernel=proposal_Normal, 
                        kernel_density=proposal_Normal_density,
                        sd=(0.25,0.25))
algo_param_list = (ABC = algo_parameter_ABC,
                   ABC_MCMC = algo_parameter_mcmc,
                   ABC_SMC = algo_parameters_smc)

look = BayesianCalibration(100,Int(500),0.10,algo_list,algo_param_list,time_window=(0.0,10.0))

# calibration
alpha_level = 0.5
print(mean(look[2],dims=1))
index = 4
print(look[2][index,:])
histogram(look[3][index,:,1,1],alpha=alpha_level)
histogram!(look[3][index,:,1,2],alpha=alpha_level)
vline!([look[1][1][index]])
# histogram!(rand(Gamma(2,1),500),alpha=alpha_level)
histogram(look[3][index,:,2,1],alpha=alpha_level)
histogram!(look[3][index,:,2,2],alpha=alpha_level)
vline!([look[1][2][index]])
# histogram!(rand(Gamma(1,1),500),alpha=alpha_level)


density(look[3][index,:,1,1],alpha=alpha_level, label="ABC")
density!(look[3][index,:,1,2],alpha=alpha_level, label="ABC-MCMC")
density!(look[3][index,:,1,3],alpha=alpha_level, label="ABC-SMC")
density!(rand(Gamma(2,1),500),alpha=alpha_level, label="Gamma")
vline!([look[1][1][index]],label="True value")
