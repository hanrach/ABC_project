using LinearAlgebra
using DifferentialEquations
using Distributions
using Random
using Printf
using Plots
# using StatsPlots
include("utils.jl")
include("sir_ode.jl")
include("abc.jl")
include("abcMCMC.jl")
include("BayesianCalibration.jl")

# test ABC
function data_generator(p)
    initial_state = [99.0;1.0;0.0]; time_window=(0,10.0)
    solve_ode(initial_state,time_window,p)
end

algo_parameters = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 25,
eta= identity_mapping, d= distanceFunction)

true_p_dist=(Gamma(2,1),Gamma(1,1))
true_p = map(x -> rand(x),true_p_dist)
y = data_generator(true_p)
output=ABC(y, data_generator, algo_parameters, 0, 20)

density(output[:,1], xlabel="Paramter value β")
density(output[:,2], xlabel="Paramter value γ")

# test ABC MCMC
algo_parameters = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 25,
eta= identity_mapping, d= distanceFunction, kernel=random_walk,
sd=(0.2,0.2))

true_p_dist=(Gamma(2,1),Gamma(1,1))
true_p = map(x -> rand(x),true_p_dist)
y = data_generator(true_p)
# add noise
print(size(y))

output=ABC_MCMC(y, data_generator, algo_parameters, 0, 100)

# Bayesian Calibration
algo_list = (ABC=ABC,ABC_MCMC=ABC_MCMC)
algo_parameter_ABC = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 10,eta = identity_mapping, d = compute_norm)
algo_parameter_mcmc = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 1, eta = identity_mapping,
                        d= compute_norm, kernel = random_walk, sd = (0.25,0.25),
                        thinning = 100)
algo_param_list = (ABC = algo_parameter_ABC,
                   ABC_MCMC = algo_parameter_mcmc)


look = BayesianCalibration(100,0.0,Int(500),0.10,algo_list,algo_param_list)

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

compute_posterior(look[3][index,:,1,2],0.10)
