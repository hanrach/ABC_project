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
y = y + rand(LogNormal(0,0.5),size(y))

output=ABC_MCMC(y, data_generator, algo_parameters, 0, 100)

# Bayesian Calibration
algo_list = (ABC=ABC,ABC_MCMC=ABC_MCMC)
algo_parameter_ABC = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 10,eta = identity_mapping, d = compute_norm)
algo_parameter_mcmc = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 10, eta = identity_mapping, d= compute_norm, kernel = random_walk, sd = (0.5,0.5))
algo_param_list = (ABC = algo_parameter_ABC,
                   ABC_MCMC = algo_parameter_mcmc)

look = BayesianCalibration(10,0.0,Int(1000),0.10,algo_list,algo_param_list)
