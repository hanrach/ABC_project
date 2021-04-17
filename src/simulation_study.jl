using LinearAlgebra, DifferentialEquations, Distributions,Random
using StatsPlots
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
# add noise
y = y + rand(LogNormal(0,0.5),size(y))

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
algo_parameters = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 10,eta = identity_mapping, d = compute_norm)

look = BayesianCalibration(100,0.0,Int(10), 0.05, ABC,algo_parameters)

algo_parameters_mcmc = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 25,
eta= identity_mapping, d= distanceFunction, kernel = random_walk)

abc_mcmc_calibration = BayesianCalibration(100, 0.0, Int(20),0.05,ABC_MCMC, algo_parameters_mcmc )
