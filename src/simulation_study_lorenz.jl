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


# sigma, rho, beta, x0

sigma_prior = Truncated(Normal(10., 5), 0, Inf)
rho_prior = Truncated(Normal(24, 6), 0, Inf)
beta_prior = Truncated(Normal(8/3., 1), 0, Inf)
x0_prior = Truncated(Normal(10., 5), 0, Inf)
prior_p = map(x -> rand(x),(sigma_prior, rho_prior, beta_prior, x0_prior))

sd = 0.01
sigma_true = Truncated(Normal(10., sd), 0, Inf)
rho_true = Truncated(Normal(26., sd), 0, Inf)
beta_true = Truncated(Normal(8/3., sd), 0, Inf)
x0_true = Truncated(Normal(10., sd), 0, Inf)

function data_generator_lorenz(p)
    sigma, rho, beta, x0 = p
    u0 = [x0, 10., 10.]
    time_window = (0., 25.)
    param = (sigma, rho, beta)
    y = solve_lorenz(u0,time_window,param)
end
true_prior=(sigma_true, rho_true, beta_true, x0_true)
true_p = map(x -> rand(x),true_prior)
y = data_generator_lorenz(true_p)

y2 = data_generator_lorenz((10., 26., 8/3, 10.))
#abc
algo_parameter_ABC = (prior = ( sigma_prior, rho_prior, beta_prior, x0_prior),epsilon = 13,
eta= identity_mapping, d= compute_full_norm)

#  ABC MCMC
algo_parameter_mcmc = (prior = ( sigma_prior, rho_prior, beta_prior, x0_prior) ,epsilon = 13, eta = identity_mapping,
                        d= compute_full_norm, proposal = proposal_Normal,
                        proposalRatio = proposalRatio_Normal,sd = (0.5,0.5, 0.5, 0.5),
                        thinning = 10, burn_in = 100,verbose=true)


#smc
algo_parameter_smc = (prior = ( sigma_prior, rho_prior, beta_prior, x0_prior), time_final=5, eps_list = [30, 20, 25, 20, 13],
eta= identity_mapping, d= compute_full_norm, kernel=proposal_Normal, 
kernel_density=proposal_Normal_density,
sd=(0.5,0.5, 0.5, 0.5), resample_method=systematic_resample, verbose=true)

output_abc = ABC(y, data_generator_lorenz, algo_parameter_ABC, 50)
output_smc = ABC_SMC(y, data_generator_lorenz, algo_parameter_smc, 50)
output_mcmc = ABC_MCMC(y, data_generator_lorenz, algo_parameter_mcmc, 50)

n= size(output_abc[1])[1]
p_abc=plot(y, vars=(1,3),w=1,legend=false,title="abc")
for k in 1:2:n
    p_pred = (output_abc[1][k,1], output_abc[1][k,2], output_abc[1][k,3], output_abc[1][k,4])
    y_fitted = data_generator_lorenz(p_pred)
    plot!(p_abc,y_fitted, vars=(1,3),alpha=0.3, color="#BBBBBB", legend=false, xlim=(-20, 20),xlabel="x", ylabel="z")
end
p_abc


n= size(output_abc[1])[1]
p_mcmc=plot(y, vars=(1,3),w=1,legend=false, title="mcmc")
for k in 1:2:n
    p_pred = (output_mcmc[1][k,1], output_mcmc[1][k,2], output_mcmc[1][k,3], output_mcmc[1][k,4])
    y_fitted = data_generator_lorenz(p_pred)
    plot!(p_mcmc,y_fitted, vars=(1,3),alpha=0.3, color="#BBBBBB", legend=false,xlim=(-20, 20), xlabel="x", ylabel="z")
end
p_mcmc

n= size(output_smc[1])[1]
p_smc=plot(y, vars=(1,3),w=1,legend=false, title="smc")
for k in 1:2:n
    p_pred = (output_smc[1][k,1], output_smc[1][k,2], output_smc[1][k,3], output_smc[1][k,4])
    y_fitted = data_generator_lorenz(p_pred)
    plot!(p_smc,y_fitted, vars=(1,3),alpha=0.3, color="#BBBBBB", legend=false,xlim=(-20, 20),xlabel="x", ylabel="z")
end
p_smc
