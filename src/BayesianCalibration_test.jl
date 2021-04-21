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
include("sir_ode.jl")
include("abc.jl")
include("abcMCMC.jl")
include("BayesianCalibration.jl")
include("abcSMC.jl")

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
                        sd=(0.25,0.25), resample_method=systematic_resample)
algo_param_list = (ABC = algo_parameter_ABC,
                    ABC_MCMC = algo_parameter_mcmc,
                    ABC_SMC = algo_parameters_smc)

look = BayesianCalibration(10,Int(500),0.10,algo_list,algo_param_list,time_window=(0.0,10.0))

# calibration
@printf("Calibration: ABC %f, ABC MCMC %f, ABC SMC %f\n", mean(look[2],dims=1)[1], mean(look[2],dims=1)[2],mean(look[2],dims=1)[3])
@printf("CPU_time: ABC %f, ABC MCMC %f, ABC SMC %f\n", mean(look[4],dims=1)[1], mean(look[4],dims=1)[2],mean(look[4],dims=1)[3])
@printf("ESS: ABC %f, ABC MCMC %f, ABC SMC %f\n", mean(look[5],dims=1)[1], mean(look[5],dims=1)[2],mean(look[5],dims=1)[3])
@printf("ESS Time: ABC %f, ABC MCMC %f, ABC SMC %f\n", mean(look[6],dims=1)[1], mean(look[6],dims=1)[2],mean(look[6],dims=1)[3])

