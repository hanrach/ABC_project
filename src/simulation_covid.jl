using LinearAlgebra
using DifferentialEquations
using Distributions
using Random
using Printf
using Plots
using RCall
using CPUTime
using StatsBase
using JLD
# using StatsPlots
include("utils.jl")
include("sir_ode.jl")
include("abc.jl")
include("abcMCMC.jl")
include("abcSMC.jl")
include("BayesianCalibration.jl")

# Bayesian Calibration
algo_list = (ABC=ABC, ABC_MCMC=ABC_MCMC, ABC_SMC=ABC_SMC)

true_prior = (Truncated(Normal(0.1,0.5),0,Inf),Truncated(Normal(0.2,0.2),0,Inf))

algo_parameter_ABC = (prior = true_prior,
                     epsilon = 0.1,eta = identity_mapping, d = compute_norm)

algo_parameter_mcmc = (prior = true_prior,
                        epsilon = 1, eta = identity_mapping,
                        d= compute_norm, proposal = proposal_Normal,
                        proposalRatio = proposalRatio_Normal,sd = (0.25,0.25),
                        thinning = 100, verbose=false,
                        burn_in = 10000)


algo_parameters_smc = (prior = true_prior,
                        time_final=5, eps_list =[10, 5, 1, 0.5, 0.1],
                        eta= identity_mapping, d= distanceFunction,
                        kernel=proposal_Normal,
                        kernel_density=proposal_Normal_density,
                        sd=(0.25,0.25), resample_method=systematic_resample,
                        verbose=false)

algo_param_list = (ABC = algo_parameter_ABC,
                    ABC_MCMC = algo_parameter_mcmc,
                    ABC_SMC = algo_parameters_smc)

simulation = BayesianCalibration(10,Int(100),0.10,algo_list,algo_param_list,
                                ode_model = solve_ode,
                                initial_state = [100.0;0.0001;0.001],
                                time_window=(0,25.0),
                                add_noise = false,
                                true_p_dist=true_prior)

using JLD
save("results/calibration_1000_250_25.jld","result",look)

look = load("results/calibration_1000_250_25.jld")["result"]
# calibration
@printf("Calibration: ABC %f, ABC MCMC %f, ABC SMC %f\n", mean(look[2],dims=1)[1], mean(look[2],dims=1)[2],mean(look[2],dims=1)[3])
@printf("CPU_time: ABC %f, ABC MCMC %f, ABC SMC %f\n", mean(look[4],dims=1)[1], mean(look[4],dims=1)[2],mean(look[4],dims=1)[3])
@printf("ESS: ABC %f, ABC MCMC %f, ABC SMC %f\n", mean(look[5],dims=1)[1], mean(look[5],dims=1)[2],mean(look[5],dims=1)[3])
@printf("ESS Time: ABC %f, ABC MCMC %f, ABC SMC %f\n", mean(look[6],dims=1)[1], mean(look[6],dims=1)[2],mean(look[6],dims=1)[3])

@show(map(x->summarystats(look[4][:,x]),[1,2,3]))
