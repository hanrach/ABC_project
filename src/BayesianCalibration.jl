using DifferentialEquations
using Distributions
using LinearAlgebra
using Random
include("sir_ode.jl")
include("abc.jl")
include("abcMCMC.jl")
include("utils.jl")

# TODO: instead of one algo, accept a list of algos and a list of algo parameters
# TODO: how to compute the posterior distribution of parameters in ABC
function BayesianCalibration(N_experiments,max_time,N_samples,alpha,
    algo,algo_parameters;
    ode_model = solve_ode,
    initial_state = [99.0;1.0;0.0],
    time_window=(0,10.0),
    add_noise = false,
    true_p_dist=(Gamma(2,1),Gamma(1,1)))

    # Calibration value
    res = zeros(N_experiments)

    # data generator
    function data_generator(p)
        solve_ode(initial_state,time_window,p)
    end

    for i in 1:N_experiments
        # generate data
        # firt sample true SIR parameters
        true_p = map(x -> rand(x),true_p_dist)
        # generate data
        y = data_generator(true_p)
        if add_noise
            y = y + rand(LogNormal(0,0.5),size(y))
        end

        # perform algo
        output = algo(y,data_generator,algo_parameters,max_time,N_samples)

        # check whether parameters are contained in the 95% posterior dist
        # given independence both have to be contained
        res[i] = all(map(ind -> check_in_interval(true_p[ind],compute_posterior(vec(output[:,ind]),alpha)),1:length(true_p)))

    end

    res

end

algo_parameters = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 10,eta = identity_mapping, d = compute_norm)

look = BayesianCalibration(100,0.0,Int(10), 0.05, ABC,algo_parameters)

algo_parameters_mcmc = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 25,
eta= identity_mapping, d= distanceFunction, kernel = random_walk)

abc_mcmc_calibration = BayesianCalibration(100, 0.0, Int(20),0.05,ABC_MCMC, algo_parameters_mcmc )
