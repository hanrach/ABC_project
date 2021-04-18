# outputs I want
# true parameter value list
#

# TODO: how to compute the posterior distribution of parameters in ABC
function BayesianCalibration(N_experiments,max_time,N_samples,alpha,
    algo_list,algo_parameter_list;
    ode_model = solve_ode,
    initial_state = [99.0;1.0;0.0],
    time_window=(0,10.0),
    add_noise = false,
    true_p_dist=(Gamma(2,1),Gamma(1,1)))

    # num of algorithms
    N_algo = length(algo_list)

    # N param
    N_param = length(true_p_dist)

    # list of param values
    true_param = rand.(true_p_dist,N_experiments)

    # Calibration value
    res = zeros(N_experiments,N_algo)

    # algo output
    param_output = zeros(N_experiments,N_samples,N_param,N_algo)

    # data generator
    function data_generator(p)
        ode_model(initial_state,time_window,p)
    end

    for i in 1:N_experiments
        # generate data
        # firt sample true SIR parameters
        true_p = (true_param[1][i],true_param[2][i])
        # generate data
        y = data_generator(true_p)
        if add_noise
            y = y + rand(LogNormal(0,0.5),size(y))
        end

        # iterate over algos
        for algo_index in 1:N_algo
            # perform algo
            output = algo_list[algo_index](y,data_generator,algo_parameter_list[algo_index],max_time,N_samples)
            # check whether parameters are contained in the 95% posterior dist
            # given independence both have to be contained
            res[i,algo_index] = all(map(ind -> check_in_interval(true_p[ind],compute_posterior(vec(output[:,ind]),alpha)),1:length(true_p)))

            # save output
            param_output[i,:,:,algo_index] = output
        end
    end

    return(true_param = true_param,
           calibration = res,
           simulated_param = param_output)

end
