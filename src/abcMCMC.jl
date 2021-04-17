using DifferentialEquations
using Distributions
using Random
using Printf
include("sir_ode.jl")
include("utils.jl")
# y: data
# yhat_generator: function that samples data given the parameters
# algo parameters: Namedtuple of algorithm parameters
# max_time: max time allowed for the algo
# N_samples: number of samples desired
function ABC_MCMC(y,yhat_generator,algo_parameters,max_time,N_samples)

    
    kernel = algo_parameters[:kernel]
    sd = (0.1,0.1)
    eta_y = algo_parameters[:eta](y)
    q = length(algo_parameters[:prior])
    result = zeros(N_samples,q)
    dist = Inf
    p_prev=(0,0)
    thinning_interval = 10

    while (dist > algo_parameters[:epsilon])
        # sample parameters from prior
        p_prev = map(x -> rand(x),algo_parameters[:prior])
        yhat = yhat_generator(p_prev)
        dist = algo_parameters[:d](eta_y, algo_parameters[:eta](yhat))
    end
    @show p_prev
    log_p_prev = map(x->log(x), p_prev)
    naccept=0
    k = 1
    for i=1:(N_samples*thinning_interval)
        # propose candidate parameters in log space
        log_p_cand = kernel(log_p_prev, sd)[1]
        # p_cand = kernel(p_prev, sd)[1]
        #generate data
        yhat = yhat_generator( map(x->exp(x),log_p_cand))
        # yhat = yhat_generator( p_cand)
        dist = algo_parameters[:d](eta_y, algo_parameters[:eta](yhat))
        
        u = rand(Uniform(0,1))
        
        kernel_cand_prev = sum(kernel(log_p_prev, sd )[2])
        kernel_prev_cand = sum(kernel(log_p_cand, sd)[2])
        # kernel_cand_prev = sum(kernel(p_prev, sd )[2])
        # kernel_prev_cand = sum(kernel(p_cand, sd)[2])
        prior_cand = sum(map((x,y)->logpdf(x,y), algo_parameters[:prior], map(x-> exp(x),log_p_cand) ))
        prior_prev = sum(map((x,y)->logpdf(x,y), algo_parameters[:prior], map(x-> exp(x),log_p_cand) ))
        # prior_cand = sum(map((x,y)->logpdf(x,y), algo_parameters[:prior], p_cand))
        # prior_prev = sum(map((x,y)->logpdf(x,y), algo_parameters[:prior], p_prev))
        alpha = prior_cand + kernel_cand_prev - (prior_prev + kernel_prev_cand)
        # @printf("log u = %f, alpha = %f \n", log(u), alpha)
        # @printf("dist = %f\n", dist)
        
        if (i % thinning_interval==0)
            if (log(u) < alpha && dist < algo_parameters[:epsilon])
                for j in 1:q
                    result[k,j] = exp(log_p_cand[j])
                    log_p_prev = log_p_cand
                    # result[i,j] = (p_cand[j])
                    # log_p_prev = p_cand
                end
                naccept +=1
            else
                for j in 1:q
                    result[k,j] = exp(log_p_prev[j])
                    # result[i,j] = (p_prev[j])
                end
            end
            k +=1
        end
    end
    @printf("acceptance rate=%f\n", naccept/N_samples)
    return result, naccept/N_samples
end



function data_generator(p)
    initial_state = [99.0;1.0;0.0]; time_window=(0,10.0)
    solve_ode(initial_state,time_window,p)
end

algo_parameters = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 25,
eta= identity_mapping, d= distanceFunction, kernel=random_walk)

true_p_dist=(Gamma(2,1),Gamma(1,1))
true_p = map(x -> rand(x),true_p_dist)
y = data_generator(true_p)
# add noise
y = y + rand(LogNormal(0,0.5),size(y))

output, acceptance_rate=ABC_MCMC(y, data_generator, algo_parameters, 0, 100)

using StatsPlots
density(output[:,1], xlabel="Paramter value β")
density(output[:,2], xlabel="Paramter value γ")