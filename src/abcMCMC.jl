# y: data
# yhat_generator: function that samples data given the parameters
# algo parameters: Namedtuple of algorithm parameters
# max_time: max time allowed for the algo
# N_samples: number of samples desired
function ABC_MCMC(y,yhat_generator,algo_parameters,max_time,N_samples)


    kernel = algo_parameters[:kernel]
    sd = algo_parameters[:sd]
    eta_y = algo_parameters[:eta](y)
    size_y = size(y)
    q = length(algo_parameters[:prior])
    thinning_interval = algo_parameters[:thinning]

    result = zeros(N_samples,q)
    dist = Inf
    p_prev=(0,0)

    while (dist > algo_parameters[:epsilon])
        # sample parameters from prior
        p_prev = rand.(algo_parameters[:prior])
        yhat = yhat_generator(p_prev)
        if size_y != size(yhat)
            continue
        end
        dist = algo_parameters[:d](eta_y, algo_parameters[:eta](yhat))
    end

    log_p_prev = log.(p_prev)
    naccept=0
    k = 1

    for i in 1:(N_samples*thinning_interval)
        # propose candidate parameters in log space
        log_p_cand = kernel(log_p_prev, sd)[1]
        # p_cand = kernel(p_prev, sd)[1]
        #generate data
        yhat = yhat_generator(exp.(log_p_cand))
        # yhat = yhat_generator( p_cand)
        if size_y != size(yhat)
            continue
        end
        dist = algo_parameters[:d](eta_y, algo_parameters[:eta](yhat))

        u = rand(Uniform(0,1))

        kernel_cand_prev = sum(kernel(log_p_prev, sd )[2])
        kernel_prev_cand = sum(kernel(log_p_cand, sd)[2])
        # kernel_cand_prev = sum(kernel(p_prev, sd )[2])
        # kernel_prev_cand = sum(kernel(p_cand, sd)[2])
        prior_cand = sum(map((x,y)->logpdf(x,y), algo_parameters[:prior], exp.(log_p_cand) ))
        prior_prev = sum(map((x,y)->logpdf(x,y), algo_parameters[:prior], exp.(log_p_cand) ))
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
                naccept += 1
            else
                for j in 1:q
                    result[k,j] = exp(log_p_prev[j])
                    # result[i,j] = (p_prev[j])
                end
            end
            k += 1
        end
    end
    @printf("acceptance rate=%f\n", naccept/N_samples)
    return result
end
