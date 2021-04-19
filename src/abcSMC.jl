# y: data
# yhat_generator: function that samples data given the parameters
# algo parameters: Namedtuple of algorithm parameters
# max_time: max time allowed for the algo
# N_samples: number of samples desired
function ABC_SMC(y,yhat_generator,algo_parameters,max_time,N_samples)

    kernel = algo_parameters[:kernel]
    kernel_density = algo_parameters[:kernel_density]
    sd = algo_parameters[:sd]
    eta_y = algo_parameters[:eta](y)
    size_y = size(y)
    q = length(algo_parameters[:prior])
    eps_list = algo_parameters[:eps_list]
    time_final = algo_parameters[:time_final]

    result = zeros(N_samples,q)
    weights = ones(N_samples)./N_samples
    theta = zeros(N_samples,q)
    t = 1
    # sample from the prior initially (same as rejection ABC)
    nParticle=1
    while nParticle < N_samples
        p = rand.(algo_parameters[:prior])
        yhat = yhat_generator(p)
        if algo_parameters[:d](eta_y,algo_parameters[:eta](yhat)) < eps_list[t]
            result[nParticle,:].=p
            nParticle +=1
        end
    end

    result_prev = result

    while (t < time_final)
        nParticle=1
        while nParticle <= N_samples
            # sample the index with with weights
            sampled_index = wsample(1:N_samples,weights,1)
            p_star = Tuple(result[sampled_index,:])
            # propose candidate parameter from the kernel
            p_cand = kernel(p_star, sd)
            # if the density  of the prior at proposed param is 0, start again
            prior_cand = prod(map( (x,y)->pdf(x,y), algo_parameters[:prior], p_cand ))
            if prior_cand == 0
                continue
            end
            yhat = yhat_generator(p_cand)
            if (algo_parameters[:d](eta_y, algo_parameters[:eta](yhat)) < eps_list[t])
                # accept the proposed parameter
                result[nParticle,:] .= p_cand
                # Compute the weight of the current particle
                weight_denom = 0
                for j=1:N_samples
                    weight_denom += weights[j]*prod(map((x,y,z) -> kernel_density(x,y,z), Tuple(result_prev[j,:]), p_cand, sd ))
                end
                weights[nParticle] = prior_cand/weight_denom
                # Normalize weights
                weights = weights./sum(weights)
                nParticle+=1
            end
        end
        
        result_prev = result
        t += 1
    end
   
    return result
end
