# y: data
# yhat_generator: function that samples data given the parameters
# algo parameters: Namedtuple of algorithm parameters
# max_time: max time allowed for the algo
# N_samples: number of samples desired
function ABC(y,yhat_generator,algo_parameters,max_time,N_samples)

    # TODO: implement max_time
    N = 1
    eta_y = algo_parameters[:eta](y)
    q = length(algo_parameters[:prior])
    result = zeros(N_samples,q)

    while N < N_samples+1
        # sample parameters
        p = map(x -> rand(x),algo_parameters[:prior])
        # generate sample
        yhat = yhat_generator(p)
        # check acceptance
        if algo_parameters[:d](eta_y,algo_parameters[:eta](yhat)) < algo_parameters[:epsilon]
            for j in 1:q
                result[N,j] = p[j]
            end
            N += 1
        end

    end

    return result
end
