# y: data
# yhat_generator: function that samples data given the parameters
# algo parameters: Namedtuple of algorithm parameters
# max_time: max time allowed for the algo
# N_samples: number of samples desired
function ABC(y,yhat_generator,algo_parameters,max_time,N_samples)

    # TODO: implement max_time
    eta_y = algo_parameters[:eta](y)
    q = length(algo_parameters[:prior])
    result = zeros(N_samples,q)
    size_y = size(y)
    i = 1

    while i <= N_samples
        # sample parameters
        p = rand.(algo_parameters[:prior])
        
        # generate sample
        yhat = yhat_generator(p)

        # in case there is no soln given the parameters
        if size_y != size(yhat)
            continue
        end

        # check acceptance
        if algo_parameters[:d](eta_y,algo_parameters[:eta](yhat)) < algo_parameters[:epsilon]
            for j in 1:q
                result[i,j] = p[j]
            end
            i += 1
        end

    end

    return result
end