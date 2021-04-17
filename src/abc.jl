using DifferentialEquations
using Distributions
using Random

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

algo_parameters = (prior = (Gamma(2,1),Gamma(1,1)),epsilon = 25,
eta= identity_mapping, d= distanceFunction)

true_p_dist=(Gamma(2,1),Gamma(1,1))
true_p = map(x -> rand(x),true_p_dist)
y = data_generator(true_p)
# add noise
y = y + rand(LogNormal(0,0.5),size(y))

output=ABC(y, data_generator, algo_parameters, 0, 20)
using StatsPlots
density(output[:,1], xlabel="Paramter value β")
density(output[:,2], xlabel="Paramter value γ")