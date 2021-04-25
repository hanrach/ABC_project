function check_in_interval(num,interval)
    (num >= interval[1]) & (num <= interval[2])
end

function compute_posterior(vec_num,alpha)
    quantile(vec_num,[alpha/2,1-alpha/2])
end

function distanceFunction(ytilde, y)
    return mean((ytilde[3,:]-y[3,:]) .^ 2)
end

function identity_mapping(y)
    y
end

function compute_norm(y,yhat;p=2)
    norm(y[2:3,2:end] - yhat[2:3,2:end], p)/size(y)[1]
end

function compute_full_norm(y,yhat;p=2)
    norm(y[2:3,2:end] - yhat[2:3,2:end], p)/size(y)[1]
end

function random_walk(p_prev, sd)
    # return proposed p, log(q(p_n+1|p_n))

    p_propose = map( (x,y) -> x + rand((Normal(0,y))) ,p_prev,sd)

    log_pdf = map( (x,y,z) -> logpdf((Normal(x,z)), y ), p_prev, p_propose, sd )
    return p_propose, log_pdf
end


function proposal_Normal_density(p1, p2, sd)
    return map( (x,y,z) -> pdf(Normal(x,z),y), p1, p2, sd )
end

function proposal_Normal(p_prev,sd)
    p_prev .+ rand.(Normal.(0,sd))
end

function proposalRatio_Normal(p_prev,p_cand,sd)
    1
end

function systematic_resample(weights)
    N = length(weights)
    partitions = rand()/N .+ (0:1:N-1)./N
    idx = zeros(Int,N)
    Q = cumsum(weights)
    i=1;j=1;
    while i<=N
        if partitions[i] < Q[j]
            idx[i]=j
            i +=1
        else
            j+=1
        end
    end
    return idx
end

function CPUtoc_modified(verbose)
    t = CPUTime.CPUtoq()
    if verbose
        println("elpased CPU time: ",t," seconds")
    end
    return t
end
