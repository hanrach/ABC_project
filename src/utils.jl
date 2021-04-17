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

function random_walk(p_prev, sd)
    # return proposed p, log(q(p_n+1|p_n))
    
    p_propose = map( (x,y) -> x + rand((Normal(0,y))) ,p_prev,sd)
    
    log_pdf = map( (x,y,z) -> logpdf((Normal(x,z)), y ), p_prev, p_propose, sd )
    return p_propose, log_pdf
end