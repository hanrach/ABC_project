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
