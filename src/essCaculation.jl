# Calculate the effective sample size per unit time [given the time budget]

function ESS_calculation_time_budget(time_budget,
    N_expriments,
    alpha,
    algo_list,algo_parameter_list;
    ode_model = solve_ode,
    initial_state = [99.0;1.0;0.0],
    time_window=(0,10.0),
    add_noise = false,
    true_p_dist=(Gamma(2,1),Gamma(1,1)))




end
