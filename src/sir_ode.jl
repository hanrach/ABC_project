function solve_ode(u0, tspan, p, time_interval=1)
        function SIR(du,u,p,t)
                du[1]=-p[1]*u[1]*u[2]/(u[1]+u[2]+u[3])
                du[2]=p[1]*u[1]*u[2]/(u[1]+u[2]+u[3]) - p[2]*u[2]
                du[3] = p[2]*u[2]
        end
        problem = ODEProblem(SIR, u0, tspan,p)
        sol=solve(problem, saveat=time_interval)
        return sol
end
