using JLD
using DataFrames
using Dates
using LinearAlgebra
using DifferentialEquations
using Distributions
using Random
using Printf
include("utils.jl")
include("sir_ode.jl")
include("abc.jl")
include("abcMCMC.jl")
include("BayesianCalibration.jl")

master_data = load("data/covid_bc.jld")

df_pre = master_data["pre"]
df_post = master_data["post"]
