using DataFrames, CSV, Dates
using JLD

# import data
df_covid_raw = DataFrame(CSV.File("data/covid19-canada.csv"))

# look at BC only
df_covid_BC = select(filter(:prname => ==("British Columbia"),df_covid_raw),
                    :date,:numconf,:numactive,:numrecover)

# population of BC in 2021 is about 4.9 mil
# assume s0 is 4.9 mil
init_pop = 4900000
compute_s0(I,R) = init_pop - (I + R)
transform!(df_covid_BC,[:numactive,:numrecover] => ByRow(compute_s0) => :S)
rename!(df_covid_BC, :numactive => :I)
rename!(df_covid_BC, :numrecover => :R)
rename!(df_covid_BC, :date => :t)
select!(df_covid_BC,:t,:S,:I,:R)

# select the initial date
# choose 2020-03-26; no missing data
filter!(:t => >=(Date(2020,3,26)), df_covid_BC)

# before vaccine is administered: Dec 2020
df_covid_BC_pre = filter(:t => <(Date(2020,12,1)),df_covid_BC)

# after vaccine
df_covid_BC_post = filter(:t => >=(Date(2020,12,1)),df_covid_BC)


# save the output
CSV.write("data/covid19_bc_pre_vaccine.csv",df_covid_BC_pre)
CSV.write("data/covid19_bc_post_vaccine.csv",df_covid_BC_post)

# save as JLD
save("data/covid_bc.jld","pre",df_covid_BC_pre,"post",df_covid_BC_post)
