# PYOMO libraries and other libraries
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
import pandas as pd

#This values are just for the draft
# (i,j): [reactance (p.u.), max_flow (MW)]
# we will assume min_flow= -max_flow
line_data = {
    (1,2): [0.10682,150],
    (3,1): [0.10682,150],
    (2,3): [0.10682,150]}

#Reading a csv with all the units from the different technologies,
df_all_units = pd.read_csv ('C:\\Users\\seque\\Documents\\UoE\\Disertation\\PyOMO\\Draft4\\Plants_3nodes_woC.csv')

#create a df for thermal units and another for the units that belong to the state
thermal_tech = ['Combined Cycle', 'Thermal Power','Internal Combustion','Turbogas','Coal-fired']
df_thermal = df_all_units.loc[df_all_units['Technology'].isin(thermal_tech)]
df_thermal_CFE = df_thermal.loc[df_thermal['CFE']==1]
df_thermal_NoCFE = df_thermal.loc[df_thermal['CFE']==0]

#create a df for nuclear units and another for the units that belong to the state
df_nuclear = df_all_units.loc[df_all_units['Technology']== 'Nuclear']
df_nuclear_CFE = df_nuclear.loc[df_nuclear['CFE']==1]

#create a df for solar units and another for the units that belong to the state
df_solar = df_all_units.loc[df_all_units['Technology']== 'PV']
df_solar_CFE = df_solar.loc[df_solar['CFE']==1]

#create a df for wind units and another for the units that belong to the state
df_wind =  df_all_units.loc[df_all_units['Technology']== 'Wind']
df_wind_CFE = df_wind.loc[df_wind['CFE']==1]

#create a df for bioenergy units and another for the units that belong to the state
df_bioenergy =  df_all_units.loc[df_all_units['Technology']== 'Bioenergy']
df_bio_CFE = df_bioenergy.loc[df_bioenergy['CFE']==1]

#create a df for cogeneration units and another for the units that belong to the state
df_cogeneration =  df_all_units.loc[df_all_units['Technology']== 'Cogeneration']
df_cog_CFE = df_cogeneration.loc[df_cogeneration['CFE']==1]

#create a df for Geothermal units and another for the units that belong to the state
df_geothermal =  df_all_units.loc[df_all_units['Technology']== 'Geothermal']
df_geo_CFE = df_geothermal.loc[df_geothermal['CFE']==1]

#create a df for Hydro units and another for the units that belong to the state
df_hydro =  df_all_units.loc[df_all_units['Technology']== 'Hydro']
df_hydro_CFE = df_hydro.loc[df_hydro['CFE']==1]

#create a df for Run of River units and another for the units that belong to the state
df_river =  df_all_units.loc[df_all_units['Technology']== 'Run of river']
df_river_CFE = df_river.loc[df_river['CFE']==1]

#reading a file with the demand at each node (region)

df_demand = pd.read_csv ('C:\\Users\\seque\\Documents\\UoE\\Disertation\\PyOMO\\Draft4\\Demand_3nodes.csv')

#df that will help me to create the dictionaries for thermal units
df_gen_u = df_thermal[['Region','Plant','Capacity (MW)','Minimum Generation (MW)']]
df_gen_CFE_u = df_thermal_CFE[['Region','Plant','Capacity (MW)','Minimum Generation (MW)']]
df_gen_NoCFE_u = df_thermal_NoCFE[['Region','Plant','Capacity (MW)','Minimum Generation (MW)']]
df_gen_u_cost = df_thermal[['Region','Plant','HR A','HR B','HR C','Fuel_price']]

#df that will help me to create the dictionaries for nuclear
df_gen_q = df_nuclear[['Region','Plant','Capacity (MW)']]
df_gen_CFE_q = df_nuclear_CFE[['Region','Plant','Capacity (MW)']]
df_gen_q_cost = df_nuclear[['Region','Plant','HR A','HR B','HR C','Fuel_price']]

#df that will help me to create the dictionaries for solar
df_gen_s = df_solar[['Region','Plant','Capacity (MW)']]
df_gen_CFE_s = df_solar_CFE[['Region','Plant','Capacity (MW)']]

#df that will help me to create the dictionaries for wind
df_gen_w = df_wind[['Region','Plant','Capacity (MW)']]
df_gen_CFE_w = df_wind_CFE[['Region','Plant','Capacity (MW)']]

#df that will help me to create the dictionaries for bioenergy
df_gen_b = df_bioenergy[['Region','Plant','Capacity (MW)']]
df_gen_CFE_b = df_bio_CFE[['Region','Plant','Capacity (MW)']]

#df that will help me to create the dictionaries for cogeneration
df_gen_c = df_cogeneration[['Region','Plant','Capacity (MW)']]
df_gen_CFE_c = df_cog_CFE[['Region','Plant','Capacity (MW)']]

#df that will help me to create the dictionaries for geothermal
df_gen_g = df_geothermal[['Region','Plant','Capacity (MW)']]
df_gen_CFE_g = df_geo_CFE[['Region','Plant','Capacity (MW)']]

#df that will help me to create the dictionaries for hydro
df_gen_h = df_hydro[['Region','Plant','Capacity (MW)']]
df_gen_CFE_h = df_hydro_CFE[['Region','Plant','Capacity (MW)']]
df_gen_h_cost = df_hydro[['Region','Plant','Fuel_price']]

#df that will help me to create the dictionaries for run of river
df_gen_r = df_river[['Region','Plant','Capacity (MW)']]
df_gen_CFE_r = df_river_CFE[['Region','Plant','Capacity (MW)']]

#Creating the dictionaries with the a,b,c coefficients for thermal units

a_gen_u = {k: f.groupby('Plant')['HR A'].sum().to_dict()
     for k, f in df_gen_u_cost.groupby('Region')}
b_gen_u = {k: f.groupby('Plant')['HR B'].sum().to_dict()
     for k, f in df_gen_u_cost.groupby('Region')}
c_gen_u = {k: f.groupby('Plant')['HR C'].sum().to_dict()
     for k, f in df_gen_u_cost.groupby('Region')}

#Creating the dictionaries with the a,b,c coefficients for nuclear units

a_gen_q= {k: f.groupby('Plant')['HR A'].sum().to_dict()
     for k, f in df_gen_q_cost.groupby('Region')}
b_gen_q = {k: f.groupby('Plant')['HR B'].sum().to_dict()
     for k, f in df_gen_q_cost.groupby('Region')}
c_gen_q = {k: f.groupby('Plant')['HR C'].sum().to_dict()
     for k, f in df_gen_q_cost.groupby('Region')}

#Creating a dictionary with the fuel prices for thermal
fuel_gen_u = {k: f.groupby('Plant')['Fuel_price'].sum().to_dict()
     for k, f in df_gen_u_cost.groupby('Region')}

#Creating a dictionary with the fuel prices for nuclear
fuel_gen_q = {k: f.groupby('Plant')['Fuel_price'].sum().to_dict()
     for k, f in df_gen_q_cost.groupby('Region')}

#Creating a dictionary with the fuel prices for hydro
fuel_gen_h = {k: f.groupby('Plant')['Fuel_price'].sum().to_dict()
     for k, f in df_gen_h_cost.groupby('Region')}

#Creating a dictionary with the maximum generation for thermal units
max_gen_u= {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_u.groupby('Region')}

#Creating a dictionary with the minimum generation for thermal units
min_gen_u = {k: f.groupby('Plant')['Minimum Generation (MW)'].sum().to_dict()
     for k, f in df_gen_u.groupby('Region')}

#Creating a dictionary with the maximum generation for nuclear units
max_gen_q = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_q.groupby('Region')}

#Creating a dictionary with the maximum generation for solar units
max_gen_s = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_s.groupby('Region')}

#Creating a dictionary with the maximum generation for wind units
max_gen_w = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_w.groupby('Region')}

#Creating a dictionary with the maximum generation for bioenergy units
max_gen_b = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_b.groupby('Region')}

#Creating a dictionary with the maximum generation for cogeneration units
max_gen_c = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_c.groupby('Region')}

#Creating a dictionary with the maximum generation for geothermal units
max_gen_g = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_g.groupby('Region')}

#Creating a dictionary with the maximum generation for hydro units
max_gen_h = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_h.groupby('Region')}

#Creating a dictionary with the maximum generation for run of river units
max_gen_r = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_r.groupby('Region')}


pd.options.mode.chained_assignment = None
#A column with a tuple of the region and the id of the generators
#df_gen_u['n_g'] = df_gen_u[['Region','Plant']].apply(tuple, axis=1)
#this tuple for thermal does not include units from CFE 
df_gen_u['n_g'] = df_gen_NoCFE_u[['Region','Plant']].apply(tuple, axis=1)

df_gen_q['n_g'] = df_gen_q[['Region','Plant']].apply(tuple, axis=1)
df_gen_s['n_g'] = df_gen_s[['Region','Plant']].apply(tuple, axis=1)
df_gen_w['n_g'] = df_gen_w[['Region','Plant']].apply(tuple, axis=1)
df_gen_b['n_g'] = df_gen_b[['Region','Plant']].apply(tuple, axis=1)
df_gen_c['n_g'] = df_gen_c[['Region','Plant']].apply(tuple, axis=1)
df_gen_g['n_g'] = df_gen_g[['Region','Plant']].apply(tuple, axis=1)
df_gen_h['n_g'] = df_gen_h[['Region','Plant']].apply(tuple, axis=1)
df_gen_r['n_g'] = df_gen_r[['Region','Plant']].apply(tuple, axis=1)

#A column with a tuple of the region and the id of the CFE plant
df_gen_CFE_u['n_g'] = df_gen_CFE_u[['Region','Plant']].apply(tuple, axis=1)
df_gen_CFE_q['n_g'] = df_gen_CFE_q[['Region','Plant']].apply(tuple, axis=1)
df_gen_CFE_s['n_g'] = df_gen_CFE_s[['Region','Plant']].apply(tuple, axis=1)
df_gen_CFE_w['n_g'] = df_gen_CFE_w[['Region','Plant']].apply(tuple, axis=1)
df_gen_CFE_b['n_g'] = df_gen_CFE_b[['Region','Plant']].apply(tuple, axis=1)
df_gen_CFE_c['n_g'] = df_gen_CFE_c[['Region','Plant']].apply(tuple, axis=1)
df_gen_CFE_g['n_g'] = df_gen_CFE_g[['Region','Plant']].apply(tuple, axis=1)
df_gen_CFE_h['n_g'] = df_gen_CFE_h[['Region','Plant']].apply(tuple, axis=1)
df_gen_CFE_r['n_g'] = df_gen_CFE_r[['Region','Plant']].apply(tuple, axis=1)

#Creating a dictionary with the inelastic demand
inelastic_dem = {k: f.groupby('Consumer')['Demand (MWh)'].sum().to_dict()
     for k, f in df_demand.groupby('Region')}

#Creating a dictionary with keys for all the generators

#this dictionary only include thermal units that do no belong to CFE
gen_u_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_u.groupby('n_g')}


gen_q_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_q.groupby('n_g')}
gen_s_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_s.groupby('n_g')}
gen_w_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_w.groupby('n_g')}
gen_b_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_b.groupby('n_g')}
gen_c_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_c.groupby('n_g')}
gen_g_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_g.groupby('n_g')}
gen_h_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_h.groupby('n_g')}
gen_r_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_r.groupby('n_g')}

#Creating a dictionary with keys for the CFE generators
gen_u_CFE_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_CFE_u.groupby('n_g')}
gen_q_CFE_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_CFE_q.groupby('n_g')}
gen_s_CFE_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_CFE_s.groupby('n_g')}
gen_w_CFE_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_CFE_w.groupby('n_g')}
gen_b_CFE_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_CFE_b.groupby('n_g')}
gen_c_CFE_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_CFE_c.groupby('n_g')}
gen_g_CFE_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_CFE_g.groupby('n_g')}
gen_h_CFE_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_CFE_h.groupby('n_g')}
gen_r_CFE_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_CFE_r.groupby('n_g')}

#Definition of capacity factor for different technologies
CF_solar = 0.25
CF_wind = 0.30
CF_cogeneration = 0.3
CF_bioenergy = 0.5
CF_geothermal = 0.5
CF_river = 0.4

#Definition of the cost of curtailment for wind and solar
VWC = 1000
VSC = 1000

# PYOMO optimization model
mdl = ConcreteModel()

#----------------------- Defintion of the model SETS-------------------------

mdl.id_u = Set(initialize = df_thermal['Plant'].unique())
mdl.id_q = Set(initialize = df_nuclear['Plant'].unique())
mdl.id_s = Set(initialize = df_solar['Plant'].unique())
mdl.id_w = Set(initialize = df_wind['Plant'].unique())
mdl.id_b = Set(initialize = df_bioenergy['Plant'].unique())
mdl.id_c = Set(initialize = df_cogeneration['Plant'].unique())
mdl.id_g = Set(initialize = df_geothermal['Plant'].unique())
mdl.id_h = Set(initialize = df_hydro['Plant'].unique())
mdl.id_r = Set(initialize = df_river['Plant'].unique())

mdl.Consumers =Set(initialize = df_demand['Consumer'].unique())
mdl.Nodes = Set(initialize = df_demand['Region'].unique())
mdl.Lines = Set(initialize=line_data.keys(), dimen=2)

#create a set for the tuples of generators with nodes
#This first tupple does not include thermal units from CFE
mdl.gen_u_node = Set(initialize=gen_u_keys.keys(), dimen=2)

mdl.gen_q_node = Set(initialize=gen_q_keys.keys(), dimen=2)
mdl.gen_s_node = Set(initialize=gen_s_keys.keys(), dimen=2)
mdl.gen_w_node = Set(initialize=gen_w_keys.keys(), dimen=2)
mdl.gen_b_node = Set(initialize=gen_b_keys.keys(), dimen=2)
mdl.gen_c_node = Set(initialize=gen_c_keys.keys(), dimen=2)
mdl.gen_g_node = Set(initialize=gen_g_keys.keys(), dimen=2)
mdl.gen_h_node = Set(initialize=gen_h_keys.keys(), dimen=2)
mdl.gen_r_node = Set(initialize=gen_r_keys.keys(), dimen=2)

#create a set for the tuples of the state generators with the nodes
mdl.gen_u_CFE_node = Set(initialize=gen_u_CFE_keys.keys(), dimen=2)
mdl.gen_q_CFE_node = Set(initialize=gen_q_CFE_keys.keys(), dimen=2)
mdl.gen_s_CFE_node = Set(initialize=gen_s_CFE_keys.keys(), dimen=2)
mdl.gen_w_CFE_node = Set(initialize=gen_w_CFE_keys.keys(), dimen=2)
mdl.gen_b_CFE_node = Set(initialize=gen_b_CFE_keys.keys(), dimen=2)
mdl.gen_c_CFE_node = Set(initialize=gen_c_CFE_keys.keys(), dimen=2)
mdl.gen_g_CFE_node = Set(initialize=gen_g_CFE_keys.keys(), dimen=2)
mdl.gen_h_CFE_node = Set(initialize=gen_h_CFE_keys.keys(), dimen=2)
mdl.gen_r_CFE_node = Set(initialize=gen_r_CFE_keys.keys(), dimen=2)

# ---Defintion of the incidence matrix----
a = {}
for (i, j) in mdl.Lines:
    for n in mdl.Nodes:
        if n == i:
            a[i,j,n] = 1
        elif n == j:
            a[i,j,n] = -1
        else:
            a[i,j,n] = 0

# Variables for the generation of thermal, nuclear, solar, wind and hydro
mdl.gen_u = Var(mdl.gen_u_node, domain=NonNegativeReals)
#new variable for the generation of thermal units from CFE
mdl.gen_u_CFE = Var(mdl.gen_u_CFE_node, domain=NonNegativeReals)

mdl.gen_q = Var(mdl.gen_q_node, domain=NonNegativeReals)
mdl.gen_s = Var(mdl.gen_s_node, domain=NonNegativeReals)
mdl.gen_w = Var(mdl.gen_w_node, domain=NonNegativeReals)
mdl.gen_h = Var(mdl.gen_h_node, domain=NonNegativeReals)

# Variables for curtailed generation
mdl.gen_s_c = Var(mdl.gen_s_node, domain=NonNegativeReals)
mdl.gen_w_c = Var(mdl.gen_w_node, domain=NonNegativeReals)

# power flow variables
mdl.f = Var(mdl.Lines, domain=Reals)
# voltage phase angle variables
mdl.theta = Var(mdl.Nodes, domain=Reals)

#Checking the sets and variables definition
mdl.pprint()

# --------------------- Restrictions definitions------------------------------

#Constraint limiting the private thermal units  at its maximum value
def gen_u_max_rule(self, n, g):
    return mdl.gen_u[n,g] <= max_gen_u[n][g]
mdl.gen_u_max_constrait = Constraint(mdl.gen_u_node , rule=gen_u_max_rule)
mdl.gen_u_max_constrait.pprint()

#Constraint limiting the private thermal units at its minimum value
# def gen_u_min_rule(self, n, g):
#     return mdl.gen_u[n,g] >= min_gen_u[n][g]
# mdl.gen_u_min_constrait = Constraint(mdl.gen_u_node , rule=gen_u_min_rule)
# mdl.gen_u_min_constrait.pprint()

#--------------------------Constraints limiting the CFE thermal units ---------
def gen_u_CFE_max_rule(self, n, g):
    return mdl.gen_u_CFE[n,g] <= max_gen_u[n][g]
mdl.gen_u_CFE_max_constrait = Constraint(mdl.gen_u_CFE_node, rule=gen_u_CFE_max_rule)
mdl.gen_u_CFE_max_constrait.pprint()

#Constraint setting the CFE thermal generators at its minimum output
def gen_u_CFE_min_rule(self, n, g):
    return mdl.gen_u_CFE[n,g] >= min_gen_u[n][g]
mdl.gen_u_CFE_min_constrait = Constraint(mdl.gen_u_CFE_node, rule=gen_u_CFE_min_rule)
mdl.gen_u_CFE_min_constrait.pprint()

#-------------------------------------------------------------------------


#Constraint limiting the nuclear generators at its maximum value
def gen_q_max_rule(self, n, g):
    return mdl.gen_q[n,g] <= max_gen_q[n][g]
mdl.gen_q_max_constrait = Constraint(mdl.gen_q_node , rule=gen_q_max_rule)
mdl.gen_q_max_constrait.pprint()

# #Constraint minimun generation for nuclear 
# def gen_q_min_rule(self, n, g):
#     return mdl.gen_q[n,g] >= max_gen_q[n][g]*0.3
# mdl.gen_q_min_constrait = Constraint(mdl.gen_q_node , rule=gen_q_min_rule)
# mdl.gen_q_min_constrait.pprint()

#Constraint limiting the solar generators at its maximum value considering
#a possible curtailment
def gen_s_max_rule(self, n, g):
    return mdl.gen_s[n,g] + mdl.gen_s_c[n,g] == max_gen_s[n][g]*CF_solar
mdl.gen_s_max_constrait = Constraint(mdl.gen_s_node , rule=gen_s_max_rule)
mdl.gen_s_max_constrait.pprint()

#Constraint limiting the wind generators at its maximum value considering
#a possible curtailment
def gen_w_max_rule(self, n, g):
    return mdl.gen_w[n,g] + mdl.gen_w_c[n,g] == max_gen_w[n][g]*CF_wind
mdl.gen_w_max_constrait = Constraint(mdl.gen_w_node , rule=gen_w_max_rule)
mdl.gen_w_max_constrait.pprint()

#Constraint limiting the hydro generators at its maximum value
def gen_h_max_rule(self, n, g):
    return mdl.gen_h[n,g] <= max_gen_h[n][g]
mdl.gen_h_max_constrait = Constraint(mdl.gen_h_node , rule=gen_h_max_rule)
mdl.gen_h_max_constrait.pprint()


# -----------------Constrain regarding the share of the state generator
# this is the restriction that will help me to model the policy that want
# to fix the participation of the state in 54% on the generation-------------------
def state_share_rule(self, n):
    tot_dem = sum(inelastic_dem[n][k] for k in mdl.Consumers for n in mdl.Nodes)
    tot_gen_u_state = sum(mdl.gen_u_CFE[n,g] for g in mdl.id_u for n in mdl.Nodes  if (n,g) in mdl.gen_u_CFE_node)
    tot_gen_q_state = sum(mdl.gen_q[n,g] for g in mdl.id_q for n in mdl.Nodes  if (n,g) in mdl.gen_q_CFE_node)
    tot_gen_s_state = sum(mdl.gen_s[n,g] for g in mdl.id_s for n in mdl.Nodes  if (n,g) in mdl.gen_s_CFE_node)
    tot_gen_w_state = sum(mdl.gen_w[n,g] for g in mdl.id_w for n in mdl.Nodes  if (n,g) in mdl.gen_w_CFE_node)
    tot_gen_h_state = sum(mdl.gen_h[n,g] for g in mdl.id_h for n in mdl.Nodes  if (n,g) in mdl.gen_h_CFE_node)
    tot_gen_b_state = sum(max_gen_b[n][g] for g in mdl.id_b for n in mdl.Nodes  if (n,g) in mdl.gen_b_CFE_node)*CF_bioenergy
    tot_gen_c_state = sum(max_gen_c[n][g] for g in mdl.id_c for n in mdl.Nodes  if (n,g) in mdl.gen_c_CFE_node)*CF_cogeneration
    tot_gen_g_state = sum(max_gen_g[n][g] for g in mdl.id_g for n in mdl.Nodes  if (n,g) in mdl.gen_g_CFE_node)*CF_geothermal
    tot_gen_r_state = sum(max_gen_r[n][g] for g in mdl.id_r for n in mdl.Nodes  if (n,g) in mdl.gen_r_CFE_node)*CF_river

    return (tot_gen_u_state + tot_gen_q_state + tot_gen_s_state +tot_gen_w_state +tot_gen_h_state
            +tot_gen_b_state +tot_gen_c_state +tot_gen_g_state +tot_gen_r_state) >= 0.54*tot_dem
mdl.gen_state_constrait = Constraint(mdl.Nodes , rule=state_share_rule)
mdl.gen_state_constrait.pprint()


# power balance Constraint
def power_balance_rule(self, n):
    tot_dem = sum(inelastic_dem[n][k] for k in mdl.Consumers)
    tot_gen_u = sum(mdl.gen_u[n,g] for g in mdl.id_u if (n,g) in mdl.gen_u_node)
    #Generation from CFE thermal units--------------------------------------------------------
    tot_gen_u_CFE = sum(mdl.gen_u_CFE[n,g] for g in mdl.id_u if (n,g) in mdl.gen_u_CFE_node)
    #-----------------------------------------------------------------------------------------
    tot_gen_q = sum(mdl.gen_q[n,g] for g in mdl.id_q if (n,g) in mdl.gen_q_node)
    tot_gen_s = sum(mdl.gen_s[n,g] for g in mdl.id_s if (n,g) in mdl.gen_s_node)
    tot_gen_w = sum(mdl.gen_w[n,g] for g in mdl.id_w if (n,g) in mdl.gen_w_node)
    tot_gen_h = sum(mdl.gen_h[n,g] for g in mdl.id_h if (n,g) in mdl.gen_h_node)
    tot_gen_b = sum(max_gen_b[n][g] for g in mdl.id_b if (n,g) in mdl.gen_b_node)*CF_bioenergy
    tot_gen_c = sum(max_gen_c[n][g] for g in mdl.id_c if (n,g) in mdl.gen_c_node)*CF_cogeneration
    tot_gen_g = sum(max_gen_g[n][g] for g in mdl.id_g if (n,g) in mdl.gen_g_node)*CF_geothermal
    tot_gen_r = sum(max_gen_r[n][g] for g in mdl.id_r if (n,g) in mdl.gen_r_node)*CF_river
    flows = sum(a[i,j,n]*mdl.f[i,j] for (i,j) in mdl.Lines)
    return tot_dem - tot_gen_u -tot_gen_u_CFE - tot_gen_q - tot_gen_s - tot_gen_w - tot_gen_h - tot_gen_b - tot_gen_c - tot_gen_g - tot_gen_r + flows == 0
mdl.power_balance = Constraint(mdl.Nodes, rule=power_balance_rule)
mdl.power_balance.pprint()



# Constraint for the min flow in the lines
# we will assume min_flow= -max_flow
def min_flow_rule(self, i,j):
    return - line_data[(i,j)][1] <= mdl.f[i,j]
mdl.min_flow = Constraint(mdl.Lines, rule=min_flow_rule)
mdl.min_flow.pprint()

# max flow in the lines between regions
def max_flow_rule(self, i,j):
    return mdl.f[(i,j)] <= line_data[(i,j)][1]
mdl.max_flow = Constraint(mdl.Lines, rule=max_flow_rule)
mdl.max_flow.pprint()

# DC load flow restriction
def DC_loadflow_rule(self, i, j):
    return mdl.f[(i,j)] == (mdl.theta[i] - mdl.theta[j])/line_data[(i,j)][0]
mdl.DC_loadflow = Constraint(mdl.Lines, rule=DC_loadflow_rule)
mdl.DC_loadflow.pprint()


# ------------------Denition of the Objective function-------------------------
cost_thermal_gen = sum((a_gen_u[n][k] * mdl.gen_u[n,k]** 2 +b_gen_u[n][k] * mdl.gen_u[n,k] + c_gen_u[n][k])*fuel_gen_u[n][k]
                        for k in mdl.id_u for n in mdl.Nodes if (n,k) in mdl.gen_u_node)

cost_thermal_CFE_gen = sum((a_gen_u[n][k] * mdl.gen_u_CFE[n,k]** 2 +b_gen_u[n][k] * mdl.gen_u_CFE[n,k] + c_gen_u[n][k])*fuel_gen_u[n][k]
                        for k in mdl.id_u for n in mdl.Nodes if (n,k) in mdl.gen_u_CFE_node)

cost_nuclear_gen = sum((a_gen_q[n][k] * mdl.gen_q[n,k]** 2 +b_gen_q[n][k] * mdl.gen_q[n,k] + c_gen_q[n][k])*fuel_gen_q[n][k]
                        for k in mdl.id_q for n in mdl.Nodes if (n,k) in mdl.gen_q_node)

curtailment_solar = sum(VSC*mdl.gen_s_c[n,k] for k in mdl.id_s for n in mdl.Nodes if (n,k) in mdl.gen_s_node)

curtailment_wind = sum(VWC*mdl.gen_w_c[n,k] for k in mdl.id_w for n in mdl.Nodes if (n,k) in mdl.gen_w_node)

cost_hydro_gen = sum(fuel_gen_h[n][k] * mdl.gen_h[n,k] for k in mdl.id_h for n in mdl.Nodes if (n,k) in mdl.gen_h_node)

mdl.obj = Objective(expr= -(cost_thermal_gen +cost_thermal_CFE_gen + cost_nuclear_gen + curtailment_solar + curtailment_wind +cost_hydro_gen),
                    sense=maximize)


# checking the objective function
mdl.obj.pprint()

# We have to tell Pyomo that we want dual variables
mdl.dual = Suffix(direction=Suffix.IMPORT)

# Create an object representing the solver, in this case GUROBI
solver = SolverFactory("gurobi")

###------------------Solving the model and printing the results----------------

# solve the optimization problem
results = solver.solve(mdl, tee=True)

# ALWAYS check solver's termination condition
if results.solver.termination_condition != TerminationCondition.optimal:
    raise Exception
else:
    print(results.solver.status)
    print(results.solver.termination_condition)
    print(results.solver.termination_message)
    print(results.solver.time)

# # allocated demand
# for n in mdl.Nodes:
#     for k in mdl.Consumers:
#         print("dem[%d, %s]=%.4f" % (n, k, inelastic_dem[n][k]))

# # allocated thermal generation
# for n in mdl.Nodes:
#     for g in mdl.id_u:
#         if (n,g) in mdl.gen_u_node and mdl.gen_u[n, g].value != 0:
#             print("thermal[%d, %s]=%.4f" % (n, g, mdl.gen_u[n,g].value))

# # allocated nuclear generation
# for n in mdl.Nodes:
#     for g in mdl.id_q:
#         if (n,g) in mdl.gen_q_node and mdl.gen_q[n, g].value != 0:
#             print("nuclear[%d, %s]=%.4f" % (n, g, mdl.gen_q[n,g].value))

# # allocated solar generation
# for n in mdl.Nodes:
#     for g in mdl.id_s:
#         if (n,g) in mdl.gen_s_node and mdl.gen_s[n, g].value != 0:
#             print("solar[%d, %s]=%.4f" % (n, g, mdl.gen_s[n,g].value))

# #  solar curtailment
# for n in mdl.Nodes:
#     for g in mdl.id_s:
#         if (n,g) in mdl.gen_s_node and mdl.gen_s_c[n, g].value != 0:
#             print("solar_c[%d, %s]=%.4f" % (n, g, mdl.gen_s_c[n,g].value))

# # allocated wind generation
# for n in mdl.Nodes:
#     for g in mdl.id_w:
#         if (n,g) in mdl.gen_w_node and mdl.gen_w[n, g].value != 0:
#             print("wind[%d, %s]=%.4f" % (n, g, mdl.gen_w[n,g].value))

# #  wind curtailment
# for n in mdl.Nodes:
#     for g in mdl.id_w:
#         if (n,g) in mdl.gen_w_node and mdl.gen_w_c[n, g].value != 0:
#             print("wind_c[%d, %s]=%.4f" % (n, g, mdl.gen_w_c[n,g].value))

# # allocated hydro generation
# for n in mdl.Nodes:
#     for g in mdl.id_h:
#         if (n,g) in mdl.gen_h_node and mdl.gen_h[n, g].value != 0:
#             print("h[%d, %s]=%.4f" % (n, g, mdl.gen_h[n,g].value))

# # bioenergy generation
# for n in mdl.Nodes:
#     for g in mdl.id_b:
#         if (n,g) in mdl.gen_b_node:
#             print("b[%d, %s]=%.4f" % (n, g, max_gen_b[n][g]*CF_bioenergy))

# # cogeneration generation
# for n in mdl.Nodes:
#     for g in mdl.id_c:
#         if (n,g) in mdl.gen_c_node:
#             print("c[%d, %s]=%.4f" % (n, g, max_gen_c[n][g]*CF_cogeneration))

# # geothermal generation
# for n in mdl.Nodes:
#     for g in mdl.id_g:
#         if (n,g) in mdl.gen_g_node:
#             print("g[%d, %s]=%.4f" % (n, g, max_gen_g[n][g]*CF_geothermal))

# # run of river generation
# for n in mdl.Nodes:
#     for g in mdl.id_r:
#         if (n,g) in mdl.gen_r_node:
#             print("r[%d, %s]=%.4f" % (n, g, max_gen_r[n][g]*CF_river))

#----- nodal prices-------!!!
for n in mdl.Nodes:
    print("nodal price[%d]=%.4f" % (n, mdl.dual[mdl.power_balance[n]]))


# -----power flows-----!!!
for (i,j) in mdl.Lines:
    print("flow[%d,%d]=%.4f" % (i,j, mdl.f[i,j].value))
    

# print objective function value
print("Objective Function value=", value(mdl.obj))
    

#------------------Saving results in csv files------------------------------

#-----results for CFE thermal units dispatched
results_thermal_CFE= pd.DataFrame(columns=['n','g','Dispatched Thermal','Marginal Cost','Cost'])

for n in mdl.Nodes:
    for g in mdl.id_u:
        if (n,g) in mdl.gen_u_CFE_node:
            results_thermal_CFE = results_thermal_CFE.append({'n':n, 'g':g,'Dispatched Thermal':mdl.gen_u_CFE[n,g].value,
                              'Marginal Cost':(2*a_gen_u[n][g]*mdl.gen_u_CFE[n,g].value+ b_gen_u[n][g])*fuel_gen_u[n][g],
                              'Cost':(a_gen_u[n][g]*mdl.gen_u_CFE[n,g].value**2+ b_gen_u[n][g]*mdl.gen_u_CFE[n,g].value)*fuel_gen_u[n][g]},
                                                      ignore_index=True)
#results_thermal.to_csv('Thermal_dispatched.csv', index=False)
#results_thermal_CFE_2 = results_thermal_CFE.groupby(by=['n'], dropna=False)[['Dispatched Thermal']].sum()


#-----results for non CFE thermal units dispatched
results_thermal= pd.DataFrame(columns=['n','g','Dispatched Thermal','Marginal Cost','Cost'])

for n in mdl.Nodes:
    for g in mdl.id_u:
        if (n,g) in mdl.gen_u_node:
            results_thermal = results_thermal.append({'n':n, 'g':g,'Dispatched Thermal':mdl.gen_u[n,g].value,
                              'Marginal Cost':(2*a_gen_u[n][g]*mdl.gen_u[n,g].value+ b_gen_u[n][g])*fuel_gen_u[n][g],
                              'Cost':(a_gen_u[n][g]*mdl.gen_u[n,g].value**2+ b_gen_u[n][g]*mdl.gen_u[n,g].value)*fuel_gen_u[n][g]},
                                                      ignore_index=True)


#-----results for nuclear units dispatched
results_nuclear= pd.DataFrame(columns=['n','g','Dispatched Nuclear','Marginal Cost','Cost'])

for n in mdl.Nodes:
    for g in mdl.id_q:
        if (n,g) in mdl.gen_q_node:
            results_nuclear = results_nuclear.append({'n':n, 'g':g,'Dispatched Nuclear':mdl.gen_q[n,g].value,
                             'Marginal Cost':(2*a_gen_q[n][g]*mdl.gen_q[n,g].value + b_gen_q[n][g])*fuel_gen_q[n][g],
                             'Cost':(a_gen_q[n][g]*mdl.gen_q[n,g].value** 2 + b_gen_q[n][g]*mdl.gen_q[n,g].value)*fuel_gen_q[n][g]},
                                                     ignore_index=True)
results_nuclear.to_csv('Nuclear_dispatched.csv', index=False)
results_nuclear_2 = results_nuclear.groupby(by=['n'], dropna=False)[['Dispatched Nuclear']].sum()


#-----results for hydro units dispatched
results_hydro= pd.DataFrame(columns=['n','g','Dispatched Hydro','Marginal Cost','Cost'])
for n in mdl.Nodes:
    for g in mdl.id_h:
        if (n,g) in mdl.gen_h_node:
            results_hydro = results_hydro.append({'n':n, 'g':g,'Dispatched Hydro':mdl.gen_h[n,g].value,
                             'Marginal Cost':fuel_gen_h[n][g],
                             'Cost':mdl.gen_h[n,g].value*fuel_gen_h[n][g]},ignore_index=True)
results_hydro.to_csv('Hydro_dispatched.csv', index=False)
results_hydro_2 = results_hydro.groupby(by=['n'], dropna=False)[['Dispatched Hydro']].sum()

#-----results for wind
results_wind= pd.DataFrame(columns=['n','g','Wind Generation','Wind Curtailment'])

for n in mdl.Nodes:
    for g in mdl.id_w:
        if (n,g) in mdl.gen_w_node:
            results_wind = results_wind.append({'n':n, 'g':g, 'Wind Generation':mdl.gen_w[n, g].value,
                                                    'Wind Curtailment':mdl.gen_w_c[n, g].value,
                                               'VWC':VWC,'Wind Cost':mdl.gen_w_c[n, g].value*VWC},ignore_index=True)
results_wind.to_csv('Wind_dispatched.csv', index=False)
results_wind_2 = results_wind.groupby(by=['n'], dropna=False)[['Wind Generation','Wind Curtailment']].sum()

#-----results for solar 
results_solar= pd.DataFrame(columns=['n','g','Solar Generation','Solar Curtailment','VSC','Solar Cost'])

for n in mdl.Nodes:
    for g in mdl.id_s:
        if (n,g) in mdl.gen_s_node:
            results_solar = results_solar.append({'n':n, 'g':g, 'Solar Generation':mdl.gen_s[n, g].value
                                                      ,'Solar Curtailment':mdl.gen_s_c[n, g].value,
                                                 'VSC':VSC,'Solar Cost':mdl.gen_s_c[n, g].value*VSC},ignore_index=True)   
results_solar.to_csv('Solar_dispatched.csv', index=False)
results_solar_2 = results_solar.groupby(by=['n'], dropna=False)[['Solar Generation','Solar Curtailment']].sum()

#-----results for demand alocated and nodal prices
demand_prices_nodes = pd.DataFrame(columns=['n','Demand MW','Nodal Price'])

for n in mdl.Nodes:
    for k in mdl.Consumers:
        demand_prices_nodes = demand_prices_nodes.append({'n':n, 'Demand MW':inelastic_dem[n][k],
                                                          'Nodal Price':mdl.dual[mdl.power_balance[n]]},ignore_index=True)
demand_prices_nodes.to_csv('Demand_Prices.csv', index=False)

#-------------Dataframes for not dispatchable technologies

results_bioenergy= pd.DataFrame(columns=['n','g','Bioenergy Generation'])
for n in mdl.Nodes:
    for g in mdl.id_b:
        if (n,g) in mdl.gen_b_node:
            results_bioenergy = results_bioenergy.append({'n':n,'g':g ,'Bioenergy Generation':max_gen_b[n][g]*CF_bioenergy},
                                                         ignore_index=True)
results_bioenergy = results_bioenergy.groupby(by=['n'], dropna=False).sum()
            
results_cogeneration= pd.DataFrame(columns=['n','g','Cogeneration'])
for n in mdl.Nodes:
    for g in mdl.id_c:
        if (n,g) in mdl.gen_c_node:
            results_cogeneration = results_cogeneration.append({'n':n,'g':g ,'Cogeneration':max_gen_c[n][g]*CF_cogeneration},
                                                         ignore_index=True)
results_cogeneration = results_cogeneration.groupby(by=['n'], dropna=False).sum()
            
results_geothermal= pd.DataFrame(columns=['n','g','Geothermal Generation'])
for n in mdl.Nodes:
    for g in mdl.id_g:
        if (n,g) in mdl.gen_g_node:
            results_geothermal = results_geothermal.append({'n':n,'g':g ,'Geothermal Generation':max_gen_g[n][g]*CF_geothermal},
                                                         ignore_index=True)
results_geothermal = results_geothermal.groupby(by=['n'], dropna=False).sum()

results_river= pd.DataFrame(columns=['n','g','Run of river Generation'])
for n in mdl.Nodes:
    for g in mdl.id_r:
        if (n,g) in mdl.gen_r_node:
            results_river = results_river.append({'n':n,'g':g ,'Run of river Generation':max_gen_r[n][g]*CF_river},
                                                         ignore_index=True)
results_river = results_river.groupby(by=['n'], dropna=False).sum()

results_ND_tech = pd.concat([results_bioenergy, results_river,results_geothermal,results_cogeneration], axis=1)
results_ND_tech.to_csv('ND Technologies Generation.csv', index=True)

# Generation_All_tech = pd.concat([results_thermal_2,results_nuclear_2,results_hydro_2,
#                                  results_wind_2,results_solar_2,results_ND_tech], axis=1)

# Generation_All_tech.to_csv('All Technologies Generation.csv', index=True)








