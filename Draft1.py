# PYOMO libraries and other libraries
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
import pandas as pd


#This values are just for the draft
# (i,j): [reactance (p.u.), max_flow (MW)]
# we will assume min_flow= -max_flow
line_data = {
    (1, 2): [0.004, 10000],
    (1, 3): [0.002, 10000],
    (2, 3): [0.002, 10000],
    (3, 4): [0.002, 0],
    (3, 5): [0.003, 10000],
    (4, 5): [0.003, 0],
}

# PYOMO optimization model
mdl = ConcreteModel()

#reading the csv file with the plant characteristics
df_plants = pd.read_csv ('C:\\Users\\seque\\Documents\\UoE\\Disertation\\PyOMO\\Generators_draft1.csv')

#Creating a df wiht only the plants that belong to the Mexico state company
df_plants_CFE = df_plants.loc[df_plants['CFE']==1]

#reading a file with the demand at each node (region)

df_demand = pd.read_csv ('C:\\Users\\seque\\Documents\\UoE\\Disertation\\PyOMO\\Demand_draft1.csv')

#df that will help me to create the dictionaries
df_gen = df_plants[['Region','Plant','Capacity (MW)']]
df_gen_CFE = df_plants_CFE[['Region','Plant','Capacity (MW)']]
df_gen_cost = df_plants[['Region','Plant','HR A','HR B','HR C','Fuel_price']]


#Creating the dictionaries with the a,b,c coefficients

a_gen = {k: f.groupby('Plant')['HR A'].sum().to_dict()
     for k, f in df_gen_cost.groupby('Region')}
b_gen = {k: f.groupby('Plant')['HR B'].sum().to_dict()
     for k, f in df_gen_cost.groupby('Region')}
c_gen = {k: f.groupby('Plant')['HR C'].sum().to_dict()
     for k, f in df_gen_cost.groupby('Region')}

#Creating a dictionary with the fuel prices
fuel_gen = {k: f.groupby('Plant')['Fuel_price'].sum().to_dict()
     for k, f in df_gen_cost.groupby('Region')}

#Creating a dictionary with the maximum generation
max_gen = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen.groupby('Region')}

#A column with a tuple of tge region and the id of the generators
df_gen['n_g'] = df_gen[['Region','Plant']].apply(tuple, axis=1)
#A column with a tuple of tge region and the id of the CFE plant
df_gen_CFE['n_g'] = df_gen_CFE[['Region','Plant']].apply(tuple, axis=1)

#Creating a dictionary with the inelastic demand
inelastic_dem = {k: f.groupby('Consumer')['Demand (MW)'].sum().to_dict()
     for k, f in df_demand.groupby('Region')}

#Creating a dictionary with the generators
gen_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen.groupby('n_g')}
#Creating a dictionary with the CFE generators
gen_CFE_keys = {k: f.groupby('Plant')['Capacity (MW)'].sum().to_dict()
     for k, f in df_gen_CFE.groupby('n_g')}


#----------------------- Defintion of the model SETS-------------------------

mdl.Generators = Set(initialize = df_plants['Plant'].unique())
mdl.Consumers =Set(initialize = df_demand['Consumer'].unique())
mdl.Nodes = Set(initialize = df_demand['Region'].unique())
mdl.Lines = Set(initialize=line_data.keys(), dimen=2)

#create a set for the tuples of generators with nodes
mdl.gen_node = Set(initialize=gen_keys.keys(), dimen=2)
#create a set for the tuples of the state generators with the nodes
mdl.gen_CFE_node = Set(initialize=gen_CFE_keys.keys(), dimen=2)


# Defintion of the incidence matrix-------
a = {}
for (i, j) in mdl.Lines:
    for n in mdl.Nodes:
        if n == i:
            a[i,j,n] = 1
        elif n == j:
            a[i,j,n] = -1
        else:
            a[i,j,n] = 0

# -----------------Definition of variables----------------------------------

# Althought the demand is inelastic, a demand variable is still defined
mdl.dem = Var(mdl.Nodes, mdl.Consumers, domain=NonNegativeReals)
# Variables for generation
mdl.gen = Var(mdl.gen_node, domain=NonNegativeReals)

# power flow variables
mdl.f = Var(mdl.Lines, domain=Reals)
# voltage phase angle variables
mdl.theta = Var(mdl.Nodes, domain=Reals)

# --------------------- Restrictions definitions---------------------------------

# Demand Constraint considering the demand being inelastic
def dem_max_rule(self, n, k):
    return mdl.dem[n,k] == inelastic_dem[n][k]
mdl.dem_max_constrait = Constraint(mdl.Nodes, mdl.Consumers, rule=dem_max_rule)

#Constraint limiting the generation of eacg generators at its maximum value
def gen_max_rule(self, n, g):
    return mdl.gen[n,g] <= max_gen[n][g]
mdl.gen_max_constrait = Constraint(mdl.gen_node , rule=gen_max_rule)

# power balance Constraint
def power_balance_rule(self, n):
    tot_dem = sum(mdl.dem[n, k] for k in mdl.Consumers)
    tot_gen = sum(mdl.gen[n,g] for g in mdl.Generators if (n,g) in mdl.gen_node)
    flows = sum(a[i,j,n]*mdl.f[i,j] for (i,j) in mdl.Lines)
    return tot_dem - tot_gen + flows == 0
mdl.power_balance = Constraint(mdl.Nodes, rule=power_balance_rule)

#Constrain regarding the share of the state generator
#this is the restriction that will help me to model the policy that want
#to fix the participation of the state in 54% on the generation
def state_share_rule(self, n, g):
    tot_dem = sum(mdl.dem[n, k] for k in mdl.Consumers for n in mdl.Nodes)
    tot_gen_state = sum(mdl.gen[n,g] for g in mdl.Generators for n in mdl.Nodes  if (n,g) in mdl.gen_CFE_node)
    return tot_gen_state == 0*tot_dem

mdl.gen_state_constrait = Constraint(mdl.gen_CFE_node , rule=state_share_rule)

# Constraint for the min flow in the lines
# we will assume min_flow= -max_flow
def min_flow_rule(self, i,j):
    return - line_data[(i,j)][1] <= mdl.f[i,j]
mdl.min_flow = Constraint(mdl.Lines, rule=min_flow_rule)

# max flow in the lines between regions
def max_flow_rule(self, i,j):
    return mdl.f[(i,j)] <= line_data[(i,j)][1]
mdl.max_flow = Constraint(mdl.Lines, rule=max_flow_rule)

# DC load flow restriction
def DC_loadflow_rule(self, i, j):
    return mdl.f[(i,j)] == (mdl.theta[i] - mdl.theta[j])/line_data[(i,j)][0]
mdl.DC_loadflow = Constraint(mdl.Lines, rule=DC_loadflow_rule)


# ------------------Denition of the Objective function----------------------

cost_thermal_gen = sum((a_gen[n][k] * mdl.gen[n,k]** 2 +b_gen[n][k] * mdl.gen[n,k] + c_gen[n][k])*fuel_gen[n][k]
                        for k in mdl.Generators for n in mdl.Nodes if (n,k) in mdl.gen_node)

mdl.obj = Objective(expr=cost_thermal_gen, sense=minimize)


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

# print objective function value
print("Objective Function value=", value(mdl.obj))

# allocated demand
for n in mdl.Nodes:
    for k in mdl.Consumers:
        if mdl.dem[n,k].value != 0:
            print("dem[%d, %s]=%.4f" % (n, k, mdl.dem[n,k].value))

# allocated generation
for n in mdl.Nodes:
    for g in mdl.Generators:
        if (n,g) in mdl.gen_node and mdl.gen[n, g].value != 0:
            print("gen[%d, %s]=%.4f" % (n, g, mdl.gen[n,g].value))

# nodal prices?
for n in mdl.Nodes:
    print("nodal price[%d]=%.4f" % (n, mdl.dual[mdl.power_balance[n]]))

# contraints of demand
for n in mdl.Nodes:
    for k in mdl.Consumers:
        print("dem_max_constrait[%d,%s]=%.3f" % (n, k, mdl.dual[mdl.dem_max_constrait[n,k]]))
# flows
for (i,j) in mdl.Lines:
    print("flow[%d,%d]=%.4f" % (i,j, mdl.f[i,j].value))

# checking the whole solved model
mdl.pprint()
