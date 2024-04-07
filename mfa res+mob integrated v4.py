# -*- coding: utf-8 -*-
"""
Created on Thu May 26 23:19:48 2022

@author: Laura À. Pérez-Sánchez
laura.perez.sanchez@uab.cat

v4 - Nov 2022 - edited to include sensitivity scenarios

"""

import pandas as pd           
import numpy as np    
import scipy.stats   
import math 
import matplotlib.pyplot as plt 
from itertools import product
from datetime import datetime

currentDateAndTimeI = datetime.now()
print("The current date and time is", currentDateAndTimeI)

################################
##### %% 1.initialization ######
################################

### CALL THE DATA ###

path = "data"

 # (a) Population Statistics Sweden per type of municipality 
 #(1968-2021) - BE0101N1
pop_BE0101N1 = pd.read_csv(path + '/a - BE0101N1 - population.csv', sep='\t', 
                           index_col=[0, 1], header = 0) 
#region	"year"	Population

#### DWELLINGS ####

# (e) Dwellings Statistics Sweden Number of dwellings by region, type of building, period of construction and year 
#(2013-2020) - BO0104AB
#"region	""type of building""	""period of construction""	""year""	Number of dwellings"
dw_BO0104AB = pd.read_csv(path + '/e - BO0104AB - dwellings dem structure.csv', sep='\t', 
                          index_col=[0, 1], header = 0)  

# (f, g) Lifetime models
lifetime_dwellings = pd.read_csv(path + '/f - g - lifetime model apartments and houses.txt', sep='\t', 
                          index_col=[0, 1, 2], header = 0)  

# (10, 12, 13) Completed dwellings in newly constructed buildings by year, region and type of building
# 1975 - 2020
constr_dwellings = pd.read_csv(path + '/m-n - BO0101A5 - construction dwellings.csv', sep='\t', 
                               index_col=[0, 1], header = 0)
#year	"region"	"type of building"	Completed dwellings in newly constructed buildings

### (l) construction cap
constr_cap = pd.read_csv(path + "/l - construction cap.txt", sep = "\t",
                         index_col = [0], header = 0)

# (p, u) Dwelling size (m2/dw) and Floor Space Index (m2/ha)
area_characteristics = pd.read_csv(path + '/p-u - area_characteristics.txt', sep='\t',
                 index_col=[0, 1, 2],header = [0,1]) 
## year	municipality type	dwelling type	dwelling_size	FSI

# (q) Material intensity dwellings - Gontia et al 2018 #kg/m2
mat_int_dwellings = pd.read_csv(path + '/q - material intensity dwellings.txt', sep='\t', 
                               index_col = [0,1], header = 0)
# type of dwelling, cohort, [list of materials]

# (s) Emission intensity materials dwellings - Lausselet 2020 #kgCO2e/kg
emi_int_dwellings = pd.read_csv(path + '/s - material emission intensity dwellings.txt', sep='\t', 
                               index_col=[0,1], header = 0)
#Material_totlist   Material   material intensity

#### CARS ####

# (x) Equation cars per capita - municipalities excel (calculated from rvu sweden 2015)
cars_per_capita_equation = pd.read_csv(path + "/x - cars per capita equation.txt", sep='\t',
                                       index_col= 0, header=[0] )

#(z) Statistics Sweden - Vehicles in use by year, region and type of vehicle
number_cars_mun = pd.read_csv(path + '/z - TK1001AC - cars municipality.csv', sep='\t',
                 header = 0, encoding='latin-1') 
## year	"region"	"type of vehicle"	Vehicles in use
#type of vehicle in this case always is "passenger cars"
#region is municipalities and counties

#(z) share cohorts in 2020
share_cohorts_cars_2020 = pd.read_csv(path + '/z - share cohorts 2020.txt', sep='\t',
                index_col=[0], header=[0]) 
share_cohorts_cars_2020.columns = share_cohorts_cars_2020.columns.map(int) 


# (ae) ICCT - Average size of cars (mass in running order) (cohorts)
size_cars_new = pd.read_csv(path + '/ae - mass in running order.txt', sep='\t',
                index_col=[0], header=[0,1]) 
# 2001-2020
# 

# (af) ICCT - Share powertrains in new cars (cohorts)
share_powertrain_new_raw = pd.read_csv(path + '/af - powertrain type.txt', sep='\t',
                index_col=[0, 1], header=[0,1]) 
# 2001-2020

# (ag) Material intensity cars (kg/car)
mat_int_cars = pd.read_csv(path + "/ag - Material intensity cars.txt", sep='\t', 
                           index_col= 0, header=[0] )
# shares of materials 
# row: powert, col: mat

# (ai) Emissions per vehicle (kg CO2e/veh)- derived from Wolfram et al 2020
emi_int_prod_cars = pd.read_csv(path + "/ai - emissions prod vehicles per vehicle.txt", sep='\t',
                                       index_col= 0, header=[0] )
#powertrain  a_coef b_coef

# (ak) Equation distance per capita #exponential
distance_per_capita_equation = pd.read_csv(path + "/ak - distance per capita equation.txt", sep='\t',
                                       index_col= 0, header=[0] )

## (al) Coefficients adaptation veh·km by age of vehicle
adapt_vehkm = pd.read_csv(path + "/al - vehkm adaptation per age of vehicle.txt", sep='\t',
                                       index_col = 0, header = [0] )

# (aj) occupancy rate
occupancy_rate_all = pd.read_csv(path + "/aj - occupancy rate.txt", sep='\t',
                                       index_col = 0, header = [0] )

# (ap) Fuel efficiency regression  - Derived from Wolfram et al 2020
# kWh/100km
fuel_economy_regression = pd.read_csv(path + '/ap - Fuel efficiency.txt', sep='\t',
                index_col=0, header=[0])
# Powertrain	scalar_mult	scalar_exp	gasoline	diesel	electricity	hydrogen	natural gas

# (aq) Fuel efficiency - coefficient adapt older cohorts
fuel_economy_coefficient = pd.read_csv(path + '/aq - Fuel efficiency coef older cohorts.txt', sep = '\t',
                                       index_col = 0, header = [0])
# cohort   coef_old_fueleff

## (an) Allocation Energy carriers
allocation_energy =  pd.read_csv(path + '/an - allocation energy carriers.txt', sep='\t',
                index_col=0, header=[0])

# (at) Emission intensity — fuel use
# gCO2e/kWh
emi_int_dir_fuel_use = pd.read_csv(path + '/at - Emission intensity - fuel use.txt', sep='\t',
                index_col=[0], header=[0]) 
 
# (av) Emission intensity - indirect fuel use 
# gCO2e/kWh
emi_int_indir_fuel_use = pd.read_csv(path + '/av - Emission intensity - fuel manufacturing.txt', sep='\t',
                index_col=[0,1], header=[0]) 

#### MAPPING ####

#scenarios
scenario_characteristics = pd.read_csv(path + '/table_scenarios_sensitivity.txt',sep = "\t",
                                       index_col=[0,1], header=[0])

# Municipality types
mun_group = pd.read_csv(path + '/mapping - municipality types.txt', sep='\t',
               index_col=[0], header=[0], encoding='latin-1') 
mun_group = mun_group.loc[:,["municipality","region"]]
mun_group = mun_group.rename(columns = {"region":"mun_group"})
# mun_code	Kommun	municipality	mun_group_num	Kommungrupp_namn	mun_group	mun_group_sum

# Year intervals of the dwelling stock
year_intervals = pd.read_csv(path + '/mapping - periods dwellings.txt', sep='\t',
               index_col=[0], header=[0]) 

# Dwelling types
dwelling_types = pd.read_csv(path + '/mapping - dwelling types.txt', sep='\t',
               index_col=[0], header=[0]) 


### % Exogenous future scenarios % ####

## (a) Population growth with reference to 2020 (time_horizon)
fut_pop = pd.read_csv(path + '/a - Eurostat - projections population - proj_19np.txt', sep='\t',
               index_col=[0], header=[0]) 

## (b) Household size (with reference to 2020)
fut_hhsize = pd.read_csv(path + '/b - Household size.txt', sep='\t',
               index_col=[0], header=[0,1]) 


# (k) Type new dwellings
type_new_buildings_tot = pd.read_csv(path + '/k - type new dwellings.txt', sep='\t',
               index_col=[0,1], header=[0]) 



######################
#### % DWELLINGS %####
######################

#### Arrange stock of dwellings - intervals

# (4) Number of dwellings [stock_dwellings_mun]:
stock_dwellings_permun  = pd.DataFrame(dw_BO0104AB.groupby(["region","year"])["Number of dwellings"].sum()) #table of number of dwellings
pop_dw = stock_dwellings_permun.merge(pop_BE0101N1, on =["region","year"], how ="left") #merge population
pop_dw["household size"] = pop_dw["Population"]/ pop_dw["Number of dwellings"] #household size 

# Joint left the time intervals
dwellings = dw_BO0104AB.copy() # Dwellings Statistics Sweden Number of dwellings by region, type of building, period of construction and year 
dwellings = dwellings.reset_index(level=["region"])
dwellings = dwellings.merge(dwelling_types, on = "type of building", how = "right") # we want to add the category "other" to multi-dwelling
# sum apartments and other buildings: 
dwellings = dwellings.groupby(by=["region","year","type of dwelling","period of construction"]).sum()

# sum by type of municipality, interval
# Divide stock of 10 year-intervals in 1-year intervals mirroring the construction trends
# From 2011 to 2020 we can see what is the difference between the data in consecutive years and period of construction
# Choose latest year of dwelling stock in dw_BO0104AB: 2020
dwellings = dwellings.reset_index(level=["year"])
dwellings_2020 = dwellings[dwellings["year"] == 2020]
dwellings_2020 = dwellings_2020.drop(["year"],axis=1)
dwellings_2020 = dwellings_2020.rename({"Number of dwellings":"stock int"}, 
                                                           axis = "columns")
# Calculate total stock of dwellings
constr_dwellings = constr_dwellings.reset_index(level=["region"])
constr_dwellings_int = constr_dwellings.merge(year_intervals, on = "year", how = "right") #we add the year intervals to the yearly construction table
constr_dwellings_int = constr_dwellings_int.reset_index(level=["year"])
constr_dwellings_int = constr_dwellings_int.merge(dwelling_types, on = "type of building", how = "right")
constr_dwellings_int = constr_dwellings_int.drop(["type of building"], axis=1)
constr_dwellings_int_sum = constr_dwellings_int.groupby(by=["region","period of construction","type of dwelling"]).sum() #the sum of the constructed buildings in the 10-year intervals
constr_dwellings_int_sum = constr_dwellings_int_sum.rename({"Completed dwellings in newly constructed buildings":"sum new dwellings int"}, 
                                                           axis = "columns")
constr_dwellings_int_sum = constr_dwellings_int_sum.drop(["year"],axis=1)
constr_dwellings_int_sum = constr_dwellings_int_sum.reset_index(level=["region","period of construction","type of dwelling"])
constr_dwellings_int = constr_dwellings_int.merge(constr_dwellings_int_sum, on = ["region","period of construction","type of dwelling"]) #we merge the table with the construction in 10-year intervals to the original one
constr_dwellings_int["perc_new_build_int"] = constr_dwellings_int["Completed dwellings in newly constructed buildings"]/constr_dwellings_int["sum new dwellings int"] #share of the new construction in a decade

stock_dwellings = constr_dwellings_int.merge(dwellings_2020, on =["region","period of construction","type of dwelling"]) # we merge the calculated table with the demographic pyramid of dwellings in 2020 to disaggregate it

stock_dwellings["current stock"] = stock_dwellings["perc_new_build_int"] * stock_dwellings["stock int"]

for i in list(range(stock_dwellings.shape[0])):
    if stock_dwellings["period of construction"].iloc[i] == "1971-1980": #the period for which we do not have all construction data so we have to apply 2 different calculations
        stock_dwellings["current stock"].iloc[i] = stock_dwellings["current stock"].iloc[i]*0.6 #here we assume that the period 1971-1980 and the period 1975-1980 have the same average yearly proportion of new construction
    
stock_dwellings["surv_rate"] = stock_dwellings["current stock"] / stock_dwellings["Completed dwellings in newly constructed buildings"] #this is only to check how many of the dwellings that were constructed are still in place (it does not work in some cases: existing buildings (2020) >> constructed buildings)

# create vertical expansion of the database for the older cohorts 
# year: (-1930) and 1931-1974
# region: municipality types (10) in municipality_list = np.array(pd.Categorical(mun_group["region"]).categories)
# type of dwelling: flats and houses

years_rec_dwe = np.array(list(range(1974-1931+1)))+1931 #all years from 1931 to 1974
municipality_list = np.array(pd.Categorical(mun_group.loc[:,"mun_group"]).categories) # list of municipalities
dw_type= np.array(["flats","houses"]) #types of dwelling

# create dataframe with all combinations of years, municipality types and dwelling types
ext_stock_dwellings = pd.DataFrame(list(product(years_rec_dwe, municipality_list,dw_type)), columns=['year', 'region', 'type of dwelling'])
# add the year intervals with a join
ext_stock_dwellings = ext_stock_dwellings.merge(year_intervals, on ="year")
ext_stock_dwellings = ext_stock_dwellings.set_index(["region","type of dwelling","period of construction"])
# merge the stock from dwellings_2020 (Number of dwellings)
ext_stock_dwellings = ext_stock_dwellings.merge(dwellings_2020, on =["region","type of dwelling","period of construction"])

#% calculate column current stock: it is the same for all years in the inter-val and equal to 10% the stock in the decade
ext_stock_dwellings.loc[:,"current stock"] = ext_stock_dwellings.loc[:,"stock int"]*0.1 
ext_stock_dwellings = ext_stock_dwellings.reset_index(level=["region","type of dwelling","period of construction"])
stock_dwellings = pd.concat([stock_dwellings,ext_stock_dwellings]) #we add the older cohorts (1931-1974) in the general table with the newer cohorts (1975-2020)
        
#append the open-ended (this is not divided by yearly cohort and is not affected by the survival curve)
dwellings_2020_ind1 = dwellings_2020.reset_index(level=["period of construction","type of dwelling","region"])
dwellings_2020_ind = dwellings_2020_ind1[dwellings_2020_ind1["period of construction"]=="-1930"]
dwellings_2020_ind["year"] = 1930 #we should remember that this is an open-ended interval!
# we include the data missing in this open ended cohort
dwellings_2020_ind_dm = dwellings_2020_ind1[dwellings_2020_ind1["period of construction"]=="data missing"]
dwellings_2020_ind = dwellings_2020_ind.merge(dwellings_2020_ind1, on =["period of construction","type of dwelling","region"], how ="left")
dwellings_2020_ind["current stock"] = dwellings_2020_ind["stock int_x"]+dwellings_2020_ind["stock int_y"]
stock_dwellings = pd.concat([stock_dwellings,dwellings_2020_ind])

#################
##### CARS ######
#################

# Number of cars per municipality type. The data is currently divided by municipality. 
number_cars_mun = number_cars_mun.merge(mun_group, how = "right", left_on = "region",right_on ="municipality") #merge classification of municipalities
number_cars_munty =  number_cars_mun.groupby(by =["year","mun_group"]).sum("Vehicles in use") #aggregate number of cars by municipality type
number_cars_munty = number_cars_munty.reset_index("year")
number_cars_munty_2020 = number_cars_munty.loc[number_cars_munty.loc[:,"year"]==2020]

#########################################
#### %% for loop municipality type ####
#########################################
scenario_list = ["base","social change","electricity mix","non-electrification","lifetime"]
municipality_list = np.array(pd.Categorical(mun_group["mun_group"]).categories)
baseScenarioTypes = ["current density","current values", "slow densification","accelerated densification"]
mun_types_scenario = np.array(["SKL02 suburban municipalities (38 municipalities)",
                               "SKL03 large cities (31 municipalities)",
                               "SKL04 suburban municipalities to large cities (22 municipalities)",
                               "SKL05 commuter municipalities (51 municipalities)",
                               "SKL07 manufacturing municipalities (54 municipalities)",
                               "SKL09 municipalities in densely populated regions (35 municipalities)"])

time_fin_hist_pop = 2021
time_horizon = 2100
    
timesteps = np.arange(time_fin_hist_pop + 1, time_horizon + 1)

# for loop to run the model to all scenario types and municipality types
for scenario_on in scenario_list:
    #each sensitivity scenario involves assumptions on electricity mix, electrification of the fleet, lifetime and other social assumptions
    scenario_elmix = scenario_characteristics.loc[scenario_on,"scenario_elmix"].values.item()
    scenario_el = scenario_characteristics.loc[scenario_on,"scenario_el"].values.item()
    scenario_life = scenario_characteristics.loc[scenario_on,"scenario_lifetime"].values.item()
    scenario_social = scenario_characteristics.loc[scenario_on,"scenario_extrasoc"].values.item() 

    for scenario_basic in baseScenarioTypes:
       # for scenario_elmix in scenarioelmixtypes:
            print("ON SOM: ", scenario_on, " amb ", scenario_basic)
            print("-----", datetime.now())

            for mun_type_on in municipality_list:
                print("-----", mun_type_on)
                if mun_type_on in mun_types_scenario: #depending on the municipality, the conditions will not be changed because the urban form is already dense
                    scenario_basic2 = scenario_basic
                    scenario_social2 = scenario_social
                    scenario_elmix2 = scenario_elmix
                    scenario_el2 = scenario_el
                    scenario_life2 = scenario_life
                else:
                    scenario_basic2 = "scenario_bas" #only the basic scenario related to densification is not included in some types of municipality (metropolitan and rural)

                stock_dwellings_mun = stock_dwellings[stock_dwellings["region"]==mun_type_on] #select rows of the given municipality
                fut_hhsize_sc = fut_hhsize.loc[:,scenario_social] #scenario_social2
                fut_hhsize_sc.index.name="year"
                pop_dw_mun = pop_dw.loc[mun_type_on] #select rows of the given municipality
                type_new_buildings_sc = float(type_new_buildings_tot.loc[(scenario_basic2,mun_type_on)])
                area_characteristics_sc = area_characteristics.loc[:,scenario_social]
                area_characteristics_mun = area_characteristics_sc.loc[(slice(None),mun_type_on),:]
                lifetime_dwellings_sc = lifetime_dwellings.loc[(scenario_life,scenario_basic2),:]
                share_powertrain_new = share_powertrain_new_raw.loc[:,scenario_el]
                number_cars_mun_2020 = number_cars_munty_2020.loc[mun_type_on] 
                size_cars_new_sc = size_cars_new.loc[:,scenario_social]
                emi_int_indir_fuel_use_sc = emi_int_indir_fuel_use.loc[scenario_elmix,:]
            
######################################
###### %% Population-dwellings #######
######################################
    
                time_fin_hist_pop = 2021

                # Expand population dataframe (pop_dw) rows to time_horizon (2050)
                timesteps = np.arange(time_fin_hist_pop, time_horizon + 1)
                timesteps1 = np.arange(time_fin_hist_pop+1, time_horizon + 1)
                ext_pop_dw = pd.DataFrame(data=timesteps1, columns = ["year"])
                ext_pop_dw = ext_pop_dw.set_index("year")
                ext_pop_dw["region"] = 0
                ext_pop_dw["Population"] = 0
                ext_pop_dw["Number of dwellings"] = 0
                ext_pop_dw["household size"] = 0

                #append these new empty table to pop_dw_mun
                pop_dw_mun = pd.concat([pop_dw_mun,ext_pop_dw])
                pop_dw_mun = pop_dw_mun.sort_values(by="year")
                
                # Join population growth and household size trend
                # given as increase with reference to 2020
                pop_dw_mun = pop_dw_mun.merge(fut_pop, how ="left", on = "year")
                pop_dw_mun = pop_dw_mun.merge(fut_hhsize_sc, how ="left", on = "year")

                # Multiply population growth by reference year population and household size
                for ryear in timesteps:
                    pop_dw_mun.loc[ryear,"Population"] = pop_dw_mun.loc[2020,"Population"]* pop_dw_mun.loc[ryear,"baseline_proj_pop"]
                    pop_dw_mun.loc[ryear,"household size"] = pop_dw_mun.loc[2020,"household size"]* pop_dw_mun.loc[ryear,"hh_size_proj"]
                    pop_dw_mun.loc[ryear,"Number of dwellings"] = pop_dw_mun.loc[ryear,"Population"] / pop_dw_mun.loc[ryear,"household size"]

###################################################
#### %% dwelling model v1.2 int-MFA and area ######
###################################################

                time_fin_hist_dw = 2020
                time_start_dw = 2020
                time_max = time_horizon - time_fin_hist_dw + 1 #2020-2050
                time_max_dat = time_horizon - 2010 + 1 # = time_horizon - time_start
                cmaxold = 1950 #limit of houses that are considered denser and more historical, which are not included for early demolishing in the accelerated densification
                time_first_cohort_dw =1930

                timesteps = np.arange(0, time_max)
            
                # Data frames of dwellings from "population-dwellings v1.0": pop_dw_mun. This includes both the past and future stocks
                #stock_dwellings 
                
                # from the stock_dwellings_mun, we select the "current stock" column and transpose it to start building the 2 cohort matrixes: houses and apartments
                stock_dwellings_mun = stock_dwellings_mun.set_index(keys=["year","region","type of dwelling"])
                stock_houses_coh = stock_dwellings_mun.loc[(slice(None),slice(None),"houses")]
                stock_houses_coh_pivot = stock_houses_coh.pivot_table("current stock", index = "region", columns = "year")
                stock_houses_coh_pivot["time"] = 2020 # add column to indicate the year of data
                stock_houses_coh_pivot = stock_houses_coh_pivot.set_index("time")

                stock_flats_coh = stock_dwellings_mun.loc[(slice(None),slice(None),"flats")]
                stock_flats_coh_pivot = stock_flats_coh.pivot_table("current stock", index= "region" , columns = "year")
                stock_flats_coh_pivot["time"] = 2020  # add column to indicate the year of data
                stock_flats_coh_pivot = stock_flats_coh_pivot.set_index("time") 
                
                perc_dwellings_hh = pop_dw_mun.loc[2020,"Number of dwellings"]/(stock_houses_coh_pivot.loc[2020].sum() + stock_flats_coh_pivot.loc[2020].sum()) #there are more dwellings than households

                # Normal distribution survival curve 
                time_surv = 500
                timesteps_surv = np.arange(0, time_surv) # the survival curves must be very long!

                curve_stddev_newho = lifetime_dwellings_sc.loc["new houses","stddev"]
                curve_mean_newho = lifetime_dwellings_sc.loc["new houses","mean"]
                curve_surv_newho = scipy.stats.norm.sf(timesteps_surv, loc = curve_mean_newho, scale = curve_stddev_newho)

                curve_stddev_oldho = lifetime_dwellings_sc.loc["new houses","stddev"] 
                curve_mean_oldho = lifetime_dwellings_sc.loc["old houses","mean"]
                curve_surv_oldho = scipy.stats.norm.sf(timesteps_surv, loc = curve_mean_oldho, scale = curve_stddev_oldho)

                curve_stddev_ap = lifetime_dwellings_sc.loc["apartments","stddev"]
                curve_mean_ap = lifetime_dwellings_sc.loc["apartments","mean"]
                curve_surv_ap = scipy.stats.norm.sf(timesteps_surv, loc = curve_mean_ap, scale = curve_stddev_ap)

                #Initialize inter-age survival curve
                curve_surv_ap_iage = np.ones(shape=[time_surv,1])
                curve_surv_ap_iage  = np.squeeze(curve_surv_ap_iage ) #somehow it does not work otherwise
                curve_surv_newho_iage = np.ones(shape=[time_surv,1])
                curve_surv_newho_iage  = np.squeeze(curve_surv_newho_iage ) 
                curve_surv_oldho_iage = np.ones(shape=[time_surv,1])
                curve_surv_oldho_iage  = np.squeeze(curve_surv_oldho_iage ) 
                
                # Survival curve previous year stock
                for time in timesteps_surv:
                    if time > 0:
                        if curve_surv_newho[time-1] > 0:
                            curve_surv_newho_iage[time] = curve_surv_newho[time]/curve_surv_newho[time-1] 
                        else:
                            curve_surv_newho_iage[time] = 0
                        if curve_surv_oldho[time-1] > 0:
                            curve_surv_oldho_iage[time] = curve_surv_oldho[time]/curve_surv_oldho[time-1] 
                        else:
                            curve_surv_oldho_iage[time] = 0
                        if  curve_surv_ap[time-1] > 0:
                            curve_surv_ap_iage[time] = curve_surv_ap[time]/curve_surv_ap[time-1] 
                        else:
                            curve_surv_ap_iage[time] = 0
                    else:
                        curve_surv_newho_iage[time] = 1 #the first is 1
                        curve_surv_oldho_iage[time] = 1 #the first is 1
                        curve_surv_ap_iage[time] = 1
                                            
                # create expansion of cohort matrix for future years (vertical)
                indexes = np.zeros(shape=[time_horizon - time_fin_hist_dw,1])
                r = 0 
                while r < time_max-1:
                    indexes[r] = r + time_fin_hist_dw + 1
                    r = r + 1

                indexes = np.squeeze(indexes)    
            
                #create table with zeros full timespan
                stock_temp_exp = np.zeros(shape = [time_horizon - time_fin_hist_dw, stock_houses_coh_pivot.shape[1]])
                stock_temp_exp = pd.DataFrame(data = stock_temp_exp, index = indexes, columns = stock_houses_coh_pivot.columns)

                stock_houses_coh_pivot = pd.concat([stock_houses_coh_pivot,stock_temp_exp]) #add this table with zeros to the original one
                stock_flats_coh_pivot = pd.concat([stock_flats_coh_pivot,stock_temp_exp])

                # create dataframe outflows
                d = {"year":[2020],"houses":[0],"flats":[0]}
                outflows_dwellings = pd.DataFrame(data = indexes,columns = ["year"])
                outflows_dwellings = outflows_dwellings.set_index("year")
                outflows_dwellings["houses"]= 0
                outflows_dwellings["flats"] = 0
                outflows_dwellings["FA_houses"]=0
                outflows_dwellings["FA_flats"]=0
                outflows_dwellings["LU_houses"]=0
                outflows_dwellings["LU_flats"]=0

                # create dataframe inflows
                d = {"year":[2020],"houses":[0],"flats":[0]}
                inflows_dwellings = pd.DataFrame(data = indexes,columns = ["year"])
                inflows_dwellings = inflows_dwellings.set_index("year")
                inflows_dwellings["houses"]= 0
                inflows_dwellings["flats"] = 0
                inflows_dwellings["FA_houses"]=0
                inflows_dwellings["FA_flats"]=0
                inflows_dwellings["LU_houses"]=0
                inflows_dwellings["LU_flats"]=0

                # add new columns to pop_dw_mun to calculate urban density afterwards
                pop_dw_mun["Floor Area"] = 0
                pop_dw_mun["Land Use"] = 0
                pop_dw_mun["construction houses"] = 0
                pop_dw_mun["construction flats"] = 0
                pop_dw_mun["demolition houses"] = 0
                pop_dw_mun["demolition flats"] = 0

                # create expansion of cohort matrix for future years (horizontal)
                columns = np.zeros(shape=[time_horizon - time_fin_hist_dw,1])
                r = 0 
                while r < time_max-1:
                    columns[r] = r + time_fin_hist_dw + 1
                    r = r + 1
                columns = np.squeeze(columns) 

                #add table with years until time_horizon - horizontally (left join)
                stock_temp_exp = np.zeros(shape=[stock_houses_coh_pivot.shape[0], time_horizon - time_fin_hist_dw])
                stock_temp_exp = pd.DataFrame(data = stock_temp_exp, index = stock_houses_coh_pivot.index, columns = columns)

                stock_houses_coh_pivot = stock_houses_coh_pivot.join(stock_temp_exp)
                stock_flats_coh_pivot = stock_flats_coh_pivot.join(stock_temp_exp)

                # initialization of FA and LU cohort 
                FA_stock_houses_coh_pivot = stock_houses_coh_pivot.copy()
                FA_stock_flats_coh_pivot  = stock_flats_coh_pivot.copy()
                
                LU_stock_houses_coh_pivot  = stock_houses_coh_pivot.copy()
                LU_stock_flats_coh_pivot  = stock_flats_coh_pivot.copy()
                                
                # Coordinated apartment-houses stock-driven MFA
                # Includes calculation of Floor Area and Land Use
                #r = 0
                #ryear = r + time_start_dw
                #c = 0
                #cyear = c + time_first_cohort_dw 
                time_first_cohort_dw = stock_houses_coh_pivot.columns[0] 
                stock_houses_coh_pivot2 = stock_houses_coh_pivot

                for ryear in stock_houses_coh_pivot.index:
                    constr_cap_mun = constr_cap.loc[mun_type_on, str(int(ryear))]
                    if ryear == time_start_dw:
                        cyear = time_first_cohort_dw
                        while cyear <= time_start_dw:
                            stock_houses_coh_pivot.loc[ryear, cyear] = math.trunc(stock_houses_coh_pivot.loc[ryear, cyear]* perc_dwellings_hh)
                            stock_flats_coh_pivot.loc[ryear, cyear] = math.trunc(stock_flats_coh_pivot.loc[ryear, cyear]* perc_dwellings_hh)
                            cyear = cyear + 1
                        pop_dw_mun.loc[ryear,"Number of dwellings - 2"] = stock_houses_coh_pivot.loc[ryear].sum() + stock_flats_coh_pivot.loc[ryear].sum()
                    elif ryear > time_start_dw:
                        a = 1
                        cyear = time_first_cohort_dw
                        while cyear < ryear+1: #change to if (at the end of the loop: if a <1, cyear=cyear, if a=1, cyear = cyear+1)
                            if cyear == time_first_cohort_dw: 
                                outflows_flats = 0 #initialize outflow variable
                                outflows_houses = 0
                                stock_houses_coh_pivot.loc[ryear, cyear] = stock_houses_coh_pivot.loc[ryear - 1, cyear] #in the first cohort, which is the open-ended one, there is no survival curve, the buildings are conserved for historical reasons
                                stock_flats_coh_pivot.loc[ryear, cyear] = stock_flats_coh_pivot.loc[ryear - 1, cyear]
                                cyear = cyear + 1 
                            elif (ryear > cyear ) and (cyear > time_first_cohort_dw):
                                if cyear > cmaxold: #older houses are not considered for early demolishing in accelerated densification
                                    sf_ia_ho = curve_surv_newho_iage[int(ryear) - int(cyear)]
                                else:
                                    sf_ia_ho = curve_surv_oldho_iage[int(ryear) - int(cyear)]
                                prev_year_stock_houses = stock_houses_coh_pivot.loc[ryear - 1, cyear]
                                prev_year_stock_flats = stock_flats_coh_pivot.loc[ryear - 1, cyear]
                                remaining_stock_houses = math.trunc(prev_year_stock_houses * sf_ia_ho)
                                sf_ia_fl = curve_surv_ap_iage[int(ryear) - int(cyear)]
                                remaining_stock_flats = math.trunc(prev_year_stock_flats * sf_ia_fl)
                                outflow_cohort_houses = math.trunc(prev_year_stock_houses - remaining_stock_houses)
                                outflow_cohort_flats =  math.trunc(prev_year_stock_flats- remaining_stock_flats)
                                stock_houses_coh_pivot.loc[ryear, cyear] = remaining_stock_houses
                                stock_flats_coh_pivot.loc[ryear, cyear] = remaining_stock_flats     
                                outflows_flats = outflows_flats + outflow_cohort_flats #-1 because the outflow table starts 1 year afterwards
                                outflows_houses = outflows_houses + outflow_cohort_houses
                                cyear = cyear + 1 
                            elif ryear == cyear: # introduction of new cohorts considering the demand and the remaining stocks from older cohorts
                                req_new_dwellings =  pop_dw_mun.loc[ryear,"Number of dwellings"] - stock_houses_coh_pivot.loc[ryear].sum() - stock_flats_coh_pivot.loc[ryear].sum()
                                if req_new_dwellings > 0:
                                    if req_new_dwellings > constr_cap_mun: #if we are constructing more than what the system can produce, we must extend the life of the stock and limit demolishing and construction in that year
                                        min_remaining_stock = pop_dw_mun.loc[ryear,"Number of dwellings"] - constr_cap_mun
                                        final_outflow = pop_dw_mun.loc[ryear-1,"Number of dwellings - 2"] - min_remaining_stock
                                        max_stock = pop_dw_mun.loc[ryear-1,"Number of dwellings - 2"] + constr_cap_mun
                                        if max_stock > pop_dw_mun.loc[ryear,"Number of dwellings"]:
                                            a = (final_outflow)/(outflows_flats + outflows_houses) # coefficient to reduce the outflow to the max level
                                            pop_dw_mun.loc[ryear,"a"] = a
                                            for cyear in stock_houses_coh_pivot.columns: #change to if (at the end of the loop: if a <1, cyear=cyear, if a=1, cyear = cyear+1)
                                                if cyear == time_first_cohort_dw: 
                                                     outflows_flats = 0 #initialize outflow variable
                                                     outflows_houses = 0
                                                     stock_houses_coh_pivot.loc[ryear, cyear] = stock_houses_coh_pivot.loc[ryear - 1, cyear] #in the first cohort, which is the open-ended one, there is no survival curve, the buildings are conserved for historical reasons
                                                     stock_flats_coh_pivot.loc[ryear, cyear] = stock_flats_coh_pivot.loc[ryear - 1, cyear]
                                                elif (ryear > cyear) and (cyear > time_first_cohort_dw):
                                                     if cyear > cmaxold: #older houses are not considered for early demolishing in accelerated densification
                                                         sf_ia_ho = curve_surv_newho_iage[int(ryear) - int(cyear)]
                                                     else:
                                                         sf_ia_ho = curve_surv_oldho_iage[int(ryear) - int(cyear)]
                                                     prev_year_stock_houses = stock_houses_coh_pivot.loc[ryear - 1, cyear]
                                                     prev_year_stock_flats = stock_flats_coh_pivot.loc[ryear - 1, cyear]
                                                     remaining_stock_houses = math.trunc(prev_year_stock_houses * sf_ia_ho)
                                                     sf_ia_fl = curve_surv_ap_iage[int(ryear) - int(cyear)]
                                                     remaining_stock_flats = math.trunc(prev_year_stock_flats * sf_ia_fl)
                                                     outflow_cohort_houses = math.trunc((prev_year_stock_houses - remaining_stock_houses)*a) # the outflow
                                                     outflow_cohort_flats =  math.trunc((prev_year_stock_flats- remaining_stock_flats)*a) # a is generally 1, when the calculated outflow with the survival curve is too high, we decrease it in the following lines with the coefficient a
                                                     if math.trunc(prev_year_stock_houses - outflow_cohort_houses)>= 0:
                                                         remaining_stock_houses =  math.trunc(prev_year_stock_houses - outflow_cohort_houses)
                                                     elif math.trunc(prev_year_stock_houses - outflow_cohort_houses)< 0:
                                                         remaining_stock_houses = 0
                                                     if  math.trunc(prev_year_stock_flats - outflow_cohort_flats)>=0:
                                                        remaining_stock_flats =  math.trunc(prev_year_stock_flats - outflow_cohort_flats)
                                                     elif  math.trunc(prev_year_stock_flats - outflow_cohort_flats)<0:
                                                         remaining_stock_flats = 0
                                                     stock_houses_coh_pivot.loc[ryear, cyear] = remaining_stock_houses
                                                     stock_flats_coh_pivot.loc[ryear, cyear] = remaining_stock_flats     
                                                     outflows_flats = outflows_flats + outflow_cohort_flats #-1 because the outflow table starts 1 year afterwards
                                                     outflows_houses = outflows_houses + outflow_cohort_houses
                                                elif ryear == cyear:
                                                    inflows_flats =  math.trunc(constr_cap_mun * type_new_buildings_sc) #we build as much as we can
                                                    inflows_houses =  constr_cap_mun - inflows_flats
                                                    stock_flats_coh_pivot.loc[ryear, cyear] = inflows_flats
                                                    stock_houses_coh_pivot.loc[ryear, cyear] = inflows_houses
                                                    pop_dw_mun.loc[ryear,"construction houses"] = inflows_houses
                                                    pop_dw_mun.loc[ryear,"construction flats"] = inflows_flats
                                                    pop_dw_mun.loc[ryear,"demolition houses"] = outflows_houses
                                                    pop_dw_mun.loc[ryear,"demolition flats"] = outflows_flats
                                                    outflows_dwellings.loc[ryear, "houses"] = outflows_houses
                                                    outflows_dwellings.loc[ryear, "flats"] = outflows_flats
                                        elif max_stock <= pop_dw_mun.loc[ryear,"Number of dwellings"]: 
                                            #in this case, the required new dwellings are larger than the cap 
                                            #but even if we did not demolish anything we won't reach the target of dwellings. 
                                            #Therefore, we don't demolish and we build as much as we can and that's it
                                            for cyear in stock_houses_coh_pivot.columns:
                                               prev_year_stock_houses = stock_houses_coh_pivot.loc[ryear - 1, cyear]
                                               prev_year_stock_flats = stock_flats_coh_pivot.loc[ryear - 1, cyear] 
                                               outflow_houses = 0
                                               outflow_flats = 0
                                               remaining_stock_houses = prev_year_stock_houses
                                               remaining_stock_flats = prev_year_stock_flats # we don't demolish anything
                                               stock_houses_coh_pivot.loc[ryear, cyear] = remaining_stock_houses
                                               stock_flats_coh_pivot.loc[ryear, cyear] = remaining_stock_flats 
                                            cyear = ryear
                                            inflows_flats =  math.trunc(constr_cap_mun * type_new_buildings_sc)
                                            inflows_houses =  constr_cap_mun - inflows_flats
                                            stock_flats_coh_pivot.loc[ryear, cyear] = inflows_flats
                                            stock_houses_coh_pivot.loc[ryear, cyear] = inflows_houses
                                            inflows_dwellings.loc[ryear, "houses"] = inflows_houses
                                            inflows_dwellings.loc[ryear, "flats"] = inflows_flats
                                            pop_dw_mun.loc[ryear,"construction houses"] = inflows_houses
                                            pop_dw_mun.loc[ryear,"construction flats"] = inflows_flats
                                            outflows_dwellings.loc[ryear, "houses"] = outflows_houses
                                            outflows_dwellings.loc[ryear, "flats"] = outflows_flats
                                            pop_dw_mun.loc[ryear,"demolition houses"] = outflows_houses
                                            pop_dw_mun.loc[ryear,"demolition flats"] = outflows_flats
                                            cyear = cyear + 1 
                                            a = 1
                                            pop_dw_mun.loc[ryear,"a"] = 0
                                    elif req_new_dwellings <= constr_cap_mun:
                                        pop_dw_mun.loc[ryear,"a"] = a
                                        a = 1
                                        inflows_flats =  math.trunc(req_new_dwellings * type_new_buildings_sc)
                                        inflows_houses =  math.trunc(req_new_dwellings - inflows_flats)
                                        stock_flats_coh_pivot.loc[ryear, cyear] = inflows_flats
                                        stock_houses_coh_pivot.loc[ryear, cyear] = inflows_houses
                                        inflows_dwellings.loc[ryear, "houses"] = inflows_houses
                                        inflows_dwellings.loc[ryear, "flats"] = inflows_flats
                                        pop_dw_mun.loc[ryear,"construction houses"] = inflows_houses
                                        pop_dw_mun.loc[ryear,"construction flats"] = inflows_flats
                                        outflows_dwellings.loc[ryear, "houses"] = outflows_houses
                                        outflows_dwellings.loc[ryear, "flats"] = outflows_flats
                                        pop_dw_mun.loc[ryear,"demolition houses"] = outflows_houses
                                        pop_dw_mun.loc[ryear,"demolition flats"] = outflows_flats
                                        cyear = cyear + 1 
                                elif req_new_dwellings < 0: #if we already have enough dwellings: we don't build new ones
                                    #print("4")
                                    inflows_flats = 0
                                    inflows_houses = 0
                                    inflows_dwellings.loc[ryear, "houses"] = inflows_houses
                                    inflows_dwellings.loc[ryear, "flats"] = inflows_flats
                                    pop_dw_mun.loc[ryear,"construction houses"] = inflows_houses
                                    pop_dw_mun.loc[ryear,"construction flats"] = inflows_flats
                                    outflows_dwellings.loc[ryear, "houses"] = outflows_houses
                                    outflows_dwellings.loc[ryear, "flats"] = outflows_flats
                                    pop_dw_mun.loc[ryear,"demolition houses"] = outflows_houses
                                    pop_dw_mun.loc[ryear,"demolition flats"] = outflows_flats
                                    cyear = cyear + 1
                                    pop_dw_mun.loc[ryear,"a"] = a
                                total_stock_houses = stock_houses_coh_pivot.loc[ryear].sum()
                                pop_dw_mun.loc[ryear,"Number of houses - 2"] = total_stock_houses
                                total_stock_flats = stock_flats_coh_pivot.loc[ryear].sum()
                                pop_dw_mun.loc[ryear,"Number of flats - 2"] = total_stock_flats
                                pop_dw_mun.loc[ryear,"Number of dwellings - 2"] = total_stock_flats + total_stock_houses


                # copy table for avoiding problems
                stock_houses_coh_pivot_copy = stock_houses_coh_pivot.copy()
                stock_flats_coh_pivot_copy = stock_flats_coh_pivot.copy()
 
                #calculate Floor Area
                av_area_houses = float(area_characteristics_mun.loc[("pre1950",slice(None), "one- or two-dwelling buildings"),"dwelling_size"])
                av_area_flats = float(area_characteristics_mun.loc[("pre1950",slice(None), "multi-dwelling buildings"),"dwelling_size"])
                for ryear in FA_stock_houses_coh_pivot.index:
                    for cyear in LU_stock_houses_coh_pivot.columns:
                        stock_houses = stock_houses_coh_pivot_copy.loc[ryear,cyear] 
                        stock_flats = stock_flats_coh_pivot_copy.loc[ryear,cyear]
                        fa_houses_temp = stock_houses * av_area_houses
                        fa_flats_temp = stock_flats * av_area_flats
                        FA_stock_houses_coh_pivot.loc[ryear, cyear] = fa_houses_temp
                        FA_stock_flats_coh_pivot.loc[ryear, cyear] = fa_flats_temp
                        if cyear == ryear:
                            pop_dw_mun.loc[ryear,"new Floor Area - houses"] = fa_houses_temp
                            pop_dw_mun.loc[ryear,"new Floor Area - flats"] = fa_flats_temp
                # when the round to the c finishes, we calculate the total FA of the stock   
                    FA_houses_r = FA_stock_houses_coh_pivot.loc[ryear].sum()
                    FA_flats_r = FA_stock_flats_coh_pivot.loc[ryear].sum()
                    pop_dw_mun.loc[ryear,"Floor Area - houses"] = FA_houses_r 
                    pop_dw_mun.loc[ryear,"Floor Area - flats"] = FA_flats_r 
                    pop_dw_mun.loc[ryear,"Floor Area"] = FA_houses_r + FA_flats_r       
        
                # copy table for avoiding problems
                FA_stock_houses_coh_pivot_copy = FA_stock_houses_coh_pivot.copy()
                FA_stock_flats_coh_pivot_copy = FA_stock_flats_coh_pivot.copy()
        
                #calculate Land Use
                av_FSI_houses = 0
                av_FSI_flats = 0
                for ryear in LU_stock_houses_coh_pivot.index:
                    for cyear in LU_stock_houses_coh_pivot.columns:
                        if cyear < 1950: #depending on the cohort we have a different FSI ratio
                            av_FSI_houses = float(area_characteristics_mun.loc[("pre1950",slice(None),"one- or two-dwelling buildings"),"FSI"])
                            av_FSI_flats = float(area_characteristics_mun.loc[("pre1950",slice(None),"multi-dwelling buildings"),"FSI"])
                        elif cyear <= 2020:
                            av_FSI_houses = float(area_characteristics_mun.loc[("post1950",slice(None),"one- or two-dwelling buildings"),"FSI"])
                            av_FSI_flats = float(area_characteristics_mun.loc[("post1950",slice(None),"multi-dwelling buildings"),"FSI"])
                        else:
                            av_FSI_houses = float(area_characteristics_mun.loc[("future",slice(None),"one- or two-dwelling buildings"),"FSI"])
                            av_FSI_flats = float(area_characteristics_mun.loc[("future",slice(None),"multi-dwelling buildings"),"FSI"])
                        FA_houses = FA_stock_houses_coh_pivot_copy.loc[ryear, cyear]
                        FA_flats = FA_stock_flats_coh_pivot_copy.loc[ryear, cyear]
                        LU_stock_houses_coh_pivot.loc[ryear, cyear] = FA_houses / av_FSI_houses
                        LU_stock_flats_coh_pivot.loc[ryear, cyear] = FA_flats / av_FSI_flats
                    # when the round to the c finishes, we calculate the total FA of the stock   
                    LU_houses_r = LU_stock_houses_coh_pivot.loc[ryear].sum()
                    LU_flats_r = LU_stock_flats_coh_pivot.loc[ryear].sum()
                    pop_dw_mun.loc[ryear,"Land Use - houses"] = LU_houses_r
                    pop_dw_mun.loc[ryear,"Land Use - flats"] = LU_flats_r
                    pop_dw_mun.loc[ryear,"Land Use"] = LU_houses_r + LU_flats_r
        
                #### % AREA AND DENSITY % ####
        
                # calculate urban density
                pop_dw_mun["residential density"] = pop_dw_mun["Population"] / pop_dw_mun["Land Use"]
                # calculation of diameter does not work. It could be that math.sqrt can't be applied to a series
                pop_dw_mun["area per capita"] = pop_dw_mun["Floor Area"] / pop_dw_mun["Population"]

                #### % CAR OWNERSHIP AND DISTANCE % ####
                ### apply the equations for car ownership and distance
                a_own = cars_per_capita_equation["value"].loc["a"]
                b_own = cars_per_capita_equation["value"].loc["b"]
                pop_dw_mun["car ownership per capita"] = a_own * np.log(pop_dw_mun["residential density"]) + b_own
                pop_dw_mun["cars"] = pop_dw_mun["car ownership per capita"]*pop_dw_mun["Population"]
                a_dist = distance_per_capita_equation["value"].loc["a"]
                b_dist = distance_per_capita_equation["value"].loc["b"]
                c_dist = distance_per_capita_equation["value"].loc["c"]
                pop_dw_mun["distance paskm per capita"] = a_dist*(pop_dw_mun["residential density"])*(pop_dw_mun["residential density"]) + b_dist*(pop_dw_mun["residential density"]) + c_dist
                pop_dw_mun["distance paskm"] = pop_dw_mun["Population"] * pop_dw_mun["distance paskm per capita"]*365
                for year in pop_dw_mun.index:
                    pop_dw_mun.loc[year,"distance vehkm"] = pop_dw_mun.loc[year,"distance paskm"] / occupancy_rate_all.loc[scenario_social,str(year)]
                pop_dw_mun["distance vehkm per car"] = pop_dw_mun["distance vehkm"]/pop_dw_mun["cars"]

                #### MATERIALS DWELLINGS ####
                
                # create list of materials
                emi_int_dwellings_ni = emi_int_dwellings.reset_index()
                materials_dw = np.array(pd.Categorical(emi_int_dwellings_ni["Material"]).categories)

                # create materials table
                stock_temp_exp = np.zeros(shape=[indexes.shape[0], materials_dw.shape[0]])
                materials_houses = pd.DataFrame(data = stock_temp_exp, index = indexes, columns = materials_dw)
                materials_flats = pd.DataFrame(data = stock_temp_exp, index = indexes, columns = materials_dw)
                materials_tot = pd.DataFrame(data = stock_temp_exp, index = indexes, columns = materials_dw)
                emissions_materials_dw = pd.DataFrame(data = stock_temp_exp, index = indexes, columns = materials_dw)
        
                #material intensity new cohorts (we take the 2000 cohort)
                mat_int_houses = mat_int_dwellings.loc[("houses", 2000),:]
                mat_int_flats = mat_int_dwellings.loc[("flats", 2000),:]
            
                pop_dw_mun_copy = pop_dw_mun.copy()
        
                ### calculate inflows of materials from Floor Area (inflows_dwellings["FA_houses"]) and material intensity indicators (mat_int_houses)
                for mat in materials_dw:
                    for year in indexes:
                        if year > 2020:
                            new_area_houses_temp = pop_dw_mun_copy.loc[year,"new Floor Area - houses"]
                            new_area_flats_temp = pop_dw_mun_copy.loc[year,"new Floor Area - flats"] 
                            in_mat_houses_temp = float(mat_int_houses.loc[mat]) * new_area_houses_temp
                            in_mat_flats_temp = float(mat_int_flats.loc[mat]) * new_area_flats_temp 
                            materials_tot.loc[year, mat] = in_mat_houses_temp + in_mat_flats_temp
        
                materials_tot_cop = materials_tot.copy()
                
                for mat in materials_dw:
                    for year in indexes:
                        if year > 2020:
                            emi_int_d = float(emi_int_dwellings.loc[(slice(None),mat),"emission intensity"])
                            quant_mat = materials_tot_cop.loc[year,mat]
                            emissions_materials_dw.loc[year, mat] =  quant_mat * emi_int_d
        
                pop_dw_mun["materials dwellings"] = materials_tot_cop.copy().sum(1)        
                emissions_materials_dw_copy = emissions_materials_dw.copy()
                emissions_materials_dw_tot =  emissions_materials_dw_copy.sum(1)    
                pop_dw_mun["emissions dwelling production"] = emissions_materials_dw_tot
                pop_dw_mun["emissions dwelling prod per dwelling"] = emissions_materials_dw_tot/(pop_dw_mun["construction houses"]+pop_dw_mun["construction flats"])
        
######################
#### %% car model ####
######################

                time_fin_hist_car = 2020
                time_max_car = share_cohorts_cars_2020.shape[1] + time_horizon - time_fin_hist_car # = time_horizon - time_first_cohort
                time_first_cohort_cars = 1985
                time_max_data_c = time_horizon - 2020 +1# = time_horizon - time_start
            
                timesteps = np.arange(0, time_max_car)
            
                # %% 2. create a single survival curve

                # Normally distributed survival curve
                # Morfeldt et al 2021 defines:
                curve_mean_car = 17 #16.93 
                curve_stddev_car = 0.26 * curve_mean_car #0.26
                curve_surv_car = scipy.stats.norm.sf(timesteps, loc = curve_mean_car, scale = curve_stddev_car)

                #Initialize inter-age survival curve
                curve_surv_car_iage = np.ones(shape=[time_max_car,1])

                # Survival curve for previous year stock (not from initial stock)
                for time in timesteps:
                    if time > 0:
                        curve_surv_car_iage[time] = curve_surv_car[time]/curve_surv_car[time-1]      
                    else:
                        curve_surv_car_iage[time] = 1
                
                #  # % Reconstruct the older cohorts trajectory (it was lost in the open-ended older interval) % ##

                columns = np.arange(0, time_max_car)
                r = 0 
                while r < time_max_car:
                    ryear = r + time_first_cohort_cars 
                    columns[r] =int(ryear)
                    r = r + 1
                columns = np.squeeze(columns)

                indexes = np.arange(0, time_max_data_c)
                r = 0 
                while r < time_max_data_c:
                    ryear = r + time_fin_hist_car
                    indexes[r] = int(ryear) #this includes the year of data of cohorts to calculate the cohorts of municipality
                    r = r + 1 
                indexes = np.squeeze(indexes)   

                #create table with zeros full timespan
                stock_cars_pivot_z = np.zeros(shape=[time_horizon - time_fin_hist_car+1,  time_max_car])
                stock_cars_pivot = pd.DataFrame(data = stock_cars_pivot_z, index = indexes, columns = columns)
        
                # create dataframe outflows
                d = {"year":[2020],"houses":[0],"flats":[0]}
                outflows_cars = pd.DataFrame(data = indexes, columns = ["year"])
                outflows_cars = outflows_cars.set_index("year")
                outflows_cars["cars_out"]= 0

                #initialize pop_dw_mun["new_cars"]
                pop_dw_mun["new_cars"] = 0
                pop_dw_mun["lithium_inflow"] = 0
                pop_dw_mun["materials cars"] = 0
                powertrain_list = np.array(share_powertrain_new.columns) 
                
                # MFA system
                for ryear in indexes:
                    for cyear in columns:
                        yearly_demand_cars = pop_dw_mun.loc[ryear, "cars"]
                        if (ryear == 2020)&(cyear<=ryear):
                            perc_coh = share_cohorts_cars_2020.loc[ryear,cyear]
                            stock_cars_pivot.loc[ryear,cyear] = perc_coh * yearly_demand_cars                  
                        elif (cyear == 1985)&(ryear > 2020):
                            stock_cars_pivot.loc[ryear,cyear] = stock_cars_pivot.loc[ryear-1,cyear] #the first cohort stays the same ("historical cars")      
                        elif (cyear != 1985)&(ryear > cyear):
                            prev_year_stock_cars = stock_cars_pivot.loc[ryear-1, cyear]
                            sf_ia_cars = float(curve_surv_car_iage[ryear-cyear])
                            remaining_stock_cars = prev_year_stock_cars * sf_ia_cars
                            stock_cars_pivot.loc[ryear, cyear] = remaining_stock_cars
                            outflows_cars.loc[ryear,"cars_out"] = outflows_cars.loc[ryear,"cars_out"] + prev_year_stock_cars - remaining_stock_cars    
                        elif ryear == cyear:
                            req_new_cars = yearly_demand_cars - stock_cars_pivot.loc[ryear].sum()
                            if req_new_cars > 0:
                                stock_cars_pivot.loc[ryear,cyear] = req_new_cars
                                pop_dw_mun.loc[ryear,"new_cars"] = req_new_cars
                                for powert in powertrain_list:
                                    pop_dw_mun.loc[ryear,"materials cars"] = pop_dw_mun.loc[ryear,"materials cars"] + (size_cars_new_sc.loc[cyear,powert] * float(share_powertrain_new.loc[cyear,powert])*req_new_cars)
                                    pop_dw_mun.loc[ryear,"lithium_inflow"] = pop_dw_mun.loc[ryear,"lithium_inflow"] + (size_cars_new_sc.loc[cyear,powert] * float(share_powertrain_new.loc[cyear,powert])*req_new_cars * mat_int_cars.loc[powert,"Li"])

                                    
                stock_cars_pivot_copy1 = stock_cars_pivot.copy()
                stock_cars_pivot_copy2 = stock_cars_pivot.copy()
            
                # mass by material type
                # column lists for the tables
                # create table with multilevel column index (powertrains and materials), and cohorts as indexes (rows)
                energyc_list = allocation_energy.columns
                materials_car = mat_int_cars.columns # select from material table  # municipality_list = np.array(pd.Categorical(mun_group.loc[:,"mun_group"]).categories)
                years_mat = np.array(list(range(time_horizon-2020+1))) + 2020 #all years from 2020 to time_horizon
                col_material_cars = list(product(powertrain_list, materials_car)) # double index for columns

                ### create matrix number of cars per powertrain
                col_cohort_power_cars = np.array(list(product(powertrain_list, columns))) # double index for columns
                mi_col_cohort = pd.MultiIndex.from_tuples(list(product(powertrain_list, columns)))
                energyc_powert_cols = pd.MultiIndex.from_tuples(list(product(powertrain_list, energyc_list)))

                # initialize matrixes 
                cars_powertrain_coh_z = np.zeros(shape = [indexes.shape[0], mi_col_cohort.shape[0]]) #matrix with all cohorts and powertrain types to calculate fuel economy, distances and energy consumption
                cars_powertrain_coh = pd.DataFrame(data = cars_powertrain_coh_z, index = indexes, columns = mi_col_cohort) #number of cars per powertrain type
                col_pow_coh = np.zeros(shape = [indexes.shape[0], col_cohort_power_cars.shape[0]])
                avmass_powertrain_coh = pd.DataFrame(data = col_pow_coh, index = indexes, columns = mi_col_cohort ) #average mass of cars_powertrain_coh from size_cars_new 
                fuelecon_powertrain_coh = pd.DataFrame(data = col_pow_coh, index = indexes, columns = mi_col_cohort) #fuel economy of cars_powertrain_coh from size_cars_new 
                distance_powertrain_coh = pd.DataFrame(data = col_pow_coh, index = indexes, columns = mi_col_cohort) #total distance of cars_powertrain_coh from pop_dw_mun["distance vehkm per car"]
                energy_powertrain_coh = pd.DataFrame(data = col_pow_coh, index = indexes, columns = mi_col_cohort) # energy consumption from fuelecon_powertrain_coh, distance_powertrain_coh
                energy_powertrain_z = np.zeros(shape = [indexes.shape[0], powertrain_list.shape[0]])
                energy_powertrain = pd.DataFrame(data = energy_powertrain_z, index = indexes, columns = powertrain_list)
                energyc_powert_cars_z = np.zeros(shape = [indexes.shape[0], energyc_powert_cols.shape[0]])
                energyc_powert_cars = pd.DataFrame(data = energyc_powert_cars_z, index = indexes, columns = energyc_powert_cols)
                emissions_dir_caruse_energyc_z = np.zeros(shape = [indexes.shape[0], energyc_list.shape[0]])
                emissions_dir_caruse_energyc = pd.DataFrame(data = emissions_dir_caruse_energyc_z, index = indexes, columns = energyc_list)
                emissions_indir_caruse_energyc = emissions_dir_caruse_energyc.copy()
        
                #### %% 2. CALCULATE EMISSIONS IN USE % ####
                # adapt veh·km per age: matrix 
                ages_cars = np.arange(0, 19)
                distance_cars_pivot = pd.DataFrame(data = stock_cars_pivot_z, index = indexes, columns = columns)
                vehkm_per_age_z = np.zeros(shape = [indexes.shape[0], ages_cars.shape[0]])
                vehkm_per_age = pd.DataFrame(data = vehkm_per_age_z, index = indexes, columns = ages_cars)
                stock_cars_pivot_cop = stock_cars_pivot.copy()
                pop_dw_mun["newer cars"] = 0
        
                for ryear in indexes:
                    for age in ages_cars:
                        totkm_newercars = distance_cars_pivot.loc[ryear,:].sum()
                        totkm_oldercars = pop_dw_mun.loc[ryear,"distance vehkm"] - totkm_newercars 
                        if totkm_oldercars > 0:
                            if age < 18:
                                dist_per_veh = float(pop_dw_mun.loc[ryear,"distance vehkm per car"]*adapt_vehkm.loc[age])
                                vehkm_per_age.loc[ryear,age] = dist_per_veh
                                num_cars_coh = stock_cars_pivot_cop.loc[ryear,ryear-age]
                                distance_cars_pivot.loc[ryear,ryear-age] = dist_per_veh * stock_cars_pivot_cop.loc[ryear,ryear-age]
                                pop_dw_mun.loc[ryear,"newer cars"] = pop_dw_mun.loc[ryear,"newer cars"] + num_cars_coh
                                if (pop_dw_mun.loc[ryear,"distance vehkm"] - distance_cars_pivot.loc[ryear,:].sum()) < 0:
                                    vehkm_per_age.loc[ryear,age] = (pop_dw_mun.loc[ryear,"distance vehkm"] - distance_cars_pivot.loc[ryear,:].sum())/num_cars_coh
                            else:
                                vehkm_per_age.loc[ryear,age] = totkm_oldercars/(pop_dw_mun.loc[ryear,"cars"] - pop_dw_mun.loc[ryear,"newer cars"])
        
                vehkm_per_age_copy = vehkm_per_age.copy()
                
                #calculate distance and energy per powertrain type
                for ryear in indexes:
                    for powert in powertrain_list:
                        for cyear in columns:
                            share_pow = float(share_powertrain_new.loc[cyear,powert])
                            cars = stock_cars_pivot_copy2.loc[ryear, cyear]
                            cars_powertrain_coh.loc[(ryear),(powert,cyear)] = cars * share_pow
        
                cars_powertrain_coh_copy2 = cars_powertrain_coh.copy()    
                pop_dw_mun["cars - 2"] = cars_powertrain_coh_copy2.sum(1)
        
                for ryear in indexes:
                    for powert in powertrain_list:
                        for cyear in columns:                    
                            age = ryear-cyear
                            if (age < 18)&(age >=0):
                                dist_per_car = vehkm_per_age_copy.loc[ryear, age] 
                            elif (age > 18):
                                dist_per_car = vehkm_per_age_copy.loc[ryear, 18]
                            else:
                                dist_per_car = 0
                            num_cars_coh_pow = cars_powertrain_coh_copy2.loc[ryear,(powert,cyear)]
                            distance_powertrain_coh.loc[(ryear),(powert, cyear)] = dist_per_car * num_cars_coh_pow
                    
                distance_powertrain_coh_copy = distance_powertrain_coh.copy()
                pop_dw_mun["distance vehkm - 2"] = distance_powertrain_coh_copy.sum(1)
                cars_powertrain_coh_copy = cars_powertrain_coh.copy()
        
                for ryear in indexes:
                    for powert in powertrain_list:
                        for cyear in columns:          
                            dist_per_veh = distance_powertrain_coh_copy.loc[(ryear),(powert, cyear)]
                            avmass = size_cars_new_sc.loc[cyear,powert]   
                            sca_fe = fuel_economy_regression.loc[powert,"scalar_mult"]
                            exp_fe = avmass*fuel_economy_regression.loc[powert,"scalar_exp"]
                            fuelecon =  sca_fe* math.exp(exp_fe)
                            dist = distance_powertrain_coh_copy.loc[(ryear),(powert, cyear)]
                            if cyear <= 2010:
                                coef_age = float(fuel_economy_coefficient.loc[cyear])
                            else:
                                coef_age = 1
                                energy_powertrain_coh.loc[(ryear),(powert,cyear)] = fuelecon * coef_age * dist/100
        
                energy_powertrain_coh_copy = energy_powertrain_coh.copy()
                energy_powertrain = energy_powertrain_coh_copy.groupby(level=0, axis =1).sum(1)

                for year in indexes:
                    for powert in powertrain_list:
                        for energyc in energyc_list:
                            alloc = allocation_energy.loc[powert,energyc]
                            energy = energy_powertrain.loc[year,powert]
                            energyc_powert_cars.loc[year,(powert,energyc)] = alloc * energy

                energyc_powert_cars_s = energyc_powert_cars.T
                energyc_cars = energyc_powert_cars_s.groupby(level = 1).sum(0) #we sum all powertrain types
        
                energyc_cars_copy1 = energyc_cars.copy()
                energyc_cars_copy2 = energyc_cars.copy()
        
                # calculation direct emissions
                for year in indexes:
                    for energyc in energyc_list:
                        energy1_temp = energyc_cars_copy1.loc[energyc,year]
                        emi_int_dir_temp = emi_int_dir_fuel_use.loc[energyc,"emis_int"]/1000 # grams to kg
                        emissions_dir_temp = energy1_temp * emi_int_dir_temp
                        emissions_dir_caruse_energyc.loc[year,energyc] = emissions_dir_temp
        
                #calculation indirect emissions
                for year in indexes:
                    for energyc in energyc_list:
                        energy2_temp = energyc_cars_copy2.loc[energyc,year]
                        emi_int_ind_temp = emi_int_indir_fuel_use_sc.loc[energyc,str(year)]/1000 #grams to kg
                        emissions_indir_temp = energy2_temp * emi_int_ind_temp
                        emissions_indir_caruse_energyc.loc[year,energyc] = emissions_indir_temp

                emissions_dir_caruse = emissions_dir_caruse_energyc.sum(1)
                emissions_indir_caruse = emissions_indir_caruse_energyc.sum(1)
                pop_dw_mun["emissions direct car use"] = emissions_dir_caruse
                pop_dw_mun["emissions indirect car use"] = emissions_indir_caruse
               
                #### %% 3. CALCULATE EMISSIONS IN PRODUCTION % ####
                new_cars_powertrain_z = np.zeros(shape =  [indexes.shape[0],share_powertrain_new.shape[1]])
                new_cars_powertrain = pd.DataFrame(data = new_cars_powertrain_z, index = indexes, columns = powertrain_list)

                ### create matrix mass new cars to calculate materials and ghg in production
                mass_new_cars_pow = pd.DataFrame(data = new_cars_powertrain_z, index = indexes, columns = share_powertrain_new.columns)

                # fill data total mass per new cars
                for powert in powertrain_list:
                    for cohort in indexes:
                        share_pow = share_powertrain_new.loc[(cohort,"Sweden"),powert]
                        new_cars = pop_dw_mun.loc[cohort,"new_cars"]
                        new_cars_pow = share_pow * new_cars
                        new_cars_powertrain.loc[cohort,powert] = new_cars_pow

                new_cars_powertrain_copy = new_cars_powertrain.copy(deep=True)
                    
                # make dataframe with all combinations 
                e_n_c_zeros = np.zeros(shape = [years_mat.shape[0],powertrain_list.shape[0]])
                emissions_new_cars_powertrain = pd.DataFrame(e_n_c_zeros,index = years_mat, columns = powertrain_list ) #same matrix with zeros to fill afterwards
            
                # fill materials_new_cars_powertrain
                for cohort in years_mat:
                    for powert in powertrain_list:
                        emissions_new_cars_powertrain.loc[cohort,powert]= new_cars_powertrain_copy.loc[cohort,powert]*(emi_int_prod_cars.loc[powert,"a_coef"]*size_cars_new_sc.loc[cohort,powert]+emi_int_prod_cars.loc[powert,"b_coef"])

                emissions_new_cars = emissions_new_cars_powertrain.sum(1) #sum of columns is the total emissions for the cars produced in a year
                pop_dw_mun["emissions car production"] = emissions_new_cars
                pop_dw_mun["emissions car prod per car"] =emissions_new_cars/pop_dw_mun["new_cars"]
        
                #column of total emissions summing all categories
                pop_dw_mun["emissions"] = pop_dw_mun["emissions dwelling production"] + pop_dw_mun["emissions car production"] + pop_dw_mun["emissions direct car use"] + pop_dw_mun["emissions indirect car use"]
    
##############################################    
################# WRAPPING UP ################
############################################## 

            #when we have calculated everything:
                pop_dw_mun_ml = pop_dw_mun.copy()
                scenario_comb = scenario_on + "-" + scenario_basic
                pop_dw_mun_ml.columns = pd.MultiIndex.from_product([[scenario_comb],[mun_type_on],pop_dw_mun.columns])
                if mun_type_on == "SKL01 metropolitan municipalities (3 municipalities)":
                    pop_dw_mun_sc = pop_dw_mun_ml.copy()
                else:
                    pop_dw_mun_sc = pop_dw_mun_sc.merge(pop_dw_mun_ml, how ="left", on = "year")
        
            pop_dw_mun_sc_cop = pop_dw_mun_sc.copy()
            
            #summary table with results from 3 scenarios and all municipality types
            if (scenario_comb == scenario_list[0]+"-"+baseScenarioTypes[0]):
                results_mun_sc = pop_dw_mun_sc_cop.copy()
            else:
                results_mun_sc = results_mun_sc.merge(pop_dw_mun_sc_cop, how ="left", on = "year") 
    
            pop_dw_sc_t1 = pop_dw_mun_sc.T
            pop_dw_sc_t = pop_dw_sc_t1.groupby(level = [0,2]).sum()
            pop_dw_sc = pop_dw_sc_t.T
            
            pop_dw_sc[(scenario_comb, 'area per capita')] = pop_dw_sc[(scenario_comb,'Floor Area')]/pop_dw_sc[(scenario_comb, 'Population')] #xxxx
            pop_dw_sc[(scenario_comb,'car ownership per capita')] = pop_dw_sc[(scenario_comb,'cars')]/ pop_dw_sc[(scenario_comb,'Population')] #xxxx
            pop_dw_sc[(scenario_comb,'distance paskm per capita')] = pop_dw_sc[(scenario_comb,'distance paskm')]/pop_dw_sc[(scenario_comb,'Population')] #xxxx
            pop_dw_sc[(scenario_comb,'distance vehkm per car')] = pop_dw_sc[(scenario_comb,'distance vehkm')]/pop_dw_sc[(scenario_comb,'cars')] #xxxx
            pop_dw_sc[(scenario_comb,'emissions car prod per car')] = pop_dw_sc[(scenario_comb,'emissions car production')]/pop_dw_sc[(scenario_comb,'new_cars')] #xxxx
            pop_dw_sc[(scenario_comb,'residential density')] = pop_dw_sc[(scenario_comb,'Land Use')]/pop_dw_sc[(scenario_comb,'Population')] #xxxx
            pop_dw_sc[(scenario_comb,'household size')] = pop_dw_sc[(scenario_comb,'Population')] /pop_dw_sc[(scenario_comb,'Number of dwellings - 2')] #xxxx
            
            #summary table with results from 3 scenarios
            if scenario_comb == scenario_list[0]+"-"+baseScenarioTypes[0]:#"base-current values":
                results_sc = pop_dw_sc.copy()
            else:
                results_sc = results_sc.merge(pop_dw_sc, how ="left", on = "year")
            results_sc_t = results_sc.T
            results_sc=results_sc.rename_axis(["scenario","variable"],axis =1)
            results_sc=results_sc.rename_axis(["year"],axis =0)
            results_sc_pivot  =results_sc.T
            results_sc_pivot.drop('a', level='variable', inplace=True)#xxxx
            results_sc_pivot.drop('hh_size_proj', level='variable', inplace=True)#xxxx
            results_sc_2  =results_sc_pivot.T #xxxx
            results_sc_2020 = results_sc_2.loc[2020]
            results_sc_2020_pivot = results_sc_2020.to_frame().pivot_table(index="variable",columns="scenario")
            results_sc_2100= results_sc_2.loc[2100]
            results_sc_2100_pivot = results_sc_2100.to_frame().pivot_table(index="variable",columns="scenario")
            results_sc_sum = results_sc_2.sum(0) #summing for the whole time interval of analysis
            results_sc_sum_pivot = results_sc_sum.to_frame().pivot_table(index="variable",columns="scenario")


currentDateAndTimeF= datetime.now()
print("The current date and time is", currentDateAndTimeF)
