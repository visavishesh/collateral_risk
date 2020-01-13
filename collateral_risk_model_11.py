import numpy as np 
import math 
import matplotlib.pyplot as plt 
import pandas as pd 

import plotly.offline as py

import plotly.graph_objs as go
from plotly.graph_objs import *

import numpy as np
import scipy.stats as stats

import plotly.express as px

print("Running simulations...") 

data_results = []
result_matrix = []

def run_iter(iteration,mu,sigma,collateral_cutoff,liquidation_penalty,sim_len,cdp_distribution,jump_probabilities,jump_severities):
    ##simulation length (should we use avg age of debt?)
    t=float(1)
    dt=float(t/sim_len)

    #time for price to recover from a crash (days)
    recovery_time=3 
    #the time after a crash before the recovery period starts (days)
    time_to_start_recovery = 3

    #the efficiency of the auction in which collateral is sold. 0 means full collateral sold, 1 means only amount equal to the debt sold. 
    auction_efficiency = 0

    #the number of days before CDPs re-enter the debt pool after liquidation
    reentry_time=7

    data = [{} for i in range(int(t/dt))]
    
    #steady state of collaterlization
    cdps = cdp_distribution

    x = [i for i in range(int(t/dt))]
    f = [0 for i in range(int(t/dt))]
    M = [0 for i in range(int(t/dt))]
    
    collateralizations = [{} for i in range(int(t/dt))]
    
    M[0]=100
    amount_to_recover = 0
    recovery_clock = 0
    clock_to_start_recovery = 0

    for time_step in range(0,int(t/dt)):
        if time_step==0:
            data[time_step] = {
            "undercollateralized_loss":0,
            "loss_gain":0,
            "liquidated_debt":0,
            }
        else:
            jump_random = np.random.random()*100        
            jump_value = .01*jump_severities[min([jump_probabilities.index(j) for j in jump_probabilities if j>jump_random])]
            data[time_step]["liquidated_debt"]=0
            data[time_step]["undercollateralized_loss"]=0
            data[time_step]["loss_gain"]=0

            #if there is recovery to be made from a crash
            if recovery_clock>0:            
                #if the n-day delay to start that recovery is still in effect
                if clock_to_start_recovery>0:
                    #then resume normal GBM values
                    f[time_step] = f[time_step-1]+math.sqrt(dt)*np.random.normal(0,1,1)
                    M[time_step] = M[0]*math.exp((mu-(math.pow(sigma,2))/2)*(time_step*dt)+sigma*f[time_step]) 

                    #and decrement the clock to start the recovery
                    clock_to_start_recovery-=1
                #otherwise, if the n-day delay to start the recovery has elapsed
                else:
                    #then use the total amount to recover divded by the recovery time
                    M[time_step] = M[time_step-1]+amount_to_recover/recovery_time
                    f[time_step] = (math.log(M[time_step]/M[0])-(mu-(math.pow(sigma,2))/2)*(time_step*dt))/sigma                
                    #and decrement the recovery clock
                    recovery_clock-=1
            #otherwise, if there is no recovery to be had
            else:
                #if the jump value has been triggered
                if jump_value>0:
                    #then drop the function by 100% less the jump value
                    
                    M[time_step] = M[time_step-1]*(1-jump_value)                
                    f[time_step] = (math.log(M[time_step]/M[0])-(mu-(math.pow(sigma,2))/2)*(time_step*dt))/sigma

                    #and set the recovery amount to be the positive value of the amount of the drop
                    amount_to_recover = abs(M[time_step-1]-M[time_step])
                    #set the recovery clock to be the recovery time parameter
                    recovery_clock = recovery_time
                    #set the clock to start the recovery to be the time to start recovery parameter
                    clock_to_start_recovery = time_to_start_recovery    
                #otherwise if no jump has been triggered
                else:
                    #then use the normal GBM values
                    f[time_step] = f[time_step-1]+math.sqrt(dt)*np.random.normal(0,1,1)
                    M[time_step] = m_value = M[0]*math.exp((mu-(math.pow(sigma,2))/2)*(time_step*dt)+sigma*f[time_step])      
            
            #for every bucket in the CDP distribution
            for bucket in cdps:                      
                #the reversion time is defined within the bucket  
                cdp_reversion_time = bucket["reversion_time"]
                #the collateralization ratio is the present collateralization ratio,
                #modified by the most recent % change in asset price and 
                #modified by the distance between the present collateralization ratio and the base state of the bucket (divided by the reversion time for that bucket)
                bucket["collat"] = M[time_step]/M[time_step-1]*(bucket["collat"]+(bucket["bucket"]-bucket["collat"])/cdp_reversion_time)
                #accrual of fees
                bucket["collat"]=bucket["collat"]*(1-0.00054794520547945)

                collateralizations[time_step][bucket["bucket"]]=bucket["collat"]

                #if the re-entry clock of the current CDP bucket is still counting
                if bucket["re_entry_clock"]>0:                
                    #then decrement the clock
                    bucket["re_entry_clock"]-=1
                    #if, after decrement, the clock is zero then
                    if bucket["re_entry_clock"]==0: 
                        #reset the collateralization ratio to be the baseline
                        bucket["collat"]=bucket["bucket"]
                        #set the bucket to be open again
                        bucket["open"]=True
                #otherwise, if the re-entry clock is finished
                elif bucket["collat"]<collateral_cutoff: 
                    #add the size of the bucket to the amount of liquidated debt
                    data[time_step]["liquidated_debt"]+=bucket["debt"]
                    #set the re-entry clock equal to the defined re-entry time
                    bucket["re_entry_clock"]=reentry_time
                    #set the bucket to be closed
                    bucket["open"]=False                        
                    #determine how much collateral is being sold in the auction range from (just debt to total collateral)
                    auction_size = bucket["debt"]*(1-auction_efficiency) + bucket["collat"]*bucket["debt"]*auction_efficiency
                    #the % lost in slippage is a function of the amount sold in auction
                    #old -- > 
                    slip = slippage(auction_size)                    
                    #the amount of dai recovered in auction is the lesser of the debt in the CDP plus the liquidation penalty, and the amount of collateral in the CDP less slippage
                    dai_obtained = min(bucket["debt"]*(1+liquidation_penalty),(bucket["collat"]*bucket["debt"])*(1-slip))
                    data[time_step]["undercollateralized_loss"] += max(0,1-bucket["collat"])*bucket["debt"]
                    #the amount of loss or gain is equal to the amount of Dai obtained in auction less the debt to recover
                    data[time_step]["loss_gain"] += (dai_obtained-bucket["debt"])
    
        data[time_step]["iteration"]=iteration
        data[time_step]["time_step"]=time_step
        data[time_step]["debt_supply"]=sum([c["debt"] for c in cdps if c["open"]==True])
        data[time_step]["asset_price"] = M[time_step]
        data[time_step]["sigma"]=sigma
        data[time_step]["collateral_cutoff"]=collateral_cutoff
        data[time_step]["sim_len"]=sim_len
    return(data)

def run_sim(num_simulations,sim_len,mu,sigma,debt_ceiling,liquidation_penalty,collateral_cutoff,cdp_distribution):
    results=[]

    for iteration in range(num_simulations):
        result = run_iter(iteration = iteration, mu=mu,sigma = sigma,collateral_cutoff = collateral_cutoff,liquidation_penalty=liquidation_penalty,sim_len = sim_len,cdp_distribution=cdp_distribution,jump_probabilities=jump_probabilities,jump_severities=jump_severities)
        results+=result

    return(pd.DataFrame(results))

num_simulations=20
sim_len=365
mu_range = list(np.arange(-.5, .5, 0.05))
sigma_range = list(np.arange(.1, 2, 0.1))

collateral_cutoff_range = list(np.arange(1.0,2,.05))
collateral_cutoff = 1.5
debt_ceiling = 1.2e6
cdp_distribution = [
            {"bucket":collateral_cutoff+.15,"collat":collateral_cutoff+.15,"debt":1.5*debt_ceiling,"open":True,"re_entry_clock":0,"reversion_time":2},
            {"bucket":collateral_cutoff+.25,"collat":collateral_cutoff+.25,"debt":8.55*debt_ceiling,"open":True,"re_entry_clock":0,"reversion_time":2},
            {"bucket":collateral_cutoff+.50,"collat":collateral_cutoff+.5,"debt":12.04*debt_ceiling,"open":True,"re_entry_clock":0,"reversion_time":3},
            {"bucket":collateral_cutoff+.75,"collat":collateral_cutoff+.75,"debt":13.78*debt_ceiling,"open":True,"re_entry_clock":0,"reversion_time":3},
            {"bucket":collateral_cutoff+1,"collat":collateral_cutoff+1,"debt":12.07*debt_ceiling,"open":True,"re_entry_clock":0,"reversion_time":5},
            {"bucket":collateral_cutoff+1.25,"collat":collateral_cutoff+1.25,"debt":8.09*debt_ceiling,"open":True,"re_entry_clock":0,"reversion_time":5},
            {"bucket":collateral_cutoff+1.50,"collat":collateral_cutoff+1.5,"debt":11.54*debt_ceiling,"open":True,"re_entry_clock":0,"reversion_time":5},
            {"bucket":collateral_cutoff+2.0,"collat":collateral_cutoff+2,"debt":11.00*debt_ceiling,"open":True,"re_entry_clock":0,"reversion_time":7},
            {"bucket":collateral_cutoff+2.50,"collat":collateral_cutoff+2.5,"debt":9.54*debt_ceiling,"open":True,"re_entry_clock":0,"reversion_time":7},
            {"bucket":collateral_cutoff+3.25,"collat":collateral_cutoff+3.25,"debt":11.90*debt_ceiling,"open":True,"re_entry_clock":0,"reversion_time":10}
        ]

a=.55
b=.8
c=.138*2
jump_probabilities = [a,(a+b),(a+b+c),100]
jump_severities = [9.5,24.96,50,0]

def slippage(auction_size):
    slip = min(1,10*((1.96e-9)*auction_size + (2.52e-18)*math.pow(auction_size,2) + (3.14e-24)*math.pow(auction_size,3) + (2.13e-32)*math.pow(auction_size,4)))
    #new -- >
    #slip = min(1, (-6.68e-10)*auction_size + (1.18e-16)*math.pow(auction_size,2) + (-4.91e-25)*math.pow(auction_size,3) )
    #new --> new
    #slip = max(0,min(1,-0.0192*(auction_size>0)+auction_size*(1.4E-9)+math.pow(auction_size,2)*(1.37E-16)+math.pow(auction_size,3)*(-6.95E-25)))
    return(slip)

#Controls Everything
data = run_sim(num_simulations=20,sim_len=365,mu=0,sigma=.8,debt_ceiling = debt_ceiling,liquidation_penalty=.13,collateral_cutoff=collateral_cutoff,cdp_distribution=cdp_distribution,jump_probabilities=jump_probabilities,jump_severities=jump_severities)

#Pretty Printing
total = data.groupby(["iteration"]).sum()
avg = data.groupby(["iteration"]).mean()
debt = avg["debt_supply"]
total["loss_gain_total"]=total["loss_gain"]
df = avg.merge(total["loss_gain_total"],how="inner",on="iteration")
df["loss_gain_perc"]=df["loss_gain_total"]/df["debt_supply"]
print(df)
print(data.nsmallest(max(1,int(num_simulations*sim_len*.001)), columns=["loss_gain"],keep="all"))
