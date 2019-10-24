import numpy as np 
import math 
import matplotlib.pyplot as plt 
import pandas as pd 

import plotly.offline as py

import plotly.graph_objs as go
from plotly.graph_objs import *

import numpy as np

print("Running simulations...")

##simulation length
t=float(1)
dt=float(1/365)
##

##number of simulations
num_simulations = 100
##

jump_probabilities = [.55,(1.1+.55),(.82+1.1+.55),100]
jump_severities = [9.5,24.96,66.64,0]

#overall drift
mu=0.06
#overall volatility
sigma=1.2

#time for price to recover from a crash
recovery_time=3

time_to_start_recovery = 3

#liquidation threshold
collateral_cutoff=1.5

#?
reentry_time=7

result_array = [0 for i in range(num_simulations)]

total_losses = []

eth_price_record = {}

for simulation in range(num_simulations):
    cdps = [
        {"bucket":1.55,"collat":1.55,"debt":1e6,"open":True,"re_entry_clock":0,"reversion_time":1},
        {"bucket":1.6,"collat":1.6,"debt":1e6,"open":True,"re_entry_clock":0,"reversion_time":1},
        {"bucket":1.75,"collat":1.75,"debt":3e6,"open":True,"re_entry_clock":0,"reversion_time":3},
        {"bucket":2.25,"collat":2.25,"debt":9.5e6,"open":True,"re_entry_clock":0,"reversion_time":3},
        {"bucket":2.5,"collat":2.5,"debt":3.5e6,"open":True,"re_entry_clock":0,"reversion_time":5},
        {"bucket":2.75,"collat":2.75,"debt":18e6,"open":True,"re_entry_clock":0,"reversion_time":5},
        {"bucket":3.25,"collat":3.25,"debt":14e6,"open":True,"re_entry_clock":0,"reversion_time":5},
        {"bucket":3.75,"collat":3.75,"debt":14e6,"open":True,"re_entry_clock":0,"reversion_time":7},
        {"bucket":4.25,"collat":4.25,"debt":11e6,"open":True,"re_entry_clock":0,"reversion_time":7},
        {"bucket":4.75,"collat":4.75,"debt":7e6,"open":True,"re_entry_clock":0,"reversion_time":10},
        {"bucket":5.25,"collat":5.25,"debt":3e6,"open":True,"re_entry_clock":0,"reversion_time":10},
        {"bucket":7.5,"collat":7.5,"debt":15e6,"open":True,"re_entry_clock":0,"reversion_time":10}
    ]
    x = [i for i in range(int(t/dt))]
    f = [0 for i in range(int(t/dt))]
    M = [0 for i in range(int(t/dt))]
    liquidated_debt = [0 for i in range(int(t/dt))]
    liquidated_collateral = [0 for i in range(int(t/dt))]
    undercollateralized_loss = [0 for i in range(int(t/dt))]
    undercollateralized_loss_perc = [0 for i in range(int(t/dt))]
    slippage_loss = [0 for i in range(int(t/dt))]
    debt_supply = [sum([c["debt"] for c in cdps]) for i in range(int(t/dt))]
    collateralizations = {}
    M[0]=100
    amount_to_recover = 0
    recovery_clock = 0
    clock_to_start_recovery = 0

    for time_step in range(1,int(t/dt)):
        
        jump_random = np.random.random()*100        
        jump_value = .01*jump_severities[min([jump_probabilities.index(j) for j in jump_probabilities if j>jump_random])]

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

                #TEST
                #print(jump_value,M[time_step],M[time_step-1])

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

        #TEST
        #print("f",f[time_step],"m",M[time_step]) 

        #for every bucket in the CDP distribution
        for bucket in cdps:                      
            #the reversion time is defined within the bucket  
            cdp_reversion_time = bucket["reversion_time"]
            #the collateralization ratio is the present collateralization ratio,
            #modified by the most recent % change in asset price and 
            #modified by the distance between the present collateralization ratio and the base state of the bucket (divided by the reversion time for that bucket)
            bucket["collat"] = M[time_step]/M[time_step-1]*(bucket["collat"]+(bucket["bucket"]-bucket["collat"])/cdp_reversion_time)

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
                liquidated_debt[time_step]+=bucket["debt"]
                #set the re-entry clock equal to the defined re-entry time
                bucket["re_entry_clock"]=reentry_time
                #set the bucket to be closed
                bucket["open"]=False        
                #if the bucket is undercollateralized, then add the  deficit to the undercollateralized losses
                if bucket["collat"]<1: undercollateralized_loss[time_step]+=bucket["debt"]*(1-bucket["collat"])

        #count up the open buckets and measure the debt supply at each time step
        debt_supply[time_step]=sum([c["debt"] for c in cdps if c["open"]==True])

        #calculate the loss from slippage as a function of liquidated debt
        slippage_loss[time_step] = liquidated_debt[time_step] * min(1,((1.96e-9)*liquidated_debt[time_step] + (2.52e-18)*math.pow(liquidated_debt[time_step],2) + (3.14e-24)*math.pow(liquidated_debt[time_step],3) + (2.13e-32)*math.pow(liquidated_debt[time_step],4)))

    #record the total loss from the simulation
    result_array[simulation] = sum(slippage_loss)+sum(undercollateralized_loss)
    #record the total array of ETH prices from the simulation
    eth_price_record[simulation] = M
print(np.average(result_array))    
##

data = [go.Scatter(x=x,
            y=M, mode='lines',line=dict(color="blue"))
       ]

layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="ETH Price"))

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="ETH Price over time.html")

##

data = [
            go.Scatter(x=x,y=liquidated_debt,mode='lines',line=dict(color="red"),yaxis="y",xaxis="x"),
            go.Scatter(x=x,y=M,mode='lines',line=dict(color="blue"),yaxis="y2",xaxis="x")
           ]
layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Dai Liquidated"),yaxis2=dict(title="ETH Price",overlaying="y",side="right"),showlegend=False)

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="Eth Price vs. Dai liquidated.html")

##

data = [
            go.Scatter(x=x,y=undercollateralized_loss,mode='lines',line=dict(color="red"),yaxis="y",xaxis="x"),
            go.Scatter(x=x,y=M,mode='lines',line=dict(color="blue"),yaxis="y2",xaxis="x")
           ]
layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Dai Lost"),yaxis2=dict(title="ETH Price",overlaying="y",side="right"),showlegend=False)

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="Eth Price vs. Undercollateralized Loss.html")

##

data = [go.Scatter(x=x,
            y=debt_supply, mode='lines',line=dict(color="blue"))
       ]

layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Dai Supply"))

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="Dai Supply.html")

##

data = [go.Scatter(x=x,
            y=slippage_loss, mode='lines',line=dict(color="red"))
       ]

layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Slippage Loss"))

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="Slippage Loss.html")

##

data = [go.Scatter(x=list(range(num_simulations)),
            y=result_array, mode='lines',line=dict(color="red"))
       ]

layout = go.Layout(xaxis=dict(title="Simulations"),yaxis=dict(title="Total Losses"))

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="Total Losses.html")

df = pd.DataFrame(eth_price_record)
print(df)