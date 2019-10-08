import numpy as np 
import math 
import matplotlib.pyplot as plt 
import pandas as pd 

import plotly.offline as py

import plotly.graph_objs as go
from plotly.graph_objs import *

import numpy as np

##simulation length
t=float(1)
dt=float(1/365)
##

##number of simulations
num_simulations = 10000
##

##Jump
#load libraries   
import scipy.stats as stats

#lower, upper, mu, and sigma are four parameters
lower, upper = 0, 1
mu, sigma = 0.2, 0.1
jump_prob_cutoff = .4

#instantiate an object X using the above four parameters,
jump_distribution = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

#generate 1000 sample data
jump_samples = jump_distribution.rvs(1000)

# pdf_probs = stats.truncnorm.pdf(jump_samples, (lower-mu)/sigma, (upper-mu)/sigma, mu, sigma)
# plt.plot(jump_samples[jump_samples.argsort()],pdf_probs[jump_samples.argsort()],linewidth=2.3,label='PDF curve')
# plt.show()
##

#overall drift
mu=0.06
#overall volatility
sigma=.47

#time for price to recover from a crash
recovery_time=3

#liquidation threshold
collateral_cutoff=1.5

#?
cdp_reversion_speed=3
#?
reentry_time=7

total_losses = []

for simulation in range(num_simulations):
    cdps = [
        {"bucket":1.75,"collat":1.75,"debt":5,"open":True,"clock":0},
        {"bucket":2.25,"collat":2.25,"debt":9.5,"open":True,"clock":0},
        {"bucket":2.5,"collat":2.5,"debt":3.5,"open":True,"clock":0},
        {"bucket":2.75,"collat":2.75,"debt":18,"open":True,"clock":0},
        {"bucket":3.25,"collat":3.25,"debt":14,"open":True,"clock":0},
        {"bucket":3.75,"collat":3.75,"debt":14,"open":True,"clock":0},
        {"bucket":4.25,"collat":4.25,"debt":11,"open":True,"clock":0},
        {"bucket":4.75,"collat":4.75,"debt":7,"open":True,"clock":0},
        {"bucket":5.25,"collat":5.25,"debt":3,"open":True,"clock":0},
        {"bucket":7.5,"collat":7.5,"debt":15,"open":True,"clock":0}
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

    for time_step in range(1,int(t/dt)):
        jump_value = jump_distribution.rvs(1)[0]        
        if amount_to_recover==0:
            if jump_value>jump_prob_cutoff:
                f[time_step] = f[time_step-1]+math.sqrt(dt)*np.random.normal(0,1,1) 
                M[time_step] = M[time_step-1]*(1-jump_value)

                amount_to_recover = abs(M[time_step-1]-M[time_step])
                recovery_clock = recovery_time
            else:                        
                f[time_step] = f[time_step-1]+math.sqrt(dt)*np.random.normal(0,1,1) 
                M[time_step] = M[0]*math.exp((mu-(math.pow(sigma,2))/2)*(time_step*dt)+sigma*f[time_step])    
            
        else:
            f[time_step] = f[time_step-1]+math.sqrt(dt)*np.random.normal(0,1,1) 
            M[time_step] = M[time_step-1]+amount_to_recover/recovery_time
            recovery_clock-=1
            if recovery_clock==0: amount_to_recover=0

        for bucket in cdps:                        
            bucket["collat"] = M[time_step]/M[time_step-1]*(bucket["collat"]+(bucket["bucket"]-bucket["collat"])/cdp_reversion_speed)

            if bucket["clock"]>0:                
                bucket["clock"]-=1
                if bucket["clock"]==0: 
                    bucket["collat"]=bucket["bucket"]
                    bucket["open"]=True
            elif bucket["collat"]<collateral_cutoff: 
                liquidated_debt[time_step]+=bucket["debt"]
                bucket["clock"]=reentry_time
                bucket["open"]=False         
                if bucket["collat"]<1:                          
                    undercollateralized_loss[time_step]+=bucket["debt"]*(1-bucket["collat"])

        debt_supply[time_step]=sum([c["debt"] for c in cdps if c["open"]==True])

        undercollateralized_loss_perc[time_step] = undercollateralized_loss[time_step]/debt_supply[time_step]

        liquidated_collateral[time_step] = liquidated_debt[time_step]/M[time_step]
        slippage_loss[time_step] = (8.97e-9)*liquidated_collateral[time_step] + (2.22e-11)*math.pow(liquidated_collateral[time_step],2)

    #print("Slippage loss",sum(slippage_loss))
    #print("Undercollateralized loss",sum(undercollateralized_loss_perc))

    total_losses+=[sum(slippage_loss)+sum(undercollateralized_loss_perc)]

print(np.average(total_losses))    
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
            y=undercollateralized_loss_perc, mode='lines',line=dict(color="blue"))
       ]

layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Undercollateralized Loss Percentage"))

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="undercollateralized_loss_perc.html")

##

data = [go.Scatter(x=x,
            y=slippage_loss, mode='lines',line=dict(color="blue"))
       ]

layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Slippage Loss"))

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="Slippage Loss.html")