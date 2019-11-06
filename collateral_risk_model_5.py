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

##simulation length
t=float(1)
dt=float(1/365)
##

##number of simulations
num_simulations = 100
##

a=.55
b=.8
c=.138*2
jump_probabilities = [a,(a+b),(a+b+c),100]
jump_severities = [9.5,24.96,50,0]


# #lower, upper, mu, and sigma are four parameters for jump risk
# lower, upper = 0, 1
# mu, sigma = 0.2, 0.1
# jump_prob_cutoff = .5


# #instantiate an object X using the above four parameters,
# jump_distribution = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

# #generate 1000 sample data
# jump_samples = jump_distribution.rvs(1000)

# pdf_probs = stats.truncnorm.pdf(jump_samples, (lower-mu)/sigma, (upper-mu)/sigma, mu, sigma)
# plt.plot(jump_samples[jump_samples.argsort()],pdf_probs[jump_samples.argsort()],linewidth=2.3,label='PDF curve')
# plt.show()
# ##


#overall drift
mu=0.0
#overall volatility
sigma=1.2

#time for price to recover from a crash (days)
recovery_time=3 
#the time after a crash before the recovery period starts (days)
time_to_start_recovery = 3

#liquidation threshold
collateral_cutoff=1.5
#liqudation penalty 
liquidation_penalty = 0.0
#the efficiency of the auction in which collateral is sold. 0 means full collateral sold, 1 means only amount equal to the debt sold. 
auction_efficiency = 0

#the number of days before CDPs re-enter the debt pool after liquidation
reentry_time=7

result_array = [0 for i in range(num_simulations)]

total_losses = []

eth_price_record = {}

for simulation in range(num_simulations):
    cdps = [
        {"bucket":1.55,"collat":1.55,"debt":2e6,"open":True,"re_entry_clock":0,"reversion_time":3},
        {"bucket":1.75,"collat":1.75,"debt":2e6,"open":True,"re_entry_clock":0,"reversion_time":3},
        {"bucket":2,"collat":2,"debt":2e6,"open":True,"re_entry_clock":0,"reversion_time":3},
        {"bucket":2.25,"collat":2.25,"debt":9e6,"open":True,"re_entry_clock":0,"reversion_time":3},
        {"bucket":2.5,"collat":2.5,"debt":3e6,"open":True,"re_entry_clock":0,"reversion_time":5},
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
    
    loss_gain = [0 for i in range(int(t/dt))]  
    loss_gain_perc = [0 for i in range(int(t/dt))]  

    slippage_loss = [0 for i in range(int(t/dt))]
    debt_supply = [sum([c["debt"] for c in cdps]) for i in range(int(t/dt))]
    
    collateralizations = [{} for i in range(int(t/dt))]
    
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
                liquidated_debt[time_step]+=bucket["debt"]
                #set the re-entry clock equal to the defined re-entry time
                bucket["re_entry_clock"]=reentry_time
                #set the bucket to be closed
                bucket["open"]=False                        
                #determine how much collateral is being sold in the auction range from (just debt to total collateral)
                auction_size = bucket["debt"]*(1-auction_efficiency) + bucket["collat"]*bucket["debt"]*auction_efficiency
                #the % lost in slippage is a function of the amount sold in auction
                slip = min(1,((1.96e-9)*auction_size + (2.52e-18)*math.pow(auction_size,2) + (3.14e-24)*math.pow(auction_size,3) + (2.13e-32)*math.pow(auction_size,4)))                
                #the amount of dai recovered in auction is the lesser of the debt in the CDP plus the liquidation penalty, and the amount of collateral in the CDP less slippage
                dai_obtained = min(bucket["debt"]*(1+liquidation_penalty),(bucket["collat"]*bucket["debt"])*(1-slip))
                #the amount of loss or gain is equal to the amount of Dai obtained in auction less the debt to recover
                loss_gain[time_step] += (dai_obtained-bucket["debt"])

        #count up the open buckets and measure the debt supply at each time step
        debt_supply[time_step]=sum([c["debt"] for c in cdps if c["open"]==True])
        #calculate the loss or gain per day as a function of the Dai supply on that day
        if liquidated_debt[time_step]!=0: loss_gain_perc[time_step] = loss_gain[time_step]/liquidated_debt[time_step]
        else: loss_gain_perc[time_step]=0

    #record the total loss from the simulation
    result_array[simulation] = sum(loss_gain)/np.average(debt_supply)
    #record the total array of ETH prices from the simulation
    eth_price_record[simulation] = M

print(np.average(result_array))

#display a graph of the ETH price over time
data = [go.Scatter(x=x,
            y=M, mode='lines',line=dict(color="blue"))
       ]

layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="ETH Price"))

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="ETH Price over time.html")

#display a graph of the colalteralization ratios of each bucket over time
df = pd.DataFrame(collateralizations)
data = [
    go.Scatter(
        x=df.index, # assign x as the dataframe column 'x'
        y=df[val],
        yaxis="y",
        name = val,
        line=dict(color="rgb("+str(255*(1-(val/7.5)))+","+str(100*(val/7.5))+", 20)")
    ) for val in df.columns
] + [go.Scatter(x=x,y=M,mode='lines',line=dict(color="blue"),yaxis="y2",xaxis="x")]
layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Collateralization"),yaxis2=dict(title="ETH Price",overlaying="y",side="right"),showlegend=False)
fig = Figure(data=data,layout=layout)
plot_url = py.plot(fig,filename="Collateralizations.html")

#display a graph of the ETH price vs. the amount of Dai liquidated
data = [
            go.Scatter(x=x,y=liquidated_debt,mode='lines',line=dict(color="red"),yaxis="y",xaxis="x"),
            go.Scatter(x=x,y=M,mode='lines',line=dict(color="blue"),yaxis="y2",xaxis="x")
           ]
layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Dai Liquidated"),yaxis2=dict(title="ETH Price",overlaying="y",side="right"),showlegend=False)

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="Eth Price vs. Dai liquidated.html")

#display a graph of the dai supply over time
data = [go.Scatter(x=x,
            y=debt_supply, mode='lines',line=dict(color="blue"))
       ]

layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Dai Supply"))

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="Dai Supply.html")

#display a graph of the ETH price vs amount of loss or gain over time
data = [
            go.Scatter(x=x,y=loss_gain,mode='lines',line=dict(color="red"),yaxis="y",xaxis="x"),
            go.Scatter(x=x,y=M,mode='lines',line=dict(color="blue"),yaxis="y2",xaxis="x")
           ]
layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Loss Gain"),yaxis2=dict(title="ETH Price",overlaying="y",side="right"),showlegend=False)

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="Eth Price vs. Loss Gain.html")

#display a graph of the ETH price vs % of loss or gain over time
data = [
            go.Scatter(x=x,y=loss_gain_perc,mode='lines',line=dict(color="red"),yaxis="y",xaxis="x"),
            go.Scatter(x=x,y=M,mode='lines',line=dict(color="blue"),yaxis="y2",xaxis="x")
           ]
layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Loss Gain Percentage"),yaxis2=dict(title="ETH Price",overlaying="y",side="right"),showlegend=False)

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="Eth Price vs. Loss Gain Percentage.html")

#display a graph of the total losses for each simulation
data = [go.Scatter(x=list(range(num_simulations)),
            y=result_array, mode='lines',line=dict(color="red"))
       ]

layout = go.Layout(xaxis=dict(title="Simulations"),yaxis=dict(title="Total Loss/Gain"))

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="Total Losses.html")

df = pd.DataFrame(eth_price_record)

results = pd.DataFrame(result_array)
fig = px.histogram(results, x=0)
plot_url = py.plot(fig,filename="total_bill.html")