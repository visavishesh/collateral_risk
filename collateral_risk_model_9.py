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

mu_range = list(np.arange(-.5, .5, 0.05))
sigma_range = list(np.arange(.1, 2, 0.1))
mu = 0
sigma = .8
liquidation_penalty=13
debt_ceiling = 1.2e6
collateral_cutoff_range = list(np.arange(1.0,2,.05))
collateral_cutoff=1.5

def run_sim(sigma,collateral_cutoff):
    ##simulation length (should we use avg age of debt?)
    t=float(1)
    dt=float(1/365)
    ##

    ##number of simulations
    num_simulations = 10
    ##

    a=.55
    b=.8
    c=.138*2
    jump_probabilities = [a,(a+b),(a+b+c),100]
    jump_severities = [9.5,24.96,50,0]

    #time for price to recover from a crash (days)
    recovery_time=3 
    #the time after a crash before the recovery period starts (days)
    time_to_start_recovery = 3

    #the efficiency of the auction in which collateral is sold. 0 means full collateral sold, 1 means only amount equal to the debt sold. 
    auction_efficiency = 0

    #the number of days before CDPs re-enter the debt pool after liquidation
    reentry_time=7

    result_array = [0 for i in range(num_simulations)]
    debt_array = [0 for i in range(num_simulations)]

    total_losses = []

    eth_price_record = {}

    for simulation in range(num_simulations):
        cdps = [
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
                    #old -- > 
                    slip = min(1,10*((1.96e-9)*auction_size + (2.52e-18)*math.pow(auction_size,2) + (3.14e-24)*math.pow(auction_size,3) + (2.13e-32)*math.pow(auction_size,4)))                
                    #new -- >
                    #slip = min(1, (-6.68e-10)*auction_size + (1.18e-16)*math.pow(auction_size,2) + (-4.91e-25)*math.pow(auction_size,3) )
                    #new --> new
                    #slip = max(0,min(1,-0.0192*(auction_size>0)+auction_size*(1.4E-9)+math.pow(auction_size,2)*(1.37E-16)+math.pow(auction_size,3)*(-6.95E-25)))
                    #the amount of dai recovered in auction is the lesser of the debt in the CDP plus the liquidation penalty, and the amount of collateral in the CDP less slippage
                    dai_obtained = min(bucket["debt"]*(1+liquidation_penalty),(bucket["collat"]*bucket["debt"])*(1-slip))
                    #the amount of loss or gain is equal to the amount of Dai obtained in auction less the debt to recover
                    loss_gain[time_step] += (dai_obtained-bucket["debt"])

            #count up the open buckets and measure the debt supply at each time step
            debt_supply[time_step]=sum([c["debt"] for c in cdps if c["open"]==True])
            #calculate the loss or gain per day as a function of the Dai supply on that day
            if liquidated_debt[time_step]!=0: loss_gain_perc[time_step] = loss_gain[time_step]/liquidated_debt[time_step]
            else: loss_gain_perc[time_step]=0
        
        result_dict = { time_step : loss_gain_perc[time_step] for time_step in range(len(loss_gain_perc)) }
        result_dict.update({"sigma":sigma,"collateral_cutoff":collateral_cutoff,"gain-loss":np.average(result_array)})
        result_matrix += [result_dict]

        #record the total loss from the simulation
        result_array[simulation] = sum(loss_gain)
        debt_array[simulation] = np.average(debt_supply)
        #record the total array of ETH prices from the simulation
        eth_price_record[simulation] = M

    result={"sigma":sigma,"collateral_cutoff":collateral_cutoff,"gain-loss":np.average(result_array),"dai_supply":np.average(debt_array)}
    return(result)

for sigma in sigma_range:
    #overall volatility    
    #liqudation penalty
    for collateral_cutoff in collateral_cutoff_range:
        run_sim(sigma,collateral_cutoff)        

        data_results+=[result]
        print(result)

fig = px.scatter_3d(pd.DataFrame(data_results), x='sigma', y='collateral_cutoff', z='gain-loss',
              color='gain-loss',size_max=5,opacity=0.7)

fig.update_layout( width=1200,
    height=800)

plot_url = py.plot(fig,filename="3d_gains.html")

#display a graph of the ETH price over time
data = [go.Scatter(x=x,
            y=M, mode='lines',line=dict(color="blue"))
       ]

layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="ETH Price"))

fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig,filename="ETH Price over time.html")

#display a graph of the colalteralization ratios of each bucket over time
collateral_df = pd.DataFrame(collateralizations)

data = [
    go.Scatter(
        x=collateral_df.index, # assign x as the dataframe column 'x'
        y=collateral_df[val],
        yaxis="y",
        name = val,
        line=dict(color="rgb("+str(255*(1-(val/7.5)))+","+str(100*(val/7.5))+", 20)")
    ) for val in collateral_df.columns
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

df = pd.DataFrame(result_matrix)
df1 = df.drop(['sigma',"collateral_cutoff","gain-loss"], axis=1)
loss_days = list(df1.values.flatten())
num_days = len(loss_days)
point_one_perc = int(num_days/1000)
print(num_days,point_one_perc)
print(sorted(loss_days)[0:point_one_perc])
print(sum(sorted(loss_days)[0:point_one_perc]))