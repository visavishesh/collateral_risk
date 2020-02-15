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
import json

print("Running simulations...") 

data_results = []
result_matrix = []

def run_iter(iteration,mu,sigma,collateral_cutoff,liquidation_penalty,sim_len,cdp_distribution,jump_probabilities,jump_severities,slippage_function,recovery_time,time_to_start_recovery,reentry_time,auction_efficiency):    
    def slippage(auction_size,slippage_function):
        s=slippage_function["scalar"]
        a=slippage_function["constant"]
        b=slippage_function["x"]
        c=slippage_function["x^2"]
        d=slippage_function["x^3"]
        e=slippage_function["x^4"]    
        slip = min(1,s*(a*(auction_size>0)+b*auction_size+c*math.pow(auction_size,2)+d*math.pow(auction_size,3) +e*math.pow(auction_size,4)))
        return(slip)

    t=float(1.0)
    dt=float(t/sim_len)

    data = [{} for i in range(int(t/dt))]
    
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
            "slippage_loss":0,
            "loss_gain":0,
            "liquidated_debt":0,
            }
        else:
            jump_random = np.random.random()*100        
            jump_value = .01*jump_severities[min([jump_probabilities.index(j) for j in jump_probabilities if j>jump_random])]
            data[time_step]["liquidated_debt"]=0
            data[time_step]["undercollateralized_loss"]=0
            data[time_step]["loss_gain"]=0
            data[time_step]["slippage_loss"]=0

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
                data[time_step]["collat_bucket_"+str(bucket["bucket"])] = bucket["collat"]

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
                    slip = slippage(auction_size,slippage_function)
                    data[time_step]["slippage_loss"]+=slip*auction_size
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

def run_sim(config):    
    num_simulations=config["num_simulations"]
    sim_len=config["sim_len"]
    mu=config["mu"]
    sigma=config["sigma"]
    debt_ceiling=config["debt_ceiling"]
    liquidation_penalty=config["liquidation_penalty"]
    collateral_cutoff=config["collateral_cutoff"]
    jump_severities=config["jump_severities"]+[0]
    slippage_function=config["slippage_function"]
    recovery_time=config["recovery_time"]
    time_to_start_recovery=config["time_to_start_recovery"]
    reentry_time=config["reentry_time"]
    auction_efficiency=config["auction_efficiency"]

    jump_probabilities = [config["jump_probabilities"][0],(config["jump_probabilities"][0]+config["jump_probabilities"][1]),(config["jump_probabilities"][0]+config["jump_probabilities"][1]+config["jump_probabilities"][2]),100]
    cdp_distribution=[
        {"bucket":config["collateral_cutoff"]+bucket["buffer"],
        "collat":config["collateral_cutoff"]+bucket["buffer"],
        "debt":bucket["percentage"]*config["debt_ceiling"]*.01,
        "open":True,
        "re_entry_clock":0,
        "reversion_time":bucket["reversion_time"]}
        for bucket in config["cdp_buckets"]
    ]

    results=[]

    for iteration in range(num_simulations):
        result = run_iter(iteration = iteration, mu=mu,sigma = sigma,collateral_cutoff = collateral_cutoff,
            liquidation_penalty=liquidation_penalty,sim_len = sim_len,cdp_distribution=cdp_distribution,
            jump_probabilities=jump_probabilities,jump_severities=jump_severities,slippage_function=slippage_function,
            recovery_time=recovery_time,time_to_start_recovery=time_to_start_recovery,reentry_time=reentry_time,
            auction_efficiency=auction_efficiency)
        results+=result

    return(pd.DataFrame(results))

def aggregate_iterations(data):
    total = data.groupby(["iteration"]).sum()
    total["loss_gain_total"]=total["loss_gain"]

    avg = data.groupby(["iteration"]).mean()
    avg["avg_debt_supply"] = avg["debt_supply"]

    df = total.merge(avg["avg_debt_supply"],how="inner",on="iteration")
    df["loss_gain_perc"]=100*(df["loss_gain_total"]/df["avg_debt_supply"])
    return(df)

def aggregate_total():
    pass

def aggregate_results(data,config):
    df = aggregate_iterations(data)
    avg_losses=df["loss_gain_perc"].mean()

    worst_cutoff=config["worst_case_cutoff"]
    num_simulations=config["num_simulations"]
    sim_len=config["sim_len"]    
    n_smallest = data.nsmallest(max(1,int(num_simulations*sim_len*worst_cutoff)), columns=["loss_gain"],keep="all")    
    worst_losses=n_smallest["loss_gain"].max()

    aggregate = {}
    aggregate["avg_losses"]=avg_losses
    aggregate["worst_losses"]=worst_losses
    aggregate["num_sim"]=config["num_simulations"]
    aggregate["sim_len"]=config["sim_len"]
    aggregate["mu"]=config["mu"]
    aggregate["sigma"]=config["sigma"]
    aggregate["debt_ceiling"] = config["debt_ceiling"]
    aggregate["liquidation_penalty"]=config["liquidation_penalty"]
    aggregate["collateral_cutoff"]=config["collateral_cutoff"]
    aggregate["mu"]=config["mu"]
    aggregate["sigma"]=config["sigma"]
    aggregate["debt_ceiling"] = config["debt_ceiling"]
    aggregate["liquidation_penalty"]=config["liquidation_penalty"]
    aggregate["collateral_cutoff"]=config["collateral_cutoff"]
    aggregate["recovery_time"]=config["recovery_time"]
    aggregate["time_to_start_recovery"]=config["time_to_start_recovery"]
    aggregate["reentry_time"]=config["reentry_time"]
    aggregate["auction_efficiency"]=config["auction_efficiency"]
    aggregate["jump_severities_1"]=config["jump_severities"][0]
    aggregate["jump_severities_2"]=config["jump_severities"][1]
    aggregate["jump_severities_3"]=config["jump_severities"][2]
    aggregate["jump_probabilities_1"]=config["jump_probabilities"][0]
    aggregate["jump_probabilities_2"]=config["jump_probabilities"][0]+config["jump_probabilities"][1]
    aggregate["jump_probabilities_3"]=config["jump_probabilities"][0]+config["jump_probabilities"][1]+config["jump_probabilities"][2]

    print("num_sim",config["num_simulations"],"sim_len",config["sim_len"],"collat_req",config["collateral_cutoff"],"mu",config["mu"],"sigma",config["sigma"],"EC",config["economic_capital"],"slip_fun",config["slippage_function"],"debt_ceiling",config["debt_ceiling"])
    return(aggregate)

def graph_one(data):
    data = run_sim(config)
    df = aggregate_iterations(data)
    first = data[data["iteration"]==0]

    #display a graph of the colalteralization ratios of each bucket over time
    fig=Figure(data= [go.Scatter(x=first["time_step"],name=i,
                y=first[i]*100, mode='lines',line=dict(color="rgb("+str(255-25*float(i[14:]))+","+str(25*float(i[14:]))+",0)"))
           for i in first if "collat_bucket_" in i]+[go.Scatter(x=first["time_step"],
                y=first["asset_price"], mode='lines',line=dict(color="blue"),yaxis="y2",name="ETH Price")
           ],
    layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Collat %"),yaxis2=dict(title="$",overlaying="y",side="right"))
    )

    plot_url = py.plot(fig,filename="Eth Price vs. Collateralizations.html")

    #display a graph of liquidated debt price over time
    fig=Figure(data= [go.Scatter(x=first["time_step"],
                y=first["liquidated_debt"], mode='lines',line=dict(color="red"),name="Liquidated Debt")
           ]+[go.Scatter(x=first["time_step"],
                y=first["asset_price"], mode='lines',line=dict(color="blue"),yaxis="y2",name="ETH Price")
           ],
    layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Dai"),yaxis2=dict(title="$",overlaying="y",side="right"))
    )

    plot_url = py.plot(fig,filename="Eth Price vs. Liquidated Debt.html")

    #display a graph of the loss/gain over time
    fig=Figure(data= [go.Scatter(x=first["time_step"],
                y=first["loss_gain"], mode='lines',line=dict(color="red"),name="Loss/Gain")
           ]+[go.Scatter(x=first["time_step"],
                y=first["asset_price"], mode='lines',line=dict(color="blue"),yaxis="y2",name="ETH Price")
           ],
    layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Dai"),yaxis2=dict(title="$",overlaying="y",side="right"))
    )

    plot_url = py.plot(fig,filename="Eth Price vs. Loss-Gain.html")

    #display a graph of the loss/gain over time
    fig=Figure(data= [go.Scatter(x=first["time_step"],
                y=first["slippage_loss"], mode='lines',line=dict(color="red"),name="Slippage Loss")
           ]+[go.Scatter(x=first["time_step"],
                y=first["asset_price"], mode='lines',line=dict(color="blue"),yaxis="y2",name="ETH Price")
           ],
    layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Dai"),yaxis2=dict(title="$",overlaying="y",side="right"))
    )

    plot_url = py.plot(fig,filename="Eth Price vs. Slippage.html")

     #display a graph of the loss/gain over time
    fig=Figure(data= [go.Scatter(x=first["time_step"],
                y=first["undercollateralized_loss"], mode='lines',line=dict(color="red"),name="Undercollateralized Loss")
           ]+[go.Scatter(x=first["time_step"],
                y=first["asset_price"], mode='lines',line=dict(color="blue"),yaxis="y2",name="ETH Price")
           ],
    layout = go.Layout(xaxis=dict(title="Days"),yaxis=dict(title="Dai"),yaxis2=dict(title="$",overlaying="y",side="right"))
    )

    plot_url = py.plot(fig,filename="Eth Price vs. Undercollateralization.html")

    #display a histogram of average loss_gain_percentage
    fig = Figure(
        data=[go.Histogram(
            x=df["loss_gain_perc"],
            marker_color='#46C5B3',
            opacity=0.75
            )],layout=go.Layout(bargap=0.2,xaxis=dict(title="avg_loss_gain_perc (pts)"),yaxis=dict(title='num_iterations'))
    )    
    plot_url = py.plot(fig,filename="avg_loss_perc_pts.html")

    #display a histogram of the losses
    fig = Figure(
        data=[go.Histogram(
            x=data["loss_gain"],
            marker_color='#4675B2',
            opacity=0.75
            )],layout=go.Layout(bargap=0.2,xaxis=dict(title="nominal_loss_gain (daily)"),yaxis=dict(title="num_days"))
    )
    fig.update_layout(yaxis_type="log")
    plot_url = py.plot(fig,filename="daily_losses.html")

    return(0)

##################EDIT BELOW THIS LINE ONLY########################
result_array = []

with open("config.json") as infile:
    config_array=json.load(infile)

for config in config_array:
    data = run_sim(config)
    results = aggregate_results(data,config)
    result_array+=[results]
    worst_losses = results["worst_losses"]*-1

print(pd.DataFrame(result_array))
pd.DataFrame(result_array).to_csv("risk_model_results.csv")

#tell the program which simulation to graph (zero indexed)
graph_one(config_array[0])

#graph_scatter
df = pd.DataFrame(result_array)
fig = px.scatter_3d(df, x='collateral_cutoff', y='mu', z='sigma',color='avg_losses')
fig.show()