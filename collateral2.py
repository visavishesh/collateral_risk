import numpy as np
import math
import matplotlib.pyplot as plt

t=5
dt=1/365
jump_risk=730
jump=.4
f = [0 for i in range(int(t/dt))]
M = [0 for i in range(int(t/dt))]
liq_debt = [0 for i in range(int(t/dt))]
M[0]=100
mu=0.05
sigma=.2
collateral_cutoff=1.5

def update_collat(cdps,M2,M1):
	cdps = [{"collat":cdp["collat"]*(M2/M1),"debt":cdp["debt"],"open":cdp["collat"]*(M2/M1)>1.5} for cdp in cdps if cdp["open"]==True]
	return(cdps)

cdps = [{"collat":1.55,"debt":10000000,"open":True},{"collat":2,"debt":10000000,"open":True},{"collat":2.5,"debt":10000000,"open":True}]

revert=0
#cdps = update_collat(cdps)
for i in range(1,int(t/dt)):
	if revert>0:
			f[i] = jump/sigma
			M[i] = M[0]*math.exp((mu-(math.pow(sigma,2))/2)*(i*dt)+sigma*f[i]) #market cap calc based on GBM
			revert-=.25
	else:
		if np.random.randint(0,jump_risk-1)==1:
			f[i] = -1*jump/sigma
			M[i] = M[0]*math.exp((mu-(math.pow(sigma,2))/2)*(i*dt)+sigma*f[i]) #market cap calc based on GBM
			revert=1		
		else:
			f[i] = f[i-1]+math.sqrt(dt)*np.random.normal(0,1,1)
			M[i] = M[0]*math.exp((mu-(math.pow(sigma,2))/2)*(i*dt)+sigma*f[i]) #market cap calc based on GBM
	cdps = update_collat(cdps,M[i],M[i-1])
	liq_debt[i] = sum([c["debt"] for c in cdps if c["open"]==False])

delta = [(M[n]-M[n-1])/M[n-1] for n in range(1,len(M))]


fig, ax1 = plt.subplots()
ax1.set_xlabel('Days')
ax1.set_ylabel('Dai Liquidated')
ax1.plot(liq_debt, color='tab:red')
ax2 = ax1.twinx()
ax2.set_ylabel('ETH Price')
ax2.plot(M, color='tab:blue')
plt.show()