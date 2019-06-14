import numpy as np
import math
import matplotlib.pyplot as plt

t=5
dt=1/365
jump_risk=730
jump=-.4
f = [0 for i in range(int(t/dt))]
M = [0 for i in range(int(t/dt))]
M[0]=100
mu=0.05
sigma=.2

for i in range(1,int(t/dt)):
	if np.random.randint(0,jump_risk-1)==1:
		f[i] = jump/sigma
		M[i] = M[0]*math.exp((mu-(math.pow(sigma,2))/2)*(i*dt)+sigma*f[i]) #market cap calc based on GBM
		i+=1
		f[i] = -1*jump/sigma
		M[i] = M[0]*math.exp((mu-(math.pow(sigma,2))/2)*(i*dt)+sigma*f[i]) #market cap calc based on GBM
	else:
		f[i] = f[i-1]+math.sqrt(dt)*np.random.normal(0,1,1)
		M[i] = M[0]*math.exp((mu-(math.pow(sigma,2))/2)*(i*dt)+sigma*f[i]) #market cap calc based on GBM

delta = [(M[n]-M[n-1])/M[n-1] for n in range(1,len(M))]

fig, ax1 = plt.subplots()
ax1.set_xlabel('T')
ax1.set_ylabel('% Drift')
ax1.plot(delta, color='tab:red')
ax2 = ax1.twinx()
ax2.set_ylabel('ETH Price')
ax2.plot(M, color='tab:blue')
plt.show()