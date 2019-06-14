library(truncnorm)
mu <- 0.05 #annual drift in M
sigma <- 0.2 #annual volatility

t=1
dt=1/365

f <- M <- rep(0, times=t/dt) #fill the q,b,f, and M vectors with zeroes
M[1]=150

for(i in 2:(t/dt)){
  f[i] <- f[i-1]+sqrt(dt)*rnorm(1) #f simulates brownian motion
  M[i] <- M[1]*exp((mu-(sigma^2)/2)*(i*dt)+sigma*f[i]) #market cap calc based on GBM

}

par(mfrow=c(1,1))
plot(M)
summary(f)
rnorm(1)

x <- seq(-10,10,by = .2)
# Choose the mean as 2.5 and standard deviation as 2. 
y <- pnorm(x, mean = 2.5, sd = 3)
# Plot the graph.
plot(x,y)

coll_rat <- 5.2
cdps <- rnorm(500,(coll_rat+1.5)/2,.5)
plot(subset(cdps, cdps>1.5))
