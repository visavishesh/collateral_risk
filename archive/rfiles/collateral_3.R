           

#set.seed(25)
mu <- 0.05 #annual drift in M
sigma <- 0.4 #annual volatility

t=1
dt=1/365

f <- M <- rep(0, times=t/dt) #fill the q,b,f, and M vectors with zeroes
M[1]=150

jump_risk <- 1/730
jump_severity <- .4 

for(i in 2:(t/dt)){
  if (sample(seq(1,1/jump_risk), 1, replace = FALSE, prob = NULL)==1){
    f[i] <- f[i-1]-jump_severity/2 #f simulates brownian motion
  }else{
    f[i] <- f[i-1]+sqrt(dt)*rnorm(1) #f simulates brownian motion
  }
  M[i] <- M[1]*exp((mu-(sigma^2)/2)*(i*dt)+sigma*f[i]) #market cap calc based on GBM
  
}

par(mfrow=c(1,1))
plot(f)
plot(M)
plot(shift(diff(M), n=1L,fill=0)/M)