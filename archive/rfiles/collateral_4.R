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
    f[i] <- f[i-1]-(jump_severity*2) #f simulates brownian motion
  }else{
    f[i] <- f[i-1]+sqrt(dt)*rnorm(1) #f simulates brownian motion
  }
  M[i] <- M[1]*exp((mu-(sigma^2)/2)*(i*dt)+sigma*f[i]) #market cap calc based on GBM
  
}

drift <- cumsum((shift(diff(M), n=1L,fill=0)/M))

cdps <- matrix( c(500, 300, 200, 3, 3, 3), nrow=3, ncol=2) 
cdps[,1]

par(mfrow=c(1,1))
plot(f)
plot(shift(diff(M), n=1L,fill=0)/M)
plot(M)

ornstein_uhlenbeck <- function(T,n,nu,lambda,sigma,x0){
  dw  <- rnorm(n, 0, sqrt(T/n))
  dt  <- T/n
  x <- c(x0)
  for (i in 2:(n+1)) {
    x[i]  <-  x[i-1] + lambda*(nu-x[i-1])*dt + sigma*dw[i-1]
  }
  return(x);
}
plot(ornstein_uhlenbeck(10,1000,1,3,.2,0))