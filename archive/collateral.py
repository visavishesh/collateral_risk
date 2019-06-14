import pandas as pd
from math import *

assets = {"asset1":{"liquidity":1,"ratio":1.5,"mu":1.05,"sigma":.2},
			"asset2":{"liquidity":.8,"ratio":1.25,"mu":1.05,"sigma":.2},
			"asset3":{"liquidity":.5,"ratio":1.5,"mu":1.05,"sigma":.2} } 

import numpy as np
import matplotlib.pyplot as plt

seed = 5       
N  = 2.**6     # increments
dt=1/365
M0=1000
#np.random.seed(seed)
mu=.05
sigma=.2
x= pd.DataFrame()
x["dt"] = pd.Series(range(1,int(1/dt)))
x["f"] = list(map(lambda x:sqrt(1/x)*np.random.normal(1),x["dt"]))
x["M"] = x.apply(lambda row: (x["dt"]+sigma*x['f']), axis=1)
print(x)