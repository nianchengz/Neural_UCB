#import tensorflow as tf
import numpy as np
import torch
import math
from torch import nn

class Bandit:
    def __init__(self, alpha = 10, regulation_parameter = 1 ,exporation_parameter = 0.01 , confidence_parameter = 1, norm_parameter = 1, step_size = 20, gradient_num = 100, network_width = 2, network_depth = 100):
        self.T = alpha
        self.rp = regulation_parameter
        self.v = exporation_parameter
        self.confi = confidence_parameter
        self.s = norm_parameter
        self.n = step_size
        self.J = gradient_num
        self.m = network_width
        self.L = network_depth
        self.gama = positive_scaling_factor

        self.Z = [[[0]*10 for i in range(10)] for j in range(10)]
        for i in range(10):
            for j in range(10):
                if i==j:
                    self.Z[0][i][j] = self.rp
                else:
                    self.Z[0][i][j] = 0

net = nn.Sequential(
    nn.Linear(1, 100),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1),
    nn.sigmoid()
    )  
                  
T = 10
K = 10
U = []
for t in range(T-1):   #run t times
    MAX = 0
    for a in range(K-1):    #has a arms
          
        X = [[0]*10 for i in range(10)]
        for i in range(10):
            for j in range(10):
                k1 = np.random.normal(0, 0.1, 1)
                k2 = np.random.normal(0, 0.1, 1)
                k1 = list(k1)
                k2 = list(k2)
                X[i][j] = torch.tensor([k1[0], k2[0]]) #context [1*2]
   
        f_output = net(X[t][a])
        U[a] = f_output + #.....line 7
        #gardient????

        if (U[a] > U[MAX]) || (a == 0):
            MAX = a
    Z[t+1] = Z[t] + #line 11

gama = math.pow(1 + C1 * (m)^(-1/6) * math.pow(math.log(x[, m]), 1/2) * math.pow(L, 4) * math.pow(t, 7/6) * math.pow(rp, -7/6), 1/2)\
    * (v * math.log(np.linalg.det(Z[t])/np.linalg.det(rp)) + C2*math.pow(m,-1/6)*math.pow(math.log(m),1/2)*math.pow(L, 4)*math.pow(t, 5/3)*math.pow(rp, -1/6) - 2*math.log(confi) + math.pow(rp, 1/2)*s)\
    + (rp + C3*t*L)\
    * (math.pow(1-n*m*rp, 1/2) * math.pow(t/rp, 1/2) + math.pow(m, -1/6)*math.pow(math.log(m),1/2)*math.pow(L, 7/2)*math.pow(t, 5/3)*math.pow(rp, -5/3)*(1+math.pow(t/rp, 1/2)))



