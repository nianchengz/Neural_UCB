#import tensorflow as tf
import numpy as np
import torch
import math
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt

net = nn.Sequential(
                    nn.Linear(2, 60),
                    #通常會加入一個 non-linear layer
                    nn.Linear(60, 60),
                    nn.ReLU(),
                    nn.Linear(60, 1),
                    nn.Sigmoid()
                    )

class Bandit:
    def __init__(self, alpha = 10, regulation_parameter = 1 ,exporation_parameter = 0.01 , confidence_parameter = 1, norm_parameter = 1, step_size = 20, gradient_num = 100, network_width = 2, network_depth = 100 ):
        
        self.T = alpha
        self.rp = regulation_parameter
        self.v = exporation_parameter
        self.confi = confidence_parameter
        self.s = norm_parameter
        self.n = step_size
        self.J = gradient_num
        self.m = network_width
        self.L = network_depth
        self.gama = 0.1
        self.optimizer = optim.Adam(net.parameters()) #?????
        self.C1 = 0.1
        self.C2 = 0.1
        self.C3 = 0.1
        
        z = (3780, 3780, 3780)
        self.Z = np.zeros(z)
        for i in range(3780):
            for j in range(3780):
                if i==j:
                    self.Z[0][i][j] = self.rp
                    
        self.NeuralUCB()
   
    def TrainNN(self, J, t, f_output_result, r, m, rp):
        MSE = 0 
        loss_MSE = []
        loss_mrp = []
        l2_reg = torch.tensor(0)
        for j in range(J-1):
            for param in net.parameters():
                l2_reg += torch.norm(param).type_as(l2_reg)
            for i in range(t+1):
                MSE = np.power((f_output_result - r[i][0][0]), 2)/2
                MSE = torch.from_numpy(MSE)
            loss_MSE.append(MSE)
            loss_mrp.append(m*rp*l2_reg/2)
            #loss = Variable(loss)
            self.optimizer.zero_grad()
            loss = torch.stack(loss_MSE).sum() + torch.stack(loss_mrp).sum()
            loss = Variable(loss, requires_grad=True)
            loss.backward()
            self.optimizer.step()
           
    
    def NeuralUCB(self):
        T = 2000
        K = 10
        U = [0]*K
        self.regret = [0]*T
        for t in range(T):   #run t times
            f_MAX = 0
            MAX = 0
            for a in range(K):    #has K arms
                X = [[0]*K for i in range(T)]
                self.r = [[0]*K for i in range(T)]
                
                #create context
                for i in range(T):
                    for j in range(K):
                        k1 = np.random.normal(0, 0.1, 1)
                        k2 = np.random.normal(0, 0.1, 1)
                        k1 = list(k1)
                        k2 = list(k2)
                        X[i][j] = np.array([[k1[0], k2[0]]]) #context [1*2]
                        phi = np.array([[k1[0]], [k2[0]]]) #context [1*2]
                        
                        self.r[i][j] = np.dot(X[i][j], phi) + np.random.normal(0, 0.1, 1)
                self.f_output = [0]*K
                self.f_output[a] = net(torch.from_numpy(X[t][a]).float())
                print(self.f_output[a])
                if (self.f_output[a] > f_MAX) :
                     f_MAX = self.f_output[a]
                self.optimizer.zero_grad()
                self.f_output[a].backward()
                
                gradient = np.concatenate((np.array(net[0].weight.grad.numpy().flatten()), np.array(net[1].weight.grad.numpy().flatten()), np.array(net[3].weight.grad.numpy().flatten())))
                gradient = np.array(gradient).reshape(1, -1)
                gradient_t = gradient.transpose()
                gradient_result = np.dot(gradient, np.linalg.inv(self.Z[t]))
                
                haha = gradient * np.linalg.inv(self.Z[t]) * gradient_t
                product1 = np.dot(gradient, np.linalg.inv(self.Z[t]))
                product2 = np.dot(product1, gradient_t)
                
                U[a] = self.f_output[a] + self.gama * math.pow(product2/self.m, 1/2)
                
                if (U[a] > U[MAX]) or (a == 0): #choose best arm by choosing best UCB
                     MAX = a
                     self.r[t][0] = self.r[t][a]
            self.regret[t] = f_MAX - self.f_output[MAX]  #best(from now) minus our choice
            self.f_output_result = self.f_output[MAX]
            self.Z[t+1] = self.Z[t] + gradient * np.transpose(gradient)/(self.m)
            
            self.TrainNN(self.J, t, self.f_output_result, self.r, self.m, self.rp)
            
            self.gama = np.power( 1 + self.C1 * np.power(self.m, -1/6) * np.power(np.math.log(self.m, 10) * np.power((self.rp), -7/6), 1/2)\
                * (self.v * np.power(np.math.log(np.linalg.det(self.Z[t])/np.power(self.rp, 990), 10) + (self.C2) * np.power(self.m,-1/6)\
                * np.power(np.math.log(self.m, 10),1/2) * np.power(self.L, 4) * np.power(self.T, 5/3) * np.power(self.rp, -1/6)\
                - 2*np.math.log(self.confi), 1/2) + np.power(self.rp, 1/2)*self.s)\
                + (self.rp + self.C3 * self.T * self.L) * (np.power(1-self.n*self.m*self.rp, self.J/2) * np.power(self.T/self.rp, 1/2) + np.power(self.m, -1/6)\
                *np.power(np.math.log(self.m, 10),1/2)*np.power(self.L, 7/2)*np.power(self.T, 5/3)*np.power(self.rp, -5/3)*(1+np.power(self.T/self.rp, 1/2))), 1/2)
            print('times: ', t)
        cumsum = np.cumsum(self.regret)
        print('cumsum: ', cumsum[i][0])

        plt.plot(cumsum)
        plt.show()

Bandit()
        