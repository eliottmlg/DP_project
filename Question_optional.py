# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:22:33 2025

@author: eliot
"""


#==============================================================================
# Dynamic Programming - Project
# QUESTION OPTIONAL
#==============================================================================


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm
import torchsummary
from d2l import torch as d2l


fontsize= 14
ticksize = 14
figsize = (12, 4.5)
params = {'font.family':'serif',
    "figure.figsize":figsize,
    'figure.dpi': 80,
    'figure.edgecolor': 'k',
    'font.size': fontsize,
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': ticksize,
    'ytick.labelsize': ticksize
}
#plt.rcParams.update(params)


class Params(d2l.HyperParameters):
    def __init__(self,
                 alpha = 0.4,
                 beta = 0.9,
                 delta = 0.05,
                 h_0 = 5,
                 sigma = 0.9,
                ):
        self.save_hyperparameters()
        
        

def f(h, l):
    alpha = Params().alpha
    return h**alpha * l

def u_prime(c):
    sigma = Params().sigma
    out = c.pow(-sigma)
    return out



class SS: #steady state
    def __init__(self):
        self.delta = Params().delta
        self.beta = Params().beta
        self.alpha = Params().alpha
        self.h_ss = (self.alpha*self.beta)/(
            (1+self.alpha*self.beta)-self.beta*(1+self.alpha)*(1-self.delta)
            )
        self.l_ss = 1-self.delta*self.h_ss
        self.c_ss = f(self.h_ss,self.l_ss)
        

class Grid_data(d2l.HyperParameters):
    def __init__(self,
                 max_T = 32,
                 batch_size = 8
                 ):
        self.save_hyperparameters()
        self.time_range = torch.arange(0.0, self.max_T , 1.0)
        self.grid = self.time_range.unsqueeze(dim = 1)
        
        
        

class Data_label(Dataset):

    def __init__(self,data):
        self.time = data
        self.n_samples = self.time.shape[0]

    def __getitem__(self,index):
            return self.time[index]

    def __len__(self):
        return self.n_samples
    

train_data = Grid_data().grid
train_labeled = Data_label(train_data)
train = DataLoader(dataset = train_labeled, batch_size = 8 , shuffle = True )



class NN(nn.Module, d2l.HyperParameters):
    def __init__(self,
                 dim_hidden = 128,
                layers = 4,
                hidden_bias = True):
        super().__init__()
        self.save_hyperparameters()

        torch.manual_seed(123)
        module = []
        module.append(nn.Linear(1,self.dim_hidden, bias = self.hidden_bias))
        module.append(nn.Tanh())

        for i in range(self.layers-1):
            module.append(nn.Linear(self.dim_hidden,self.dim_hidden, bias = self.hidden_bias))
            module.append(nn.Tanh())

        module.append(nn.Linear(self.dim_hidden,2))
        module.append(nn.Softplus(beta = 1.0)) #The softplus layer ensures c>0,k>0

        self.q = nn.Sequential(*module)

   # def forward(self, x):
    #        out = self.q(x) # first element is consumption, the second element is capital
     #       l_t = torch.clamp(out[:, [0]], 0, 1)  # Clamp l_t to [0, 1]
      #      h_t = F.softplus(out[:, [1]])
       #     return torch.cat((l_t, h_t), dim=1)

    def forward(self, x):
        out = self.q(x) # first element is consumption, the second element is capital
        return  out
    
    
    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    


q_hat= NN()
learning_rate = 1e-3
optimizer = torch.optim.Adam(q_hat.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)


print(q_hat)

# Optimization of the network’s weights.

delta = Params().delta
beta = Params().beta
alpha = Params().alpha
h_0 = Params().h_0

num_epochs = 1001


# Torchsummary provides a more readable summary of the neural network
torchsummary.summary(q_hat, input_size=(1,))


for epoch in range(num_epochs):
    for i, time in enumerate(train):
        time_zero = torch.zeros([1,1])
        time_next = time+1
        time_2next = time_next+1
        l_t = q_hat(time)[:,[0]]
        h_t = q_hat(time)[:,[1]]
        l_tp1 = q_hat(time_next)[:,[0]]
        h_tp1 = q_hat(time_next)[:,[1]]
        h_tp2 = q_hat(time_2next)[:,[1]]
        h_t0 = q_hat(time_zero)[0,1]

 
        # Slackness conditions (penalty terms)
        slack_penalty = 0.0
        
        # Add penalties for l_t
        slack_penalty += torch.sum(torch.clamp(l_t - 1, min=0)**2)  # Penalize l_t > 1
        slack_penalty += torch.sum(torch.clamp(-l_t, min=0)**2)     # Penalize l_t < 0
        
        # Add penalties for l_tp1
        slack_penalty += torch.sum(torch.clamp(l_tp1 - 1, min=0)**2)  # Penalize l_tp1 > 1
        slack_penalty += torch.sum(torch.clamp(-l_tp1, min=0)**2)     # Penalize l_tp1 < 0
        
        # Total loss
        lambda_penalty = 1e3  # Coefficient for penalty strength
        
        res_1 = h_tp1 - (1-delta)*h_t - (1-l_t) # Law of motion of human capital
        res_2 = h_t**(alpha)*u_prime(f(h_t,l_t))/u_prime(f(h_tp1,l_tp1)) - beta*(
            (1+alpha)*(1-delta)*h_tp1**(alpha) + alpha*h_tp1**(alpha-1)*(1-h_tp2)
            ) # euler equation
        res_3 = h_t0-h_0 #Initial Condition
            
        # imposing slackness condition on euler equation
        loss_1 = res_1.pow(2).mean()
        loss_2 = res_2.pow(2).mean() + lambda_penalty * slack_penalty
        loss_3 = res_3.pow(2).mean() 
            
        loss = 0.1*loss_1+0.8*loss_2+0.1*loss_3

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    scheduler.step()

    if epoch == 0:
         print('epoch' , ',' , 'loss' , ',', 'loss_bc' , ',' , 'loss_euler' , ',' , 'loss_initial' ,
               ',', 'lr_rate')
    if epoch % 100 == 0:
          print(epoch,',',"{:.2e}".format(loss.detach().numpy()),',',
                "{:.2e}".format(loss_1.detach().numpy()) , ',' , "{:.2e}".format(loss_2.detach().numpy())
               , ',' , "{:.2e}".format(loss_3.detach().numpy()), ',', "{:.2e}".format(get_lr(optimizer)) )
          
          
          
          
          
### TEST WITHOUT SLACKNESS CONDITION ON LABOUR 


class NN(nn.Module, d2l.HyperParameters):
    def __init__(self,
                 dim_hidden = 128,
                layers = 4,
                hidden_bias = True):
        super().__init__()
        self.save_hyperparameters()

        torch.manual_seed(123)
        module = []
        module.append(nn.Linear(1,self.dim_hidden, bias = self.hidden_bias))
        module.append(nn.Tanh())

        for i in range(self.layers-1):
            module.append(nn.Linear(self.dim_hidden,self.dim_hidden, bias = self.hidden_bias))
            module.append(nn.Tanh())

        module.append(nn.Linear(self.dim_hidden,2))
        module.append(nn.Softplus(beta = 1.0)) #The softplus layer ensures c>0,k>0

        self.qq = nn.Sequential(*module)

   # def forward(self, x):
    #        out = self.q(x) # first element is consumption, the second element is capital
     #       l_t = torch.clamp(out[:, [0]], 0, 1)  # Clamp l_t to [0, 1]
      #      h_t = F.softplus(out[:, [1]])
       #     return torch.cat((l_t, h_t), dim=1)

    def forward(self, x):
        out = self.qq(x) # first element is consumption, the second element is capital
        return  out
    
    
    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    


qq_hat= NN()
learning_rate = 1e-3
optimizer = torch.optim.Adam(qq_hat.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)


print(qq_hat)

# Optimization of the network’s weights.

delta = Params().delta
beta = Params().beta
alpha = Params().alpha
h_0 = Params().h_0

num_epochs = 1501


# Torchsummary provides a more readable summary of the neural network
torchsummary.summary(qq_hat, input_size=(1,))


for epoch in range(num_epochs):
    for i, time in enumerate(train):
        time_zero = torch.zeros([1,1])
        time_next = time+1
        time_2next = time_next+1
        l_t = qq_hat(time)[:,[0]]
        h_t = qq_hat(time)[:,[1]]
        l_tp1 = qq_hat(time_next)[:,[0]]
        h_tp1 = qq_hat(time_next)[:,[1]]
        h_tp2 = qq_hat(time_2next)[:,[1]]
        h_t0 = qq_hat(time_zero)[0,1]

        l_t = torch.where(l_t >= 1.0, torch.tensor(1.0), l_t)
        l_t = torch.where(l_t <= 0.0, torch.tensor(0.0), l_t)
        l_tp1 = torch.where(l_tp1 >= 1.0, torch.tensor(1.0), l_tp1)
        l_tp1 = torch.where(l_tp1 <= 0.0, torch.tensor(0.0), l_tp1)

        res_1 = h_tp1 - (1-delta)*h_t - (1-l_t) # Law of motion of human capital
        res_2 = h_t**(alpha)*u_prime(f(h_t,l_t))/u_prime(f(h_tp1,l_tp1)) - beta*(
            (1+alpha)*(1-delta)*h_tp1**(alpha) + alpha*h_tp1**(alpha-1)*(1-h_tp2)
            ) # euler equation
        res_3 = h_t0-h_0 #Initial Condition
            
        # imposing slackness condition on euler equation
        loss_1 = res_1.pow(2).mean()
        loss_2 = res_2.pow(2).mean()
        loss_3 = res_3.pow(2).mean() 
            
        loss = 0.1*loss_1+0.8*loss_2+0.1*loss_3

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    scheduler.step()

    if epoch == 0:
         print('epoch' , ',' , 'loss' , ',', 'loss_bc' , ',' , 'loss_euler' , ',' , 'loss_initial' ,
               ',', 'lr_rate')
    if epoch % 100 == 0:
          print(epoch,',',"{:.2e}".format(loss.detach().numpy()),',',
                "{:.2e}".format(loss_1.detach().numpy()) , ',' , "{:.2e}".format(loss_2.detach().numpy())
               , ',' , "{:.2e}".format(loss_3.detach().numpy()), ',', "{:.2e}".format(get_lr(optimizer)) )
      
          
          
# Euler with slackness condition on labour         
time_test = Grid_data().grid
l_hat_path = q_hat(time_test)[:,[0]].detach()
h_hat_path = q_hat(time_test)[:,[1]].detach()
c_hat_path = f(h_hat_path, l_hat_path)
# without
l_hat_path2 = qq_hat(time_test)[:,[0]].detach()
h_hat_path2 = qq_hat(time_test)[:,[1]].detach()
c_hat_path2 = f(h_hat_path2, l_hat_path2)



plt.figure(figsize=(9, 5)) 
plt.plot(time_test, h_hat_path, label=r"human capital path -- slackness condition")
plt.plot(time_test, h_hat_path2, label=r"human capital path -- unconstrained")
plt.axhline(y=SS().h_ss, linestyle='--', label="h Steady State")
plt.ylabel(r"h(t)")
plt.xlabel(r"Time(t)")
#plt.ylim([Params().h_0 - 0.1, SS().h_ss + 0.1])
plt.legend(loc='upper right')
plt.show()


plt.figure(figsize=(9, 5))  # Set figure size to 9 by 5
plt.plot(time_test,l_hat_path,label= r"labour supply path -- slackness condition")
plt.plot(time_test,l_hat_path2,label= r"labour supply path -- unconstrained")
plt.axhline(y=SS().l_ss, linestyle='--',label="l Steady State")
plt.xlabel(r"Time(t)")
plt.ylabel(r"l(t)")
#plt.ylim([l_hat_path[0]-0.1,SS().l_ss+0.1 ])
plt.tight_layout()
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(9, 5))  # Set figure size to 9 by 5
plt.plot(time_test,c_hat_path,label= r"consumption path -- slackness condition")
plt.plot(time_test,c_hat_path2,label= r"consumption path -- unconstrained")
plt.axhline(y=SS().c_ss, linestyle='--',label="c Steady State")
plt.xlabel(r"Time(t)")
plt.ylabel(r"c(t)")
#plt.ylim([c_hat_path[0]-0.1,SS().c_ss+0.1 ])
plt.tight_layout()
plt.legend(loc='upper right')
plt.show()

