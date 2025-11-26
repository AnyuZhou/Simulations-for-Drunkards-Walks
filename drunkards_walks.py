# -*- coding: utf-8 -*-
import numpy as np
import random
import os

NT = 10000 #travelling time ~5h
V0 = 1.0 #average velocity ~km/h
count = 10000
theta = np.zeros([count,NT]) #direction
tt = np.arange(NT)  #time
x = np.zeros([count,NT]) #position-time
y = np.zeros([count,NT])
memory = np.zeros([count,NT])
eta_label = [1.0,0.5,0.25,0.0625] #degree of drunkenness
#lam_label = [0.02,0.1,0.5]
    
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print(path+' success')
        return True
    else:
        print(path+' dir exist')
        return False
    
def scale_theta(theta):
    theta = np.mod((theta + np.pi),2*np.pi)-np.pi
    return theta

def initialposvel(trace_id): #初始化
    x[trace_id,0] = 0
    y[trace_id,0] = 0
    drift0 = random.uniform(-1,1)*eta*np.pi
    theta[trace_id,0] = drift0
    memory[trace_id,0] = drift0
#    drift[:] = 0
#    drift[0] = random.uniform(-1,1)*eta*np.pi
#    theta[trace_id,0] = drift[0]
#    memory[trace_id,0] = drift[0]
""" 
#simulations with limited memory time in supplementary 
tau = 100    
drift = np.zeros(NT)
def timevolution(trace_id): 
    for it in np.arange(NT-1): 
        #if np.mod(it,2000) == 0:
        #   print('it=',it)
        drift[it+1] = random.uniform(-1,1)*eta*np.pi-lam*memory[trace_id][it]
        if it>=tau:
            memory[trace_id][it+1] = memory[trace_id][it] + drift[it+1] - drift[it-tau]
        else:
            memory[trace_id][it+1] = memory[trace_id][it] + drift[it+1]
        theta[trace_id][it+1] = scale_theta(theta[trace_id][it] + drift[it+1]) 
        x[trace_id][it+1] = x[trace_id][it] + np.cos(theta[trace_id][it]) * V0 #存储位置
        y[trace_id][it+1] = y[trace_id][it] + np.sin(theta[trace_id][it]) * V0
"""
def timevolution(trace_id): 
    for it in np.arange(NT-1): 
        #if np.mod(it,2000) == 0:
        #   print('it=',it)
        drift = random.uniform(-1,1)*eta*np.pi-lam*memory[trace_id][it]
        memory[trace_id][it+1] = memory[trace_id][it] + drift
        theta[trace_id][it+1] = scale_theta(theta[trace_id][it] + drift) 
        x[trace_id][it+1] = x[trace_id][it] + np.cos(theta[trace_id][it]) * V0 
        y[trace_id][it+1] = y[trace_id][it] + np.sin(theta[trace_id][it]) * V0


#without memory
for eta in eta_label:
    lam = 0
    path = f''
    mkdir(path)
    for i in range(count):
        if np.mod(i,500)==0:
            np.save(path+f'/x.npy',x)
            np.save(path+f'/y.npy',y)
            np.save(path+f'/theta.npy',theta)
        if np.mod(i,1000)==0:
            print('i=',i)
        initialposvel(i) 
        timevolution(i)
    np.save(path+f'/x.npy',x)
    np.save(path+f'/y.npy',y)
    np.save(path+f'/theta.npy',theta)

#with memory
for eta in eta_label:
    lam = 1-eta
    path = f''
    mkdir(path)
    for i in range(count):
        if np.mod(i,500)==0:
            np.save(path+f'/x.npy',x)
            np.save(path+f'/y.npy',y)
            np.save(path+f'/theta.npy',theta)
        if np.mod(i,1000)==0:
            print('i=',i)
        initialposvel(i) 
        timevolution(i)
    np.save(path+f'/x.npy',x)
    np.save(path+f'/y.npy',y)
    np.save(path+f'/theta.npy',theta)