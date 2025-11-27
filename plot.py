# -*- coding: utf-8 -*-
import numpy as np

def load_data(path):
    x = np.load(path+f'/x.npy')
    y = np.load(path+f'/y.npy')
    theta = np.load(path+f'/theta.npy')
    return x,y,theta
NN = 10000
NT = 10000
path =''

#Fig.3
x,y,theta = load_data(path)
#average position
xmean = np.mean(x,axis=0)
#mean square displacement
MSD = np.mean(x**2+y**2,axis=0)


#Fig.4
#global displacement direction

phi = np.zeros([NN,101])

count = 0
for eta in np.linspace(0,1,101):
    path = f'eta={round(eta,2)}'
    x,y,theta = load_data(path)
    phi[:,count] = np.arctan2(y[:,-1],x[:,-1])/np.pi*180
    count += 1



#arrive probility
time_arrive_target = np.zeros([101,1000,NN])
count = 0
for eta in np.linspace(0,1,101):
    path = f'eta={round(eta,2)}'
    x = np.load(path+'/x.npy')
    y = np.load(path+'/y.npy')
    for ix in np.arange(1000):
        distance = np.sqrt((x-ix*10)**2+y**2)
        for i in range(NN):
            reachtime = np.where(distance[i]<=10)[0]
            if len(reachtime):
                time_arrive_target[count][ix][i] = np.min(reachtime)
            else:
                time_arrive_target[count][ix][i] = np.inf
    count += 1    

x_d = np.arange(0,10000,10)
y_eta = np.arange(0,1.01,0.01)

p = np.zeros([101,1000])
for i in range(101):
    for j in range(1000):
        p[i][j] = len(np.where(time_arrive_target[i][j]<10000)[0])/NN

critical_eta = np.zeros(100)
for i in range(100):
    critical_eta[i] = np.where(p[:,i*10]>1e-2)[0][-1]*0.01


#Fig.S1-2
#correlation
for it in range(NT):
    correlation[:,it] = np.mean(np.cos(theta[:,0]-theta[:,it]),axis=1)

#Fig.S6
NN_label = [1,5,10,20,50]
p = np.zeros(1000)
path = f'nomemory/noleader'
for i_NN in range(5):
    NN = NN_label[i_NN]
    print('N=',NN)
    x = np.load(path+f'/N={NN}/x.npy').reshape(-1,10000)
    y = np.load(path+f'/N={NN}/y.npy').reshape(-1,10000)
    for ix in np.arange(0,1000):
        if np.mod(ix,100)==0:
            print('distance=',ix)
        p[ix] = len(np.where(np.min(np.sqrt((x-ix*10)**2+y**2),axis=1)<=10)[0])/20000    


#The other calculations for average positions,mean square displacement,probability and so on are similar to above given codes.