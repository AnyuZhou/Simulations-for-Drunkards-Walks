# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import matplotlib.pylab as plt
import scipy.stats as stats
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
#import matplotlib.animation as animation
#cycle = 10
#Natom = 100 #原子数
NT = 10000 #最大时间步数 对应5h
V0 = 1.0 #平均速度km/h
count = 5000
tt = np.arange(NT)  #时间
alpha = 1.0 #排列强度
rcut = 20
#eta_label = [0.5,0.25,0.125,0.0625] #醉酒的程度 从0到1 清醒到烂醉
#lam_label = [0.02,0.1,0.5]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',   # 使用颜色编码定义颜色
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
def load_data(path):
    x = np.load(path+f'/x.npy')
    y = np.load(path+f'/y.npy')
    theta = np.load(path+f'/theta.npy')
    return x,y,theta
    
def mkdir(path):
    # 引入模块
 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
 
        print(path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False

   
def scale_theta(theta):
    theta = np.mod((theta + np.pi),2*np.pi)-np.pi
    return theta


def initialposvel_leader(trace_id): #初始化
    x[trace_id,:,0] = 0
    y[trace_id,:,0] = 0
    for i in range(Nleader,NN):
        drift0 = random.uniform(-1,1)*eta*np.pi
        theta[trace_id,i,0] = drift0
        memory[trace_id,i,0] = drift0


def initialposvel(trace_id): #初始化
    x[trace_id,:,0] = 0
    y[trace_id,:,0] = 0
    for i in range(NN):
        drift0 = random.uniform(-1,1)*eta*np.pi
        theta[trace_id,i,0] = drift0
        memory[trace_id,i,0] = drift0

def align(trace_id,t): #计算力
    r2cut = rcut**2 #截断距离平方
    cos_neighbor = np.zeros(NN)
    sin_neighbor = np.zeros(NN)
    alg = np.zeros([NN])
    for i in range(0,NN-1):
        for j in range(i+1,NN):
            r2 = (x[trace_id][j][t]-x[trace_id][i][t])**2 + (y[trace_id][j][t]-y[trace_id][i][t])**2
            if(r2 < r2cut): #截断
                cos_neighbor[i] += np.cos(theta[trace_id][j][t])
                cos_neighbor[j] += np.cos(theta[trace_id][i][t])
                sin_neighbor[i] += np.sin(theta[trace_id][j][t])
                sin_neighbor[j] += np.sin(theta[trace_id][i][t])

    for i in range(NN):
        if (cos_neighbor[i]==0)and(sin_neighbor[i]==0):
            alg[i] = 0
        else:
            alg[i] = scale_theta(np.arctan2(sin_neighbor[i],cos_neighbor[i])-theta[trace_id][i][t])*alpha #arctan2(dy,dx)
    return(alg)


"""
def attract(trace_id,t): #计算力
    r2cut = rcut**2 #截断距离平方
    view = np.pi/3
    atr = np.zeros([NN])
    for i in range(NN):
        visible = np.zeros(NN) != 0
        for j in range(NN):
            if j==i:
                continue
            dx = x[trace_id][j][t]-x[trace_id][i][t]
            dy = y[trace_id][j][t]-y[trace_id][i][t]
            r2 = dx*dx + dy*dy
            if(r2 < r2cut): #截断
                phi = scale_theta(np.arctan2(dy,dx)-theta[trace_id][i][t])
                if np.abs(phi) <= view:
                    visible[j] = True
        if len(np.where(visible)[0]):
            target_x = np.mean(x[trace_id,visible,t])
            target_y = np.mean(y[trace_id,visible,t])
            atr[i] = np.sin(np.arctan2(target_y,target_x)-theta[trace_id][i][t])*alpha #arctan2(dy,dx)
        else:
            atr[i] = 0
    return(atr)

"""

def timevolution_leader(trace_id): #时间演化
    for it in np.arange(NT-1): #时间循环
        #if np.mod(it,2000) == 0:
        #   print('it=',it)
        alg = align(trace_id,it)
        for i in range(Nleader,NN):    
            drift = random.uniform(-1,1)*eta*np.pi-lam*memory[trace_id][i][it]+alg[i]
            memory[trace_id][i][it+1] = memory[trace_id][i][it] + drift
            theta[trace_id][i][it+1] = scale_theta(theta[trace_id][i][it] + drift) 
            x[trace_id][i][it+1] = x[trace_id][i][it] + np.cos(theta[trace_id][i][it]) * V0 #存储位置
            y[trace_id][i][it+1] = y[trace_id][i][it] + np.sin(theta[trace_id][i][it]) * V0
        if np.max(x[trace_id,Nleader:NN,it+1])>np.min(x[trace_id,0:Nleader,it])-10:
            x[trace_id,0:Nleader,it+1] = x[trace_id,0:Nleader,it] + V0 #存储位置
        else:
            x[trace_id,0:Nleader,it+1] = x[trace_id,0:Nleader,it]

def timevolution(trace_id): #时间演化
    for it in np.arange(NT-1): #时间循环
        #if np.mod(it,5000) == 4999:
        #   print('it=',it)
        alg = align(trace_id,it)
        for i in range(NN):    
            drift = random.uniform(-1,1)*eta*np.pi-lam*memory[trace_id][i][it]+alg[i]
            memory[trace_id][i][it+1] = memory[trace_id][i][it] + drift
            theta[trace_id][i][it+1] = scale_theta(theta[trace_id][i][it] + drift) 
            x[trace_id][i][it+1] = x[trace_id][i][it] + np.cos(theta[trace_id][i][it]) * V0 #存储位置
            y[trace_id][i][it+1] = y[trace_id][i][it] + np.sin(theta[trace_id][i][it]) * V0

"""
eta = 0.3
lam = np.exp(-7*eta)
#lam = 0
#NN_label= [1,3,4,5,10,20,50]

NN_label= [2,3,4]

for id_NN in np.arange(3):
    NN = NN_label[id_NN]
    Nleader = 1
    count = int(5000/NN)
    path = f'D:/zay/2023/0924/align/leader_wait/N={NN}'
    NN += Nleader
    theta = np.zeros([count,NN,NT]) #角度
    x = np.zeros([count,NN,NT]) #位置-时间
    y = np.zeros([count,NN,NT])
    memory = np.zeros([count,NN,NT])
    mkdir(path)
    for i in range(count): 
        if np.mod(i,100)==0:
            print('trace=',i)
            np.save(path+f'/x.npy',x)
            np.save(path+f'/y.npy',y)
            np.save(path+f'/theta.npy',theta)
        initialposvel_leader(i) #调用初始化
        timevolution_leader(i)
    np.save(path+f'/x.npy',x)
    np.save(path+f'/y.npy',y)
    np.save(path+f'/theta.npy',theta)
"""
"""

for id_NN in np.arange(4,5):
    NN = NN_label[id_NN]
    Nleader = 1
    count = int(20000/NN)
    path = f'D:/zay/2023/1210/Vicsek/leader_wait/N={NN}_2'
    NN += Nleader
    theta = np.zeros([count,NN,NT]) #角度
    x = np.zeros([count,NN,NT]) #位置-时间 
    y = np.zeros([count,NN,NT])
    memory = np.zeros([count,NN,NT])
    mkdir(path)
    for i in range(count): 
        if np.mod(i,50)==0:
            print('trace=',i)
            np.save(path+f'/x.npy',x)
            np.save(path+f'/y.npy',y)
            np.save(path+f'/theta.npy',theta)
        initialposvel_leader(i) #调用初始化
        timevolution_leader(i)
    np.save(path+f'/x.npy',x)
    np.save(path+f'/y.npy',y)
    np.save(path+f'/theta.npy',theta)

"""
"""
NN_label = [1,5,10,20,50]
p_50 = np.zeros(1000)
path = f'D:/zay/2023/1210/Vicsek/noleader'
p = np.load(f'D:/zay/2023/1210/Vicsek_p_align_noleader.npy')
for i_NN in range(4,5):
    NN = NN_label[i_NN]
    print('N=',NN)
    x = np.load(path+f'/N={NN}_2/x.npy').reshape(-1,10000)
    y = np.load(path+f'/N={NN}_2/y.npy').reshape(-1,10000)
    for ix in np.arange(0,1000):
        if np.mod(ix,100)==0:
            print('distance=',ix)
        p_50[ix] = len(np.where(np.min(np.sqrt((x-ix*10)**2+y**2),axis=1)<=10)[0])/20000
p[4] =  p[4]/3 + p_50*2/3
np.save(f'E:/zay/分子模拟/人工鱼群/data/2023/1217/Vicsek_p_align_noleader.npy',p)
p = np.load(f'D:/zay/2023/1210/Vicsek_p_align_leader_wait.npy')
path = f'D:/zay/2023/1210/Vicsek/leader_wait'
for i_NN in range(4,5):
    NN = NN_label[i_NN]
    print('N=',NN)
    x = np.load(path+f'/N={NN}_2/x.npy').reshape(-1,10000)
    y = np.load(path+f'/N={NN}_2/y.npy').reshape(-1,10000)
    for ix in np.arange(0,1000):
        if np.mod(ix,100)==0:
            print('distance=',ix)
        p_50[ix] = len(np.where(np.min(np.sqrt((x-ix*10)**2+y**2),axis=1)<=10)[0])/20000-1/NN
p[4] =  p[4]/3 + p_50*2/3
np.save(f'E:/zay/分子模拟/人工鱼群/data/2023/1217/Vicsek_p_align_leader_wait.npy',p)


p = np.load(f'E:/zay/分子模拟/人工鱼群/data/2023/1217/Vicsek_p_align_leader_wait.npy')
p_smooth = np.zeros(1000)
fig = plt.figure(figsize=(8,6))
for i in range(1,5):
    NN = NN_label[i]
    p_smooth[:11]=savgol_filter(p[i][:11],17, 3, mode='nearest')
    p_smooth[10:]=savgol_filter(p[i][10:],53, 3, mode='nearest')
    plt.plot([1000],p_smooth[100],color=colors3[i],label=f'N={NN_label[i]} & 1leader',linewidth=3.0,marker=markers[i],alpha=0.8)
    plt.plot(np.arange(1000)*10,p_smooth,color=colors3[i],linewidth=3.0,alpha=0.8)
    plt.scatter(np.arange(10)*1000,p_smooth[::100],color=colors3[i],marker=markers[i],s=70,alpha=0.8)

plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower right')
plt.show()
fig.savefig(f'E:\\zay\\分子模拟\\人工鱼群\\data\\2024\\0317\\Vicsek_probability_align_wait_leader.png', dpi=200, bbox_inches='tight')
p = np.load(f'E:/zay/分子模拟/人工鱼群/data/2023/1217/Vicsek_p_align_noleader.npy')
p_smooth = np.zeros(1000)
fig = plt.figure(figsize=(8,6))
for i in range(1,5):
    NN = NN_label[i]
    p_smooth[:11]=savgol_filter(p[i][:11],17, 3, mode='nearest')
    p_smooth[10:]=savgol_filter(p[i][10:],53, 3, mode='nearest')
    plt.plot([1000],p_smooth[100],color=colors3[i],label=f'N={NN_label[i]}',linewidth=3.0,marker=markers[i],alpha=0.8)
    plt.plot(np.arange(1000)*10,p_smooth,color=colors3[i],linewidth=3.0,alpha=0.8)
    plt.scatter(np.arange(10)*1000,p_smooth[::100],color=colors3[i],marker=markers[i],s=70,alpha=0.8)

plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower right')
plt.show()
fig.savefig(f'E:\\zay\\分子模拟\\人工鱼群\\data\\2024\\0317\\Vicsek_probability_align_noleader.png', dpi=200, bbox_inches='tight')


path = f'D:/zay/2023/1210/Vicsek/leader_wait'
x = np.load(path+f'/N=50/x.npy')
y = np.load(path+f'/N=50/y.npy')
theta = np.load(path+f'/N=50/theta.npy')
fig,ax1 = plt.subplots(figsize=(6,6))
ax1.scatter(x[0][:,-1], y[0][:,-1], color='b', s=40)
#ax2 = fig.add_axes([0.2,0.55,0.3,0.3])
#ax2.quiver(x[0][:,-1], y[0][:,-1], np.cos(theta[0][1:,-1]), np.sin(theta[0][1:,-1]), pivot='tail', color='b',scale=7, units='inches')
#plt.xlim(-ticksrange[id_eta]*1.2,ticksrange[id_eta]*1.2)
#plt.ylim(-ticksrange[id_eta]*1.2,ticksrange[id_eta]*1.2)
ax1.plot(np.mean(x[0],axis=0),np.mean(y[0],axis=0),lw=3.0,ls='--',color='k')
ax1.set_xlim(-10,6010)
ax1.set_ylim(-1010,1010)
ax1.xaxis.set_tick_params(labelsize=24)
ax1.xaxis.set_ticks(np.linspace(0,6000,3))
ax1.yaxis.set_tick_params(labelsize=24)
ax1.yaxis.set_ticks(np.linspace(-1000,1000,3))
ax1.set_xlabel(r'x',fontsize=24)
ax1.set_ylabel(r'y',fontsize=24)
#ax2.xaxis.set_ticks([])
#ax2.yaxis.set_ticks([])
#ax2.set_xlim(8355,8525)
#ax2.set_ylim(-105,65)
plt.show()
fig.savefig(f'D:/zay/2023/1210/Vicsek_trace_2.png', dpi=200, bbox_inches='tight') 
"""
"""


"""
"""
correlation = np.zeros([5,NT])
theta = np.load(f'D:/zay/2023/0917/coupled/eta=0.3/theta.npy')
for it in range(NT):
    correlation[0][it] = np.mean(np.cos(theta[:,0]-theta[:,it]))
for i_NN in range(1,5):
    NN = NN_label[i_NN]
    theta = np.load(f'../0924/align/noleader/N={NN}/theta.npy')
    for it in range(NT):
        correlation[i_NN][it] = np.mean(np.mean(np.cos(theta[:,:,0]-theta[:,:,it]),axis=1))

fig = plt.figure(figsize=(8,6))
for i in range(5):
    plt.plot(np.arange(NT),correlation[i],lw=3.0,color=colors3[i],label=f'N={NN_label[i]}')
    #fit = np.power(np.sin(etalabel[neta]*np.pi)/(np.pi*etalabel[neta]),tlabel)
    #plt.plot(tlabel,fit,lw=2.0,color='k',ls='--')
plt.xlabel(r'time',fontsize=24)
plt.ylabel(r'correlation',fontsize=24)
plt.xlim(0,NT)
#plt.xscale('log')
#plt.yscale('log')
plt.ylim(-0.05,1.05)
plt.xticks(fontsize=24)
plt.yticks(np.linspace(0,1,3),fontsize=24)
plt.legend(fontsize=18)
plt.show()
fig.savefig('./correlation.png', dpi=200, bbox_inches='tight')
"""
"""
def find_cluster(trace_id,t):
    contact = np.zeros([NN,NN])
    for i in range(NN):
        for j in range(i,NN):
            if np.sqrt((x[trace_id][j][t]-x[trace_id][i][t])**2 + (y[trace_id][j][t]-y[trace_id][i][t])**2)<50:
                contact[i][j] = 1
    while(np.sum(contact)>NN):
        link = np.argmax(np.sum(contact,axis=0))
        index = np.where(contact[:,link]>0)[0]
        for i in range(len(index)):
            link_agents = np.where(contact[index[i],:]>0)[0]
            contact[index[i],link_agents] = 0
            contact[index[0],link_agents] = 1
    max_size = np.max(np.sum(contact,axis=1))
    return int(max_size)

NN_label = [3,4,5,10,20,50]
cluster_size = np.zeros([6,NT])
for i_NN in range(6):
    NN = NN_label[i_NN]
    path = f'D:/zay/2023/1112/collective/noleader/N={NN}'
    x = np.load(path+f'/x.npy')
    y = np.load(path+f'/y.npy')
    count = int(5000/NN)
    for it in range(10000):
        size = np.zeros(count)
        for trace_id in range(count):
            size[trace_id] = find_cluster(trace_id,it)
        cluster_size[i_NN][it] = np.mean(size)
np.save(f'D:/zay/2023/1112/collective/noleader/cluster_size.npy',cluster_size)           
cluster_size_2 = np.zeros([7,NT])
#np.save('../0924/align/noleader/cluster_size_1013.npy',cluster_size)
cluster_size = np.load('../0924/align/noleader/cluster_size_1013.npy')

"""
"""

fluctuation = np.zeros([6,NT])
theta = np.load(f'D:/zay/2023/1112/drunkardmemory/eta=0.3/theta.npy')
for it in range(NT):
    fluctuation[0][it] = np.std(theta[:,it])
for i_NN in range(1,6):
    NN = NN_label[i_NN]
    theta = np.load(f'../0924/align/noleader/N={NN}/theta.npy')
    x = np.load(f'../0924/align/noleader/N={NN}/x.npy')
    y = np.load(f'../0924/align/noleader/N={NN}/y.npy')
    count = int(5000/NN)
    for it in range(NT):
        theta_std = []
        for trace_id in range(count):
            contact = np.zeros([NN,NN])
            for i in range(NN-1):
                for j in range(i+1,NN):
                    if np.sqrt((x[trace_id][j][it]-x[trace_id][i][it])**2 + (y[trace_id][j][it]-y[trace_id][i][it])**2)<50:
                        contact[i][j] = contact[j][i] = 1
            theta_std = np.hstack([theta_std,theta[trace_id,np.where(np.sum(contact,axis=0)>=1)[0],it]])
        fluctuation[i_NN][it] = np.std(theta_std)
np.save('cluster_fluctuation.npy',fluctuation)
cmapscale = mpl.cm.rainbow(np.linspace(0,1,7)).shape
colors3 = cmapscale[:,0:3]

NN_label = [2,3,4,5,10,20,50]
fluctuation = np.zeros([7,NT])
for i_NN in range(7):
    NN = NN_label[i_NN]
    #theta = np.load(f'D:/zay/2023/0924/align/leader_wait/N={NN}/theta.npy')[:,1:,:]
    theta = np.load(f'D:/zay/2023/0924/align/noleader/N={NN}/theta.npy')
    for it in range(NT):
        fluctuation[i_NN][it] = np.std(theta[:,:,it])

fig = plt.figure(figsize=(8,6))
for i in range(7):
    plt.plot(np.arange(NT),fluctuation[i],lw=3.0,color=colors3[i],label=f'N={NN_label[i]}')
plt.xlabel(r'time',fontsize=24)
plt.ylabel(r'${\sigma}_{θ}$',fontsize=24)
plt.xlim(0,NT)
#plt.xscale('log')
#plt.yscale('log')
#plt.ylim(-0.05,2.05)
plt.xticks(fontsize=24)
plt.yticks(np.linspace(0,2,3),fontsize=24)
plt.legend(fontsize=18)
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2023/fluctuation_2.png', dpi=200, bbox_inches='tight')
"""
"""
R = np.linspace(197,74,6)/255
G = np.linspace(223,85,6)/255
B = np.linspace(248,162,6)/255
colors3 = np.zeros([6,3])
colors3[:,0] = R
colors3[:,1] = G
colors3[:,2] = B
colors2=['#FF6666','#FF8989','#FCAEAE','#FFEADD']
NN_label=[2,3,4,5,10,20,50]
#data = np.zeros([4998,7,2])
#for i_NN in range(7):
#    NN = NN_label[i_NN]
#    x = np.load(f'D:/zay/2023/0924/align/leader_wait/N={NN}/x.npy')[:,1:,-1]
#    y = np.load(f'D:/zay/2023/0924/align/leader_wait/N={NN}/y.npy')[:,1:,-1]
#    data[:,i_NN,0] = np.arctan2(y,x).reshape(1,-1)[0,:4998]
#    x = np.load(f'D:/zay/2023/0924/align/noleader/N={NN}/x.npy')[:,:,-1]
#    y = np.load(f'D:/zay/2023/0924/align/noleader/N={NN}/y.npy')[:,:,-1]
#    data[:,i_NN,1] = np.arctan2(y,x).reshape(1,-1)[0,:4998]
data = np.load('E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/gdd.npy')
fig = plt.figure(figsize=(8,6))
plt.plot(NN_label,np.std(data[:,:,1],axis=0)/np.pi*180,lw=4.0,color=colors3[5],label='without leader',marker='*',markersize=30,alpha=0.8)
plt.plot(NN_label,np.std(data[:,:,0],axis=0)/np.pi*180,lw=4.0,color=colors2[0],label='with one leader',marker='^',markersize=20,alpha=0.8)
plt.xlabel(r'group size',fontsize=24)
plt.ylabel(r'${\sigma}_{\phi}$',fontsize=24)
plt.xlim(0,50)
plt.ylim(1.5,4)
#plt.yscale('log')
plt.xticks(np.linspace(10,50,3),fontsize=24)
plt.yticks(np.arange(2,5),fontsize=24)
plt.legend(fontsize=18)
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/collective.png', dpi=200, bbox_inches='tight')
"""

"""
R = np.linspace(197,74,6)/255
G = np.linspace(223,85,6)/255
B = np.linspace(248,162,6)/255
colors3 = np.zeros([6,3])
colors3[:,0] = R
colors3[:,1] = G
colors3[:,2] = B
NN_label = [3,4,5,10,20,50]
fluctuation = np.zeros([6,NT])
for i_NN in range(6):
    NN = NN_label[i_NN]
    theta = np.load(f'../0924/align/noleader/N={NN}/theta.npy')
    for it in range(NT):
        fluctuation[i_NN][it] = np.std(theta[:,:,it])
    fluctuation[i_NN] = savgol_filter(fluctuation[i_NN], 53, 3, mode='nearest')

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(2,1,1)
for i in range(6):
    plt.plot(np.arange(NT),fluctuation[i],lw=3.0,color=colors3[i],label=f'N={NN_label[i]}')
plt.xlabel(r'time',fontsize=24)
plt.ylabel(r'fluctuation',fontsize=24)
plt.xlim(0,NT)
#plt.xscale('log')
#plt.yscale('log')
plt.ylim(-0.05,2.05)
plt.xticks(fontsize=24)
plt.yticks(np.linspace(0,2,3),fontsize=24)
#plt.legend(fontsize=18)
ax2 = fig.add_subplot(2,1,2)
for i in range(6):
    plt.plot(np.arange(NT), savgol_filter(cluster_size[i], 53, 3, mode='nearest')/NN_label[i],lw=3.0,color=colors3[i],label=f'N={NN_label[i]}')
plt.xlabel(r'time',fontsize=24)
plt.ylabel(r'group size',fontsize=24)
plt.xlim(0,NT)
#plt.xscale('log')
#plt.yscale('log')
plt.ylim(-0.1,1.1)
plt.xticks(fontsize=24)
plt.yticks(np.linspace(0,1,2),fontsize=24)
plt.legend(fontsize=14,loc='lower left')
plt.show()
fig.savefig(f'E:\\zay\\分子模拟\\人工鱼群\\data\\2023\\1210\\fluctuation_cluster_size.png', dpi=200, bbox_inches='tight')

NN = 20
x = np.load(f'D:/zay/2023/0924/align/leader_wait/N={NN}/x.npy')[0,:,:]
y = np.load(f'D:/zay/2023/0924/align/leader_wait/N={NN}/y.npy')[0,:,:]
theta = np.load(f'D:/zay/2023/0924/align/leader_wait/N={NN}/theta.npy')[0,:,:]
fig,ax1 = plt.subplots(figsize=(6,6))
ax1.scatter(x[1:,-1], y[1:,-1], color='b', s=20)
ax1.scatter(x[0,-1], y[0,-1], color='r', s=30)
ax2 = fig.add_axes([0.2,0.55,0.3,0.3])
ax2.quiver(x[1:,-1], y[1:,-1], np.cos(theta[1:,-1]), np.sin(theta[1:,-1]), pivot='tail', color='b',scale=7, units='inches')
ax2.quiver(x[0,-1], y[0,-1], np.cos(theta[0,-1]), np.sin(theta[0,-1]), pivot='tail', color='red',scale=7, units='inches')
#plt.xlim(-ticksrange[id_eta]*1.2,ticksrange[id_eta]*1.2)
#plt.ylim(-ticksrange[id_eta]*1.2,ticksrange[id_eta]*1.2)
ax1.plot(np.mean(x[1:,:],axis=0),np.mean(y[1:,:],axis=0),lw=3.0,ls='--',color='k')
ax1.set_xlim(-100,10100)
ax1.set_ylim(-510,510)
ax1.xaxis.set_tick_params(labelsize=24)
ax1.xaxis.set_ticks(np.linspace(0,10000,3))
ax1.yaxis.set_tick_params(labelsize=24)
ax1.yaxis.set_ticks(np.linspace(-400,400,3))
ax1.set_xlabel(r'x',fontsize=24)
ax1.set_ylabel(r'y',fontsize=24)
ax2.xaxis.set_ticks([])
ax2.yaxis.set_ticks([])
ax2.set_xlim(8415,8525)
ax2.set_ylim(-95,15)
plt.show()
fig.savefig(f'E:/zay/2025/0907/collective_trace_withleader.png', dpi=200, bbox_inches='tight') 


NN = 20
x = np.load(f'D:/zay/2023/0924/align/leader_wait/N={NN}/x.npy')[0,:,:]
y = np.load(f'D:/zay/2023/0924/align/leader_wait/N={NN}/y.npy')[0,:,:]
theta = np.load(f'D:/zay/2023/0924/align/leader_wait/N={NN}/theta.npy')[0,:,:]
fig,ax1 = plt.subplots(figsize=(6,6))
ax1.scatter(x[1:,-1], y[1:,-1], color='b', s=20)
ax1.scatter(x[0,-1], y[0,-1], color='r', s=30)
ax2 = fig.add_axes([0.24,0.59,0.26,0.26])
ax2.quiver(x[1:,-1], y[1:,-1], np.cos(theta[1:,-1]), np.sin(theta[1:,-1]), pivot='tail', color='b',scale=7, units='inches')
ax2.quiver(x[0,-1], y[0,-1], np.cos(theta[0,-1]), np.sin(theta[0,-1]), pivot='tail', color='red',scale=7, units='inches')
#plt.xlim(-ticksrange[id_eta]*1.2,ticksrange[id_eta]*1.2)
#plt.ylim(-ticksrange[id_eta]*1.2,ticksrange[id_eta]*1.2)
ax1.plot(np.mean(x[1:,:],axis=0),np.mean(y[1:,:],axis=0),lw=3.0,ls='--',color='k')
ax1.set_xlim(-100,10100)
ax1.set_ylim(-510,510)
ax1.xaxis.set_tick_params(labelsize=24)
ax1.xaxis.set_ticks(np.linspace(0,10000,3))
ax1.yaxis.set_tick_params(labelsize=24)
ax1.yaxis.set_ticks(np.linspace(-400,400,3))
ax1.set_xlabel(r'x',fontsize=24)
ax1.set_ylabel(r'y',fontsize=24)
ax2.xaxis.set_ticks([8250,8450])
ax2.yaxis.set_ticks([-100,100])
ax2.set_xlim(8240,8460)
ax2.set_ylim(-110,110)
plt.show()
fig.savefig(f'E:/zay/2025/0907/collective_trace_withleader_{NN}.png', dpi=200, bbox_inches='tight') 


x = np.load(f'D:/zay/2023/0924/align/noleader/N={NN}/x.npy')[0,:,:]
y = np.load(f'D:/zay/2023/0924/align/noleader/N={NN}/y.npy')[0,:,:]
theta = np.load(f'D:/zay/2023/0924/align/noleader/N={NN}/theta.npy')[0,:,:]
fig,ax1 = plt.subplots(figsize=(6,6))
ax1.scatter(x[:,-1], y[:,-1], color='b', s=40)
ax2 = fig.add_axes([0.24,0.59,0.26,0.26])
ax2.quiver(x[:,-1], y[:,-1], np.cos(theta[1:,-1]), np.sin(theta[1:,-1]), pivot='tail', color='b',scale=7, units='inches')
#plt.xlim(-ticksrange[id_eta]*1.2,ticksrange[id_eta]*1.2)
#plt.ylim(-ticksrange[id_eta]*1.2,ticksrange[id_eta]*1.2)
ax1.plot(np.mean(x,axis=0),np.mean(y,axis=0),lw=3.0,ls='--',color='k')
ax1.set_xlim(-100,10100)
ax1.set_ylim(-510,510)
ax1.xaxis.set_tick_params(labelsize=24)
ax1.xaxis.set_ticks(np.linspace(0,10000,3))
ax1.yaxis.set_tick_params(labelsize=24)
ax1.yaxis.set_ticks(np.linspace(-400,400,3))
ax1.set_xlabel(r'x',fontsize=24)
ax1.set_ylabel(r'y',fontsize=24)
ax2.xaxis.set_ticks([8100,8300])
ax2.yaxis.set_ticks([-100,100])
ax2.set_xlim(8090,8310)
ax2.set_ylim(-110,110)
plt.show()
fig.savefig(f'E:/zay/2025/0907/collective_trace_withoutleader_{NN}.png', dpi=200, bbox_inches='tight') 



x = np.load(f'D:/zay/2023/1119/collective/noleader/N=50/x.npy')[0]
y = np.load(f'D:/zay/2023/1119/collective/noleader/N=50/y.npy')[0]
theta = np.load(f'D:/zay/2023/1119/collective/noleader/N=50/theta.npy')[0]
fig,ax1 = plt.subplots(figsize=(6,6))
ax1.scatter(x[0][:,-1], y[0][:,-1], color='b', s=40)
ax2 = fig.add_axes([0.2,0.55,0.3,0.3])
ax2.quiver(x[0][:,-1], y[0][:,-1], np.cos(theta[0][1:,-1]), np.sin(theta[0][1:,-1]), pivot='tail', color='b',scale=7, units='inches')
#plt.xlim(-ticksrange[id_eta]*1.2,ticksrange[id_eta]*1.2)
#plt.ylim(-ticksrange[id_eta]*1.2,ticksrange[id_eta]*1.2)
ax1.plot(np.mean(x[0],axis=0),np.mean(y[0],axis=0),lw=3.0,ls='--',color='k')
ax1.set_xlim(-100,10100)
ax1.set_ylim(-510,510)
ax1.xaxis.set_tick_params(labelsize=24)
ax1.xaxis.set_ticks(np.linspace(0,10000,3))
ax1.yaxis.set_tick_params(labelsize=24)
ax1.yaxis.set_ticks(np.linspace(-400,400,3))
ax1.set_xlabel(r'x',fontsize=24)
ax1.set_ylabel(r'y',fontsize=24)
ax2.xaxis.set_ticks([])
ax2.yaxis.set_ticks([])
ax2.set_xlim(8355,8525)
ax2.set_ylim(-105,65)
plt.show()
fig.savefig(f'D:/zay/2023/1119/collective/noleader/collective_trace.png', dpi=200, bbox_inches='tight') 

x = np.load(f'D:/zay/2023/0924/align/noleader/N={NN}/x.npy')[0,:,:2000:50]
y = np.load(f'D:/zay/2023/0924/align/noleader/N={NN}/y.npy')[0,:,:2000:50]
x = np.load(f'D:/zay/2023/1210/Vicsek/noleader/N={NN}/x.npy')[0,:,:2000:50]
y = np.load(f'D:/zay/2023/1210/Vicsek/noleader/N={NN}/y.npy')[0,:,:2000:50]
import matplotlib.animation as animation
def init():
    d1.set_data([], [])
    return d1,
def update_line(num, x, y, dot1):
    dot1.set_data(x[:,num],y[:,num])
    return dot1,
num = 50
fig1 = plt.figure(figsize=(7,6))
d1, = plt.plot([], [], 'ro',markersize=5)
plt.xlim(-100,2100)
plt.ylim(-1100,1100)
plt.xlabel('X',fontsize=24)
plt.ylabel('Y',fontsize=24)
plt.xticks(np.linspace(0,2000,3),fontsize=18)
plt.yticks(np.linspace(-1000,1000,3),fontsize=18)
#plt.title('Exp')
dot_ani = animation.FuncAnimation(fig1, update_line, np.arange(40),\
fargs=(x,y,d1),interval=10, init_func=init, blit=True)
plt.tight_layout()
dot_ani.save("E:\\zay\\分子模拟\\人工鱼群\\data\\2024\\0317\\memory.gif",writer='pillow')
plt.show()

"""
"""
colors3 = ['k','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['','o','s','D','v']
p = np.zeros([5,1000])
labels = ['','no memory, no leader','with memory, no leader','no memory, with leader','with memory, with leader']
p = np.load(f'E:/zay/分子模拟/人工鱼群/data/2023/1217/Vicsek_p_align_leader_wait.npy')
p[1,:11] = savgol_filter(np.load(f'E:/zay/分子模拟/人工鱼群/data/2023/1217/Vicsek_p_align_noleader.npy')[2,:11],17, 3, mode='nearest')
p[1,10:] = savgol_filter(np.load(f'E:/zay/分子模拟/人工鱼群/data/2023/1217/Vicsek_p_align_noleader.npy')[2,10:],53, 3, mode='nearest')
p[2] = p1[2]*0.2+np.load(f'D:/zay/2023/1210/p_align_noleader.npy')[2]*0.8
p[3,:11] = savgol_filter(np.load(f'E:/zay/分子模拟/人工鱼群/data/2023/1217/Vicsek_p_align_leader_wait.npy')[2,:11],17, 3, mode='nearest')
p[3,10:] = savgol_filter(np.load(f'E:/zay/分子模拟/人工鱼群/data/2023/1217/Vicsek_p_align_leader_wait.npy')[2,10:],53, 3, mode='nearest')
p[4] = p2[2]*0.2+np.load(f'D:/zay/2023/1210/p_align_leader_wait.npy')[2]*0.8

fig = plt.figure(figsize=(8,6))
for i in range(1,5):
    NN = NN_label[i] 
    plt.plot([1000],p[i][100],color=colors3[i],label=labels[i],linewidth=3.0,marker=markers[i],alpha=0.8)
    plt.plot(np.arange(1000)*10,p[i],color=colors3[i],linewidth=3.0,alpha=0.8)
    plt.scatter(np.arange(10)*1000,(p[i]*0.2+p[i]*0.8)[::100],color=colors3[i],marker=markers[i],s=70,alpha=0.8)
plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower left')
plt.show()
fig.savefig(f'E:\\zay\\分子模拟\\人工鱼群\\data\\2024\\0317\\probability_n=10.png', dpi=200, bbox_inches='tight')
"""
"""
R = np.linspace(197,74,7)/255
G = np.linspace(223,85,7)/255
B = np.linspace(248,162,7)/255
colors3 = np.zeros([7,3])
colors3[:,0] = R
colors3[:,1] = G
colors3[:,2] = B

NN_label = [2,3,4,5,10,20,50]
isolated = load_data(f'D:/zay/2023/1210/isolated')
noleader = []
waitleader = []
for i in range(7):
    noleader.append(load_data(f'D:/zay/2023/0924/align/noleader/N={NN_label[i]}'))
    waitleader.append(load_data(f'D:/zay/2023/0924/align/leader_wait/N={NN_label[i]}'))
max_distance = np.zeros([7,10000])
for it in range(1,10000):
    for i in range(7):
        max_distance[i,it] = np.max(noleader[i][0][:,:,it])



isolated_distance = np.zeros(10000)
for it in range(1,10000):
    isolated_distance[it] = np.max(isolated[0][:,it])
isolated_velocity = np.mean(np.clip(np.gradient(isolated_distance),0,1)[-3000:])
velocity = np.mean(np.gradient(max_distance,axis=1)[:,-5000:],axis=1)


max_distance_leader = np.zeros([7,10000])
for it in range(1,10000):
    for i in range(7):
        max_distance_leader[i,it] = np.max(waitleader[i][0][:,1:,it])
velocity_leader = np.mean(np.gradient(max_distance_leader,axis=1)[:,-5000:],axis=1)

fig = plt.figure(figsize=(8,6))
for i in range(7):
    NN = NN_label[i] 
    plt.plot(max_distance_leader[i],color=colors3[i],label=f'N={NN}&1leader',linewidth=3.0,alpha=0.8)
    plt.plot(max_distance_leader[i][9999]-velocity_leader[i]*9999+velocity_leader[i]*np.arange(10000),lw=1.0,color='k')
plt.xlabel('steps',fontsize=24)
plt.ylabel('critical distance',fontsize=24)
plt.xlim(0,10000)
plt.ylim(0,10000)
#plt.xscale('log')
#plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(np.linspace(0,10000,6),fontsize=24)
plt.legend(fontsize=14,loc='upper left')
plt.show()

fig = plt.figure(figsize=(8,6))
for i in range(7):
    NN = NN_label[i] 
    plt.plot(max_distance[i],color=colors3[i],label=f'N={NN}',linewidth=3.0,alpha=0.8)
    plt.plot(max_distance[i][9999]-velocity[i]*9999+velocity[i]*np.arange(10000),lw=1.0,color='k')
plt.xlabel('steps',fontsize=24)
plt.ylabel('critical distance',fontsize=24)
plt.xlim(0,10000)
plt.ylim(0,10000)
#plt.xscale('log')
#plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(np.linspace(0,10000,6),fontsize=24)
plt.legend(fontsize=14,loc='upper left')
plt.show()



max_distance = np.zeros([7,10000])
for it in range(1,10000):
    for i in range(7):
        max_distance[i,it] = np.max(noleader[i][0][:,:,it])



isolated_distance = np.zeros(10000)
for it in range(1,10000):
    isolated_distance[it] = np.max(isolated[0][:,it])
isolated_velocity = np.mean(np.clip(np.gradient(isolated_distance),0,1)[-3000:])
velocity = np.mean(np.gradient(max_distance,axis=1)[:,-5000:],axis=1)


max_distance_leader = np.zeros([7,10000])
for it in range(1,10000):
    for i in range(7):
        max_distance_leader[i,it] = np.max(waitleader[i][0][:,1:,it])
velocity_leader = np.mean(np.gradient(max_distance_leader,axis=1)[:,-5000:],axis=1)

fig = plt.figure(figsize=(8,6))
for i in range(7):
    NN = NN_label[i] 
    plt.plot(max_distance_leader[i],color=colors3[i],label=f'N={NN}&1leader',linewidth=3.0,alpha=0.8)
    plt.plot(max_distance_leader[i][9999]-velocity_leader[i]*9999+velocity_leader[i]*np.arange(10000),lw=1.0,color='k')
plt.xlabel('steps',fontsize=24)
plt.ylabel('critical distance',fontsize=24)
plt.xlim(0,10000)
plt.ylim(0,10000)
#plt.xscale('log')
#plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(np.linspace(0,10000,6),fontsize=24)
plt.legend(fontsize=14,loc='upper left')
plt.show()

fig = plt.figure(figsize=(8,6))
for i in range(7):
    NN = NN_label[i] 
    plt.plot(max_distance[i],color=colors3[i],label=f'N={NN}',linewidth=3.0,alpha=0.8)
    plt.plot(max_distance[i][9999]-velocity[i]*9999+velocity[i]*np.arange(10000),lw=1.0,color='k')
plt.xlabel('steps',fontsize=24)
plt.ylabel('critical distance',fontsize=24)
plt.xlim(0,10000)
plt.ylim(0,10000)
#plt.xscale('log')
#plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(np.linspace(0,10000,6),fontsize=24)
plt.legend(fontsize=14,loc='upper left')
plt.show()

collective_effect = (velocity-isolated_velocity)/isolated_velocity
collective_effect_leader = (velocity_leader-isolated_velocity)/isolated_velocity

fig = plt.figure(figsize=(8,6))
plt.hlines(np.mean(collective_effect_leader),0,60,lw=4.0,ls=':',color='r',alpha=0.6)
plt.plot(NN_label,collective_effect,lw=4.0,color=colors3[5],label='N agents without leader',marker='*',markersize=30,alpha=0.8)
plt.xlabel(r'group size',fontsize=24)
plt.ylabel(r'${({v_N}-{v_0})}/{v_0}$',fontsize=24)
plt.xlim(0,53)
plt.ylim(-0.2,0.5)
#plt.yscale('log')
plt.xticks(np.linspace(10,50,3),fontsize=24)
plt.yticks(np.linspace(0,0.4,3),fontsize=24)
plt.legend(fontsize=18,loc='lower right')
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2024/0324/collective.png', dpi=200, bbox_inches='tight')
"""
"""
########
NN_label = [1,5,10,20,50]
p = np.zeros([5,1000])
p1 = np.load(f'D:/zay/2023/0924/p_align_noleader.npy')
p2 = np.load(f'D:/zay/2023/0924/p_align_wait_leader.npy')
#x = np.load(f'D:/zay/2023/1210/isolated/x.npy').reshape(-1,10000)
#y = np.load(f'D:/zay/2023/1210/isolated/y.npy').reshape(-1,10000)
#for ix in np.arange(0,1000):
#    if np.mod(ix,100)==0:
#        print('distance=',ix)
#    p[0][ix] = len(np.where(np.min(np.sqrt((x-ix*10)**2+y**2),axis=1)<=10)[0])/10000
#path = f'D:/zay/2023/1210/collective/noleader'
#for i_NN in range(1,5):
#    NN = NN_label[i_NN]
#    print('N=',NN)
#    x = np.load(path+f'/N={NN}/x.npy').reshape(-1,10000)
#    y = np.load(path+f'/N={NN}/y.npy').reshape(-1,10000)
#    for ix in np.arange(0,1000):
#        if np.mod(ix,100)==0:
#            print('distance=',ix)
#        p[i_NN][ix] = len(np.where(np.min(np.sqrt((x-ix*10)**2+y**2),axis=1)<=10)[0])/20000
#np.save(f'D:/zay/2023/1210/p_align_noleader.npy',p)
#p = np.load(f'D:/zay/2023/1210/p_align_noleader.npy')

path = f'D:/zay/2023/1210/collective/leader_wait'
for i_NN in range(1,5):
    NN = NN_label[i_NN]
    print('N=',NN)
    x = np.load(path+f'/N={NN}/x.npy').reshape(-1,10000)
    y = np.load(path+f'/N={NN}/y.npy').reshape(-1,10000)
    for ix in np.arange(0,1000):
        if np.mod(ix,100)==0:
            print('distance=',ix)
        p[i_NN][ix] = len(np.where(np.min(np.sqrt((x-ix*10)**2+y**2),axis=1)<=10)[0])/20000-1/NN
#np.save(f'D:/zay/2023/1210/p_align_leader_wait.npy',p)
#p = np.load(f'D:/zay/2023/1210/p_align_leader_wait.npy')
#######

colors3 = ['k','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['','o','s','D','v']
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(1000)*10,p2[0]*0.1+p[0]*0.9,color=colors3[0],label=f'isolated individual',linewidth=3.0)
for i in range(1,5):
    NN = NN_label[i] 
    plt.plot([1000],p2[i][100]*0.2+p[i][100]*0.8,color=colors3[i],label=f'N={NN_label[i]} & 1leader',linewidth=3.0,marker=markers[i],alpha=0.8)
    plt.plot(np.arange(1000)*10,p2[i]*0.2+p[i]*0.8,color=colors3[i],linewidth=3.0,alpha=0.8)
    plt.scatter(np.arange(10)*1000,(p2[i]*0.2+p[i]*0.8)[::100],color=colors3[i],marker=markers[i],s=70,alpha=0.8)
plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower left')
plt.show()
#fig.savefig(f'E:\\zay\\分子模拟\\人工鱼群\\data\\2024\\0317\\probability_align_wait_leader.png', dpi=200, bbox_inches='tight')
p = np.load(f'D:/zay/2023/1210/p_align_noleader.npy')
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(1000)*10,p1[0]*0.1+p[0]*0.9,color=colors3[0],label=f'isolated individual',linewidth=3.0)
for i in range(1,5):
    NN = NN_label[i]
    plt.plot([1000],p1[i][100]*0.2+p[i][100]*0.8,color=colors3[i],label=f'N={NN_label[i]}',linewidth=3.0,marker=markers[i],alpha=0.8)
    plt.plot(np.arange(1000)*10,p1[i]*0.2+p[i]*0.8,color=colors3[i],linewidth=3.0,alpha=0.8)
    plt.scatter(np.arange(10)*1000,(p1[i]*0.2+p[i]*0.8)[::100],color=colors3[i],marker=markers[i],s=70,alpha=0.8)

plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower left')
plt.show()
#fig.savefig(f'E:\\zay\\分子模拟\\人工鱼群\\data\\2024\\0317\\probability_align_noleader.png', dpi=200, bbox_inches='tight')
"""


"""
########
NN_label = [1,5,10,20,50]  ##p [2,3,4,5,10,20,50]
p0 = np.load(f'D:/zay/2023/0924/p_align_noleader.npy')[0]*0.1+np.load(f'D:/zay/2023/1210/p_align_noleader.npy')[0]*0.9
label = ['align_noleader','align_leader_wait','vicsek_noleader','vicsek_leader_wait']
colors3 = ['k','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['','o','s','D','v']


fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(1000)*10,p0,color=colors3[0],label=f'isolated individual',linewidth=3.0)
p = np.load(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=10/p_align_noleader.npy')[2:,:]
for i in range(1,5):
    NN = NN_label[i] 
    plt.plot([1000],p[i][100],color=colors3[i],label=f'N={NN_label[i]}',linewidth=3.0,marker=markers[i],alpha=0.8)
    plt.plot(np.arange(1000)*10,p[i],color=colors3[i],linewidth=3.0,alpha=0.6)
    plt.scatter(np.arange(50)*200,p[i][::20],color=colors3[i],marker=markers[i],s=70,alpha=0.8)
plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower left')
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=10/pro_align_noleader.png', dpi=200, bbox_inches='tight')

fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(1000)*10,p0,color=colors3[0],label=f'isolated individual',linewidth=3.0)
p = np.load(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=10/p_align_leader_wait.npy')[2:,:]
for i in range(1,5):
    NN = NN_label[i] 
    plt.plot([1000],p[i][100],color=colors3[i],label=f'N={NN_label[i]} & 1leader',linewidth=3.0,marker=markers[i],alpha=0.8)
    plt.plot(np.arange(1000)*10,p[i],color=colors3[i],linewidth=3.0,alpha=0.6)
    plt.scatter(np.arange(50)*200,p[i][::20],color=colors3[i],marker=markers[i],s=70,alpha=0.8)
plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower left')
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=10/pro_align_wait_leader.png', dpi=200, bbox_inches='tight')

fig = plt.figure(figsize=(8,6))
p = np.load(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=10/p_vicsek_noleader.npy')[2:,:]
for i in range(1,5):
    NN = NN_label[i] 
    plt.plot([1000],p[i][100],color=colors3[i],label=f'N={NN_label[i]}',linewidth=3.0,marker=markers[i],alpha=0.8)
    plt.plot(np.arange(1000)*10,p[i],color=colors3[i],linewidth=3.0,alpha=0.6)
    plt.scatter(np.arange(50)*200,p[i][::20],color=colors3[i],marker=markers[i],s=70,alpha=0.8)
plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower left')
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=10/pro_vicsek_noleader.png', dpi=200, bbox_inches='tight')

fig = plt.figure(figsize=(8,6))
p = np.load(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=10/p_vicsek_leader_wait.npy')[2:,:]
for i in range(1,5):
    NN = NN_label[i] 
    plt.plot([1000],p[i][100],color=colors3[i],label=f'N={NN_label[i]} & 1leader',linewidth=3.0,marker=markers[i],alpha=0.8)
    plt.plot(np.arange(1000)*10,p[i],color=colors3[i],linewidth=3.0,alpha=0.6)
    plt.scatter(np.arange(50)*200,p[i][::20],color=colors3[i],marker=markers[i],s=70,alpha=0.8)
plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower left')
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=10/pro_vicsek_wait_leader.png', dpi=200, bbox_inches='tight')

############
############

fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(1000)*10,p0,color=colors3[0],label=f'isolated individual',linewidth=3.0)
p = np.load(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/p_align_noleader.npy')[2:,:]
for i in range(1,5):
    NN = NN_label[i] 
    plt.plot([1000],p[i][100],color=colors3[i],label=f'N={NN_label[i]}',linewidth=3.0,marker=markers[i],alpha=0.8)
    plt.plot(np.arange(1000)*10,p[i],color=colors3[i],linewidth=3.0,alpha=0.6)
    plt.scatter(np.arange(50)*200,p[i][::20],color=colors3[i],marker=markers[i],s=70,alpha=0.8)
plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower left')
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/pro_align_noleader.png', dpi=200, bbox_inches='tight')

fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(1000)*10,p0,color=colors3[0],label=f'isolated individual',linewidth=3.0)
p = np.load(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/p_align_leader_wait.npy')[2:,:]
for i in range(1,5):
    NN = NN_label[i] 
    plt.plot([1000],p[i][100],color=colors3[i],label=f'N={NN_label[i]} & 1leader',linewidth=3.0,marker=markers[i],alpha=0.8)
    plt.plot(np.arange(1000)*10,p[i],color=colors3[i],linewidth=3.0,alpha=0.6)
    plt.scatter(np.arange(50)*200,p[i][::20],color=colors3[i],marker=markers[i],s=70,alpha=0.8)
plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower left')
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/pro_align_wait_leader.png', dpi=200, bbox_inches='tight')

fig = plt.figure(figsize=(8,6))
p = np.load(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/p_vicsek_noleader.npy')[2:,:]
for i in range(1,5):
    NN = NN_label[i] 
    plt.plot([1000],p[i][100],color=colors3[i],label=f'N={NN_label[i]}',linewidth=3.0,marker=markers[i],alpha=0.8)
    plt.plot(np.arange(1000)*10,p[i],color=colors3[i],linewidth=3.0,alpha=0.6)
    plt.scatter(np.arange(50)*200,p[i][::20],color=colors3[i],marker=markers[i],s=70,alpha=0.8)
plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower left')
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/pro_vicsek_noleader.png', dpi=200, bbox_inches='tight')

fig = plt.figure(figsize=(8,6))
p = np.load(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/p_vicsek_leader_wait.npy')[2:,:]
for i in range(1,5):
    NN = NN_label[i] 
    plt.plot([1000],p[i][100],color=colors3[i],label=f'N={NN_label[i]} & 1leader',linewidth=3.0,marker=markers[i],alpha=0.8)
    plt.plot(np.arange(1000)*10,p[i],color=colors3[i],linewidth=3.0,alpha=0.6)
    plt.scatter(np.arange(50)*200,p[i][::20],color=colors3[i],marker=markers[i],s=70,alpha=0.8)
plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower left')
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/pro_vicsek_wait_leader.png', dpi=200, bbox_inches='tight')



colors3 = ['k','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['','o','s','D','v']
p = np.zeros([4,1000])
labels = ['no memory, no leader','with memory, no leader','no memory, with leader','with memory, with leader']
p[0] = np.load(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/p_vicsek_noleader.npy')[4,:]
p[1] = np.load(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/p_align_noleader.npy')[4,:]
p[2] = np.load(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/p_vicsek_leader_wait.npy')[4,:]
p[3] = np.load(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/p_align_leader_wait.npy')[4,:]
fig = plt.figure(figsize=(8,6))
for i in range(4):
    NN = NN_label[i] 
    plt.plot([1000],p[i][100],color=colors3[i+1],label=labels[i],linewidth=3.0,marker=markers[i+1],alpha=0.8)
    plt.plot(np.arange(1000)*10,p[i],color=colors3[i+1],linewidth=3.0,alpha=0.6)
    plt.scatter(np.arange(50)*200,p[i][::20],color=colors3[i+1],marker=markers[i+1],s=70,alpha=0.8)
plt.xlabel('distance',fontsize=24)
plt.ylabel('probability',fontsize=24)
plt.xlim(0,10000)
plt.ylim(3e-3,1.1)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(np.linspace(0,10000,6),fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=14,loc='lower left')
plt.text(8000,0.6,'N=10',fontsize=24)
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2024/0421_N=10.png', dpi=200, bbox_inches='tight')

"""
R = np.linspace(197,74,7)/255
G = np.linspace(223,85,7)/255
B = np.linspace(248,162,7)/255
colors3 = np.zeros([7,3])
colors3[:,0] = R
colors3[:,1] = G
colors3[:,2] = B
colors2=['#FF6666','#FF8989','#FCAEAE','#FFEADD']
NN_label = [2,3,4,5,10,20,50]
data = np.load(f'E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/criticaldistance.npy')/10000
fig = plt.figure(figsize=(8,6))
noleader = np.mean(data[:,:,0],axis=0)
leader = np.mean(data[:,:,1],axis=0)
leader = leader-noleader
noleader = noleader-noleader[0]
plt.plot(NN_label,noleader,lw=6.0,ls='-',color=colors3[5],label='collective effect',marker='*',markersize=25,alpha=0.8)
plt.plot(NN_label,leader,lw=6.0,ls='-',color=colors2[0],label='leader effect',marker='^',markersize=20,alpha=0.8)
#samplenum =50
#for i in range(7):
#    NN = NN_label[i]   
#    noleader = random.sample(list(data[:,i,0]),samplenum)
#    plt.scatter([NN for i in range(samplenum)],noleader,facecolors = 'None',edgecolors=colors3[5],marker='o',s=100,alpha=0.05)
#    leader = random.sample(list(data[:,i,1]),samplenum)
#    plt.scatter([NN for i in range(samplenum)],leader,facecolors = 'None',edgecolors=colors2[0],marker='o',s=100,alpha=0.05)

plt.xlabel('groupsize',fontsize=24)
plt.ylabel(r'$\Delta\~{v}$',fontsize=24)
plt.xlim(0,51)
plt.ylim(-0.02,0.29)
plt.xticks([0,10,20,50],fontsize=24)
plt.yticks(np.linspace(0,0.2,3),fontsize=24)
plt.legend(fontsize=18,loc='upper left')
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2024/0421/collective2.png', dpi=200, bbox_inches='tight')


R = np.linspace(197,74,6)/255
G = np.linspace(223,85,6)/255
B = np.linspace(248,162,6)/255
colors3 = np.zeros([6,3])
colors3[:,0] = R
colors3[:,1] = G
colors3[:,2] = B
colors2=['#FF6666','#FF8989','#FCAEAE','#FFEADD']
NN_label=[2,3,4,5,10,20,50]
data = np.load('E:/zay/分子模拟/人工鱼群/data/2024/0414/drunkwalk/r=20/gdd.npy')
cl = np.std(data[:,:,1],axis=0)/np.pi*180
ld = np.std(data[:,:,0],axis=0)/np.pi*180
ld = ld -cl
cl = cl - cl[0]
fig = plt.figure(figsize=(8,6))
plt.plot(NN_label,cl,lw=4.0,color=colors3[5],label='collective effect',marker='*',markersize=30,alpha=0.8)
plt.plot(NN_label,ld,lw=4.0,color=colors2[0],label='leader effect',marker='^',markersize=20,alpha=0.8)
plt.xlabel(r'group size',fontsize=24)
plt.ylabel(r'$\Delta{\sigma}_{\phi}$',fontsize=24)
plt.xlim(0,51)
plt.ylim(-2.1,0.2)
#plt.yscale('log')
plt.xticks(np.linspace(10,50,3),fontsize=24)
plt.yticks(np.arange(-2,1),fontsize=24)
plt.legend(fontsize=18,loc = 'lower left')
plt.show()
fig.savefig(f'E:/zay/分子模拟/人工鱼群/data/2024/0421/collective.png', dpi=200, bbox_inches='tight')