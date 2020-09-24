

import sys
import cupy as cp# for GPU-acceleration when available
from numba import jit, njit, cuda, prange
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numba
#from multiprocessing import Pool, freeze_support
#import multiprocessing.dummy as mp
import numpy as np
import random as rand
#import fastrand as frand
from math import exp
#import multi_proc
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import torch
from monte_analys import run_plot

# when using numba, the first "run" of a function, with @jit @njit,... , works as a 
# compilation of that particular code. Thus before you do any benchmarks make sure 
# you have executed all @jit-, @njit-,... functions once before you time it.
# Also, since the first time a @jit-, @njit-,... function compiles a particular function
# it might take a while for large inputs, here, a large latic.

#black or white sweep, used in numba_parallel_sweep
@njit(fastmath=True, parallel=True)
def b_or_w_sweep(E, L, M, N, T, is_black=1):
    xup=0
    xdown=0
    yup=0
    ydown=0
    zup=0
    zdown=0
    for x in prange(N):
        for y in prange(N):
            for z in prange(N):
                if (x+y+z)%2==is_black:
                    xup=x+1
                    xdown=x-1
                    yup=y+1
                    ydown=y-1
                    zup=z+1
                    zdown=z-1
                    if xup ==N:
                        xup = 0
                    if xdown == -1:
                        xdown =N-1
                    if yup ==N:
                        yup = 0
                    if ydown == -1:
                        ydown =N-1
                    if zup ==N:
                        zup = 0 
                    if zdown == -1:
                        zdown =N-1
                    dE =  2 * L[x,y,z] * (L[xup,y,z] + L[xdown,y,z] + L[x,yup,z] + L[x,ydown,z] + L[x,y,zup] + L[x,y,zdown])
                    if exp(-dE/T) > np.random.rand():
                        L[x,y,z] = -1*L[x,y,z]
                        dE = -1*dE
                    #dE = L[x,y,z] * (L[xup,y,z] + L[xdown,y,z] + L[x,yup,z] + L[x,ydown,z] + L[x,y,zup] + L[x,y,zdown])
                    E -= dE*(1/2)
                    M += L[x,y,z]
    return L, E, M

@njit(fastmath=True)
def numba_sweep(L_, T):
    #L_ = np.ascontiguousarray(L_in)
    N = L_.shape[0]
    #i = 0
    xup=0
    xdown=0
    yup=0
    ydown=0
    zup=0
    zdown=0
    dE = 0.0
    M = 0.0
    E = 0.0
    for x in range(N):
        for y in range(N):
            for z in range(N):
                xup=x+1
                xdown=x-1
                yup=y+1
                ydown=y-1
                zup=z+1
                zdown=z-1
                if xup == N:
                    xup = 0
                if xdown == -1:
                    xdown = N-1

                if yup == N:
                    yup = 0
                if ydown == -1:
                    ydown = N-1
                
                if zup == N:
                    zup = 0 
                if zdown == -1:
                    zdown = N-1

                dE =  2 * L_[x,y,z] * (L_[xup,y,z] + L_[xdown,y,z] + L_[x,yup,z] + L_[x,ydown,z] + L_[x,y,zup] + L_[x,y,zdown])
                if exp(-dE/T) > np.random.rand():
                    L_[x,y,z] = -1*L_[x,y,z]
                    dE = -1*dE
                #dE = L_[x,y,z] * (L_[xup,y,z] + L_[xdown,y,z] + L_[x,yup,z] + L_[x,ydown,z] + L_[x,y,zup] + L_[x,y,zdown])
                E -= dE*(1/2)
                M += L_[x,y,z]
    #return L_, E/(N**3), (E/(N**3))*(E/(N**3)), M/(N**3), (M/(N**3))*(M/(N**3)), (M/(N**3))*(M/(N**3))*(M/(N**3))*(M/(N**3))
    return L_, E, M

@njit
def numba_sweep_parallel(L_, T):
    N = L_.shape[0]
    M = 0.0
    E = 0.0
    E1 = 0.0
    E2 = 0.0

    #black sweep
    L_, E1, M = b_or_w_sweep(E=E, L=L_, M=M,  N=N, T=T, is_black=1)

    #white sweep
    L_, E2, M = b_or_w_sweep(E=E, L=L_, M=M,  N=N, T=T, is_black=0)

    return L_, (E1+E2), M
    #return L_, E/(N**3), (E**2)/(N**3), M/(N**3), (M**2/(N**3)), (M**4/(N**3))


# TO DO, PRIO
@cuda.jit
def cuda_parallel(L_, N, T, rng_states):
    #threads = 128
    x,y,z = cuda.grid(3)
    while x < N:
        while y < N:
            while z < N:
                # black sweep
                if (x+y+z) % 2 == 0:
                    xup,xdown,yup,ydown,zup,zdown = x+1,x-1,y+1,y-1,z+1,z-1
                    if xup == N:
                        xup = 0
                    if xdown == -1:
                        xdown = N-1

                    if yup == N:
                        yup = 0
                    if ydown == -1:
                        ydown = N-1
                    
                    if zup == N:
                        zup = 0 
                    if zdown == -1:
                        zdown = N-1
                    dE = 2 * L_[x,y,z] * (L_[xup,y,z] + L_[xdown,y,z] +\
                        L_[x,yup,z] + L_[x,ydown,z] +\
                            L_[x,y,zup] + L_[x,y,zdown])
                    Lxyz = L_[x,y,z]
                    if exp(-dE/T) > xoroshiro128p_uniform_float32(rng_states, x+y+z):
                        L_[x,y,z] = -Lxyz
                    elif dE < 0:
                        L_[x,y,z] = -Lxyz
    while x < N:
        while y < N:
            while z < N:
                #white sweep
                if (x+y+z) % 2 == 1:
                    xup,xdown,yup,ydown,zup,zdown = x+1,x-1,y+1,y-1,z+1,z-1
                    if xup == N:
                        xup = 0
                    if xdown == -1:
                        xdown = N-1

                    if yup == N:
                        yup = 0
                    if ydown == -1:
                        ydown = N-1
                    
                    if zup == N:
                        zup = 0 
                    if zdown == -1:
                        zdown = N-1
                    dE = 2 * L_[x,y,z] * (L_[xup,y,z] + L_[xdown,y,z] +\
                        L_[x,yup,z] + L_[x,ydown,z] +\
                            L_[x,y,zup] + L_[x,y,zdown])
                    Lxyz = L_[x,y,z]
                    if exp(-dE/T) > xoroshiro128p_uniform_float32(rng_states, x*y*z+x+y-z):
                        L_[x,y,z] = -Lxyz
                    elif dE < 0:
                        L_[x,y,z] = -Lxyz

    #return L_

class Latice(object):
    def __init__(self, N, M, if_cuda):
        self.M = M
        self.N = N
        self.E, self.E_2 = 0.0, 0.0
        self.M_, self.M_2, self.M_4 = [],[],[]
        try:
            if if_cuda:
                self.cp_ = True
            else:
                self.cp_ = False             
        except NameError:
            print('Cupy not imported')
    
    def Make_Latice(self):
        if self.cp_:
            self.L = cp.zeros((self.N,self.N,self.N))
        else:
            self.L = np.zeros((self.N,self.N,self.N))
        for x in range(self.N):
            for y in range(self.N):
                for z in range(self.N):
                    if M != 0:
                        self.L[x,y,z] = self.M
                    else:
                        self.L[x,y,z] = rand.choice([-1, 1]) # [0, 1]
    
    def temps(self, T_start, T_end, dT):
        self.T_start = T_start
        self.T_end = T_end
        self.dT = dT

        self.mean_E = np.arange(T_start,T_end,dT)*0
        self.mean_E_2 = np.arange(T_start,T_end,dT)*0
        self.mean_M = np.arange(T_start,T_end,dT)*0
        self.mean_M_2 = np.arange(T_start,T_end,dT)*0
        self.mean_M_4 = np.arange(T_start,T_end,dT)*0


    def bench_test(self, t=np.float32(4.5), do='do_all', loops=1000):
        if do=='do_all':
            start_time = time.time()
            for i in tqdm(range(loops), leave=False):
                L.sweep_Latice_single_thread(T=t, num_loop=num_loop, do_brownian=True)
            print(' Stochastic walk sweep: %f seconds per sweep ' % ((time.time() - start_time)/loops))
            
            start_time = time.time()
            for i in tqdm(range(loops), leave=False):
                L.sweep_Latice_single_thread(T=t)
            print(' Single Thread structured sweep: %f seconds per sweep ' % ((time.time() - start_time)/loops))

            '''start_time = time.time()
            for i in tqdm(range(loops), leave=False):
                L.sweep_Latice_parallel(T=t)
            print(' Parallel structured sweep: %f seconds per sweep ' % ((time.time() - start_time)/loops))'''

            L.sweep_Latice_parallel(T=t, do_='numba')
            start_time = time.time()
            for i in tqdm(range(loops), leave=False):
                L.sweep_Latice_parallel(T=t, do_='numba')
            #print(' Numba structured sweep: %f seconds per sweep ' % ((time.time() - start_time)/loops))

            L.sweep_Latice_parallel(T=t, do_='numba_parallel')
            start_time = time.time()
            for i in tqdm(range(loops), leave=False):
                L.sweep_Latice_parallel(T=t, do_='numba_parallel')
            #print(' Numba parallel structured sweep: %f seconds per sweep ' % ((time.time() - start_time)/loops))

            '''start_time = time.time()
            for i in tqdm(range(100), leave=False):
                L.sweep_Latice_parallel(T=t, do_=True)
            print(' Cuda structured sweep: %f seconds per sweep ' % ((time.time() - start_time)/loops))'''

        elif do=='stochastic':
            start_time = time.time()
            for i in tqdm(range(loops), leave=False):
                L.sweep_Latice_single_thread(T=t, num_loop=num_loop, do_brownian=True)
            print(' Stochastic walk sweep: %f seconds per sweep ' % ((time.time() - start_time)/loops))

        elif do=='single':
            start_time = time.time()
            for i in tqdm(range(loops), leave=False):
                L.sweep_Latice_single_thread(T=t)
            print(' Single Thread structured sweep: %f seconds per sweep ' % ((time.time() - start_time)/loops))
            return ((time.time() - start_time)/loops)
            
        elif do=='multi':
            start_time = time.time()
            for i in tqdm(range(loops), leave=False):
                L.sweep_Latice_parallel(T=t, do_=do)
            print(' Parallel structured sweep: %f seconds per sweep ' % ((time.time() - start_time)/loops))

        elif do=='numba':
            L.sweep_Latice_parallel(T=t, do_=do)
            start_time = time.time()
            for i in tqdm(range(loops), leave=False):
                L.sweep_Latice_parallel(T=t, do_=do)
            #print(' Numba structured sweep: %f seconds per sweep ' % ((time.time() - start_time)/loops))
            return ((time.time() - start_time)/loops)
        
        elif do=='numba_parallel':
            L.sweep_Latice_parallel(T=t, do_=do)
            start_time = time.time()
            for i in tqdm(range(loops), leave=False):
                L.sweep_Latice_parallel(T=t, do_=do)
            #print(' Numba structured sweep: %f seconds per sweep ' % ((time.time() - start_time)/loops))
            return ((time.time() - start_time)/loops)

        elif do=='cuda':
            start_time = time.time()
            for i in tqdm(range(loops), leave=False):
                L.sweep_Latice_parallel(T=t, do_=do)
            print(' Cuda structured sweep: %f seconds per sweep ' % ((time.time() - start_time)/loops))
            
    def Warm_up(self, T=1, C=False, do='numba_parallel'):
        #if C:
        for i in range(50000):
            L.sweep_Latice_parallel(T=T, do_=do ,do_return=False)


    def sweep_Latice_single_thread(self, T=1, num_loop=1e+3, do_return=False, n_range=1, do_brownian=False):
        # initiall coodrinates for the sweep
        #self.m
        if do_brownian:
            x, y, z = rand.choice(range(self.N)),\
                rand.choice(range(self.N)),\
                    rand.choice(range(self.N))
            #start_time = time.time()
            for k in range(int(num_loop)):
            #np.arange(int(num_loop)):
                x,y,z = x+frand.pcg32bounded(2*n_range+1)-n_range,\
                    y+frand.pcg32bounded(2*n_range+1)-n_range,\
                        z+frand.pcg32bounded(2*n_range+1)-n_range
                if x == self.N:
                    x=0
                elif x < 0:
                    x=x+self.N
                elif x > self.N:
                    x=x-self.N

                if y == self.N:
                    y=0
                elif y < 0:
                    y=y+self.N
                elif y > self.N:
                    y=y-self.N

                if z == self.N:
                    z=0
                elif z < 0:
                    z=z+self.N
                elif z > self.N:
                    z=z-self.N

                xup,xdown,yup,ydown,zup,zdown = x+1,x-1,y+1,y-1,z+1,z-1

                if xup == self.N:
                    xup = 0
                if xdown == -1:
                    xdown = self.N-1

                if yup == self.N:
                    yup = 0
                if ydown == -1:
                    ydown = self.N-1
                
                if zup == self.N:
                    zup = 0 
                if zdown == -1:
                    zdown = self.N-1

                dE = 2 * self.L[x,y,z] * (self.L[xup,y,z] + self.L[xdown,y,z] +\
                    self.L[x,yup,z] + self.L[x,ydown,z] +\
                        self.L[x,y,zup] + self.L[x,y,zdown])

                if np.exp(-dE/T) > np.random.uniform():
                    self.L[x,y,z] = -1*self.L[x,y,z]
                elif dE < 0:
                    self.L[x,y,z] = -1*self.L[x,y,z]
                #elif math.exp(-1*dE/T) > uniform(0,1):
                #elif np.exp(-dE/T) > frand.pcg32bounded(1):
            
            #if int(num_loop/k) % num_loop
        else:
            for x in range(self.N):
                for y in range(self.N):
                    for z in range(self.N):
                        xup,xdown,yup,ydown,zup,zdown = x+1,x-1,y+1,y-1,z+1,z-1
                        if xup == self.N:
                            xup = 0
                        if xdown == -1:
                            xdown = self.N-1

                        if yup == self.N:
                            yup = 0
                        if ydown == -1:
                            ydown = self.N-1
                        
                        if zup == self.N:
                            zup = 0 
                        if zdown == -1:
                            zdown = self.N-1
                        dE = 2 * self.L[x,y,z] * (self.L[xup,y,z] + self.L[xdown,y,z] +\
                            self.L[x,yup,z] + self.L[x,ydown,z] +\
                                self.L[x,y,zup] + self.L[x,y,zdown])
                        #if math.exp(-1*dE/T) > uniform(0,1):
                        #if np.exp(-dE/T) > frand.pcg32bounded(1):
                        if dE < 0:
                            self.L[x,y,z] = -1*self.L[x,y,z]
                        elif np.exp(-dE/T) > np.random.uniform():
                            self.L[x,y,z] = -1*self.L[x,y,z]

        #print("--- %.2E seconds ---" % (time.time() - start_time))
    
    def sweep_Latice_parallel(self, T=1, do_return=False,  do_='numba_parallel', **kwargs):
        #@cuda.jit
        self.T = np.float64(T)
        if do_ == 'cuda':
            #@jit(python=True,target="cuda")    
            '''blocks = 100
            threads_per_block = 100
            rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
            L_d = cuda.to_device(self.L)
            cuda_parallel[blocks, threads_per_block](L_d, self.N, self.T, rng_states)
            self.L = L_d.copy_to_host()'''
            print('TO DO, DONT DO CUDA')
                    
        if do_ == 'multi':
            print('TO DO, DONT DO MULTI, USE NUMBA PARALLEL INSTEAD')
            if __name__ == '__main__':
                #if self.Par == False:
                freeze_support()
                #lim = self.N
                num_processors = 3
                p = Pool(num_processors)

                self.is_black = True
                p.map(multi_, np.arange(self.N))
                for i in range(num_processors):
                    p.close()
                    p.join()         

                self.is_black = False
                p.map(multi_, np.arange(self.N))
                for i in range(num_processors):
                    p.close()
                    p.join()
            

        if do_ == 'numba':
            #rand_seq = rand_num()
            #self.L, self.E, self.E_2, self.M_, self.M_2, self.M_4 = numba_sweep(self.L, self.T)
            self.L, self.E, self.M = numba_sweep(self.L, self.T)
        if do_ == 'numba_parallel':
            #self.L, self.E, self.E_2, self.M_, self.M_2, self.M_4 = numba_sweep_parallel(self.L, self.T)
            self.L, self.E, self.M = numba_sweep_parallel(self.L, self.T)
            #numba_sweep_parallel(self.L, self.T)
            #numba_sweep_parallel(self.L, self.T).parallel_diagnostics(level=4)

    def Plot_Latice_3D(self, M_s = 10): #3D scatter plot for the latice
        plt.ion()
        self.M_s = M_s
        self.fig_Latice = plt.figure()
        self.ax_Latice = self.fig_Latice.add_subplot(111, projection='3d')
        x,y,z = np.where(self.L == 1)
        self.sc = self.ax_Latice.scatter(x,y,z, s=self.M_s, zdir='z', c='blue', marker=(6), alpha=0.5)
        #x,y,z = np.where(self.L == -1)
        #self.sc = self.ax_Latice.scatter(x,y,z, s=self.M_s,zdir='z', c='red', marker=(7), alpha=0.5)
        self.ax_Latice.set_xlabel('X')
        self.ax_Latice.set_ylabel('Y')
        self.ax_Latice.set_zlabel('Z')
        self.ax_Latice.set_xlim(0,self.N)
        self.ax_Latice.set_ylim(0,self.N)
        self.ax_Latice.set_zlim(0,self.N)
        plt.show()
        plt.pause(0.5)

    def Update_Fig_Latice_3D(self, T):
        #start_time = time.time()
        if not self.ax_Latice: 
            self.ax_Latice = self.fig_Latice.add_subplot(111, projection='3d')
        #self.sc._offset3d = (x,y,z)
        self.ax_Latice.clear()
        x,y,z = np.where(self.L == 1)
        self.sc = self.ax_Latice.scatter(x,y,z, s=self.M_s, zdir='z', c='blue', marker=(6), alpha=0.5)
        #x,y,z = np.where(self.L == -1)
        #self.sc = self.ax_Latice.scatter(x,y,z, s=self.M_s, zdir='z', c='red', marker=(7), alpha=0.5)
        self.ax_Latice.set_xlabel('X')
        self.ax_Latice.set_ylabel('Y')
        self.ax_Latice.set_zlabel('Z')
        self.ax_Latice.set_xlim(0,self.N)
        self.ax_Latice.set_ylim(0,self.N)
        self.ax_Latice.set_zlim(0,self.N)
        title = 'T ='+str(T)
        plt.title(title)
        plt.draw()
        #print("--- %s seconds ---" % (time.time() - start_time))
        plt.pause(0.001)


    def Plot_Latice_2D(self, M_s = 10, T=1.5): #3D scatter plot for the latice
        plt.ion()
        self.M_s = M_s # marker size
        self.fig_Latice = plt.figure()
        self.ax_Latice = self.fig_Latice.add_subplot(111)
        self.ax_Latice.imshow(self.L[int(self.N/2),:,:], vmin=-1, vmax=1)
        title = 'T ='+str(T)
        plt.title(title)
        plt.draw()
        plt.pause(0.1)

    def Update_Fig_Latice_2D(self, T):
        if not self.ax_Latice: 
            self.ax_Latice = self.fig_Latice.add_subplot(111)#, projection='2d')
        self.ax_Latice.clear()
        self.ax_Latice.imshow(self.L[int(self.N/2),:,:], vmin=-1, vmax=1)
        title = 'T ='+str(T)
        plt.title(title)
        plt.draw()
        plt.pause(0.0001)
    
    def Plot_values(self, T_start, t):
        plt.ion()
        self.fig_vals = plt.figure()
        plt.subplot(2,1,1)
        print(np.asarray(self.M_).shape)
        plt.clear()
        plt.plot(self.M_, np.arange(T_start, t+self.dT, self.dT))

        plt.subplot(2,1,2)
        plt.clear()
        plt.plot(self.M_2, np.arange(T_start, t+self.dT, self.dT))

        plt.draw()
        plt.pause(1)

        #plt.subplots(2,2,3)
        #plt.plot(self.M_4, np.arange(self.T_start, self.T_end, self.dT))

        #plt.subplots(2,2,4)
        #plt.plot(self.M_2, np.arange(self.T_start, self.T_end, self.dT))


 

print('Enable 3D-scatter plotting? (True/False) NOTE: should only be enable for small Latice size!! N < 30')
#A = str({"true":True,"false":False}[input().lower()])
A=False

print('Enable 2D-scatter plotting? (True/False) NOTE: should only be enable for small Latice size!! N < 300')
#A = str({"true":True,"false":False}[input().lower()])
B=False

print('Bench different sweep methods? (True/False)')
#do_bench = str({"true":True,"false":False}[input().lower()])
do_bench=False

print('Use cuda? (True/False)')
#C = str({"true":True,"false":False}[input().lower()])
C=False

print('Warm up?? (True/False)')
#C = str({"true":True,"false":False}[input().lower()])
warmup=True

#For best performance with numba parallel
#make N a multiple of the # of threads in the processor
print('How large Latice?')
#N = int(input())
#N=int(8)

print('Magnetisation? 1 or -1 for uniform initial magetisation, 0 for random initial magetisation')
#M = float(input())
M=1

take_every = 3
M_s = 9 # marker size
dT = 0.02
T_start = 4.3-dT
T_end = 4.7+dT
loops = 300000

n_arr=[8,16,32,48]


# benching different sweep methods
# python loops and numpy loops are WAAAY slower than
# proper numba code 
if do_bench:
    single, numb, numb_para = [], [], []
    for i in tqdm(range(2,50,1), leave=False):
        L = Latice(i,M,C)
        L.Make_Latice()
        #L.bench_test(loops=1)
        #single.append(L.bench_test(do='single'))
        numb.append(L.bench_test(do='numba'))
        numb_para.append(L.bench_test(do='numba_parallel'))
        #L.bench_test(do='cuda_parallel', loops=2)
    torch.save([numb, numb_para, np.arange(2,50,1)], 'numba_bench_laptop.pth')
    #torch.save(single, 'python_single_bench_.pth') 



for N in n_arr:
    # Create the Latice
    print(N)
    L = Latice(N,M,C)
    L.Make_Latice()
    L.temps(T_start, T_end, dT)
    scaling = N**3
    if N > 15:
        utilize ='numba_parallel'
    else: 
        utilize = 'numba'

    if A:
        L.Plot_Latice_3D(M_s=M_s)
    if B:
        L.Plot_Latice_2D(M_s=M_s, T=np.float64(T_start))

    Energy, Cv, Binder, Suscept, magnetic = [], [], [], [], []

    for t in tqdm(np.arange(T_start, T_end, dT), desc='temp_steps',  leave=False):
        #start1 = time.time()
        m, m2, m4, En, En_2 = 0.0, 0.0, 0.0, 0.0, 0.0
        cv_num, binder_num, suscept_num = 0.0, 0.0, 0.0
        if warmup:
            L.Warm_up(T=t, do=utilize, C=C)
            
        for i in tqdm(range(loops), leave=False):
            L.sweep_Latice_parallel(T=t, do_=utilize)
            if i % take_every == 0:
                #-------Energy-------
                '''En += L.E/loops
                En_2 += L.E**2*take_every/loops'''
                En, En_2 = En+L.E*take_every/loops, En_2+(L.E**2)*take_every/loops

                #-----Magnetisation-----
                '''m = m + np.abs(L.M)*take_every/loops
                m2 = m2 + np.abs(L.M)**2*take_every/loops
                m4 = m4 + np.abs(L.M)**4*take_every/loops'''
                m, m2, m4 = m + np.abs(L.M)*take_every/loops, m2 + np.abs(L.M)**2*take_every/loops, m4 + np.abs(L.M)**4*take_every/loops

            #start = time.time()

            #------- Animation(if activated)-----
            if A:
                L.Update_Fig_Latice_3D(T=t)
            if B:
                L.Update_Fig_Latice_2D(T=t)

            #print(' Plot -- %.2E'%(time.time()-start))

        En, En_2, m, m2, m4 = En/scaling, En_2/(scaling**2), m/scaling, m2/(scaling**2), m4/(scaling**4)
        cv_num = (En_2-En**2)*scaling/t
        binder_num = 1-m4/(3*m2**2)
        suscept_num = (m2-m**2)*scaling/t
        
        Energy.append(En)
        Cv.append(cv_num)
        Binder.append(binder_num)
        Suscept.append(suscept_num)
        magnetic.append(m)
        #L.Plot_values(T_start,t)
        #print(' time per temp step - %.2f'%(loops/(time.time()-start1)))


    filename = 'monte_data_'+str(N)+'cubed'+str(int(loops/1000))+'k'+str(np.arange(T_start,T_end,dT).shape)+'temps5'
    torch.save([Energy, Cv, Binder, Suscept, magnetic, np.arange(T_start,T_end,dT)], filename+'.pth')

#run_plot()

