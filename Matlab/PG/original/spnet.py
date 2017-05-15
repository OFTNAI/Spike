import numpy as np
import random
import matplotlib.pyplot as plt
import time
from copy import copy

# spnet.m: Spiking network with axonal conduction delays and STDP
# Created by Eugene M.Izhikevich.                February 3, 2004
# Modified to allow arbitrary delay distributions.  April 16,2008
# converted to Pyhton by Martin Pyka             January 24, 2013

random.seed(1)
t0 = 0

def tic():
    global t0
    t0 = time.clock()

def toc():
    global t0
    print time.clock() - t0, "seconds"

def cell(rows, cols):
    return [[ np.array([]) for i in range(cols)] for j in range(rows)]


def ind2sub(shape, ind):
    """ From the given shape, returns the subscrips of the given index"""
    return ind % shape[0], np.floor(ind/shape[0]).astype(int)


dur = 1

def spnet(dur):
    M = 100         # number of synapses per neuron
    D = 20          # maximal conduction delay
    Ne = 800        # excitatory neurons    
    Ni = 200        # inhibitory neurons      
    N = Ne + Ni     # total number
    
    a = np.concatenate([0.02 * np.ones([Ne,1]), 0.1*np.ones([Ni,1])])
    d = np.concatenate([8 * np.ones([Ne,1]), 2 * np.ones([Ni,1])])
    sm = 10         # maximal synaptic strength
    
    print("Setup post and delay")
    tic()
    
    post = np.zeros([N, M])
    delays = cell(N, D)
    for i in xrange(Ne):
        p = random.sample(range(N), M)
        post[i, :] = p
        for j in xrange(M):
            r = int(np.floor(D*random.random()))
            delays[i][r] = np.hstack([delays[i][r], j]).astype(int)
    
    for i in range(Ne, N):
        p = random.sample(range(Ne), M)
        post[i,:] = p
        delays[i][0] = range(M)
        
    post = post.astype(int)
    s = np.concatenate([6 * np.ones([Ne,M]), -5*np.ones([Ni,M])])
    sd = np.zeros([N, M])
    
    toc()
    print("Setup pre and aux")
    tic()
    
    pre = cell(N, 2)
    aux = cell(N, 1)
    
    for i in xrange(Ne):
        for j in xrange(D):
            for k in xrange(len(delays[i][j])):
                pre[ int(post[i][ delays[i][j][k] ]) ] = np.hstack([pre[ int(post[i][ delays[i][j][k] ]) ], 
                                                                       ind2sub(sd.shape, np.array([N*(delays[i][j][k])+i]))]).astype(int)
                aux[ int(post[i][ delays[i][j][k] ]) ][0] = np.hstack([aux[ int(post[i][ delays[i][j][k] ]) ][0],
                                                                       N*(D-1-j)+i]).astype(int)  # takes into account delay, maybe needs to be checked again, if correctly converted
    
    STDP = np.zeros([N, 1001+D])
    v = -65 * np.ones([N,1])
    u = 0.2 * v
    firings = np.array([[-D, 0]])
    
    toc()
    print("Run simulation")
    tic()
    
    # Figure setup
    ###########################
    plt.ion()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data, = ax.plot(firings[:,0], firings[:,1], '.')
    ax.axis([0, 1000, 0, N])
    ###########################
    
    
    for sec in xrange(dur):         # stimulation for a certain number of seconds
        print(sec)
        for t in xrange(1000):       # stimulation of 1 sec
            I = np.zeros([N, 1])
            I[np.floor(N*random.random())] = 20;
            fired = np.array([np.nonzero(v>5)[0]])
            v[fired] = -65
            u[fired] =u[fired] + d[fired]
            STDP[fired,t+D] = 0.1
            for k in fired[0]:
                #subx, suby = ind2sub(sd.shape, pre[k][0])
                sub2x, sub2y = ind2sub(np.shape(STDP), N*t+aux[k][0])
                sd[pre[k][0], pre[k][1]] = sd[pre[k][0], pre[k][1]] + STDP[sub2x, sub2y]
                
            firings = np.concatenate([firings, np.concatenate([t*np.ones([len(fired[0]),1]),fired.transpose()], axis=1)]).astype(int)
            
            k = firings.shape[0]-1
    
            while (firings[k,0]>t-D):
                del_ind = delays[firings[k,1]][t-firings[k,0]]
                if (len(del_ind)>0):
                    ind = post[firings[k,1], del_ind]
                    I[ind]=I[ind]+np.array([s[firings[k,1],del_ind]]).transpose()
                    sd[firings[k,1],del_ind] = sd[firings[k,1],del_ind] - 1.2*STDP[ind,t+D]
                k=k-1
                
            v = v+0.5*((0.04*v+5)*v + 140 - u + I)      # for numerical
            v = v+0.5*((0.04*v+5)*v + 140 - u + I)      # stability time
            u = u + a * (0.2*v -u)                      # step is 0.5 ms
            STDP[:,t+D+1] = 0.95 * STDP[:,t+D]          # tau = 20ms
    
        # Update figure ##################################
        data.set_data(firings[:,0], firings[:,1])
        fig.canvas.draw()
        ##################################################
            
        STDP[:,0:(D+1)] = STDP[:,1000:(1000+D+1)]
        ind = np.nonzero(firings[:,0]>1000-D)[0]
        firings = np.concatenate([np.array([[-D, 0]]), np.array([firings[ind,0]-1000, firings[ind,1]]).T])
        s[0:Ne,:] = np.maximum(
                               np.minimum(0.01 + s[0:Ne,:] + sd[0:Ne,:], sm), 
                               0)
        sd = 0.9 * sd
        toc()
        
spnet(10)