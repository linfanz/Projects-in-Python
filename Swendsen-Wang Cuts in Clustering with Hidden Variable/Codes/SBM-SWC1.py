### mostly successful try on stochastic block model####
## generalized gibbs version ###
### SWC2####
import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
from itertools import chain
from sklearn import neighbors
from scipy.stats import multivariate_normal
var = multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
import os
os.chdir('C:/Users/Linfan/OneDrive - personalmicrosoftsoftware.ucla.edu/2018 Spring/201C/project/')

# update sigma:
def para_update(X, pi, k):
    gamma = np.zeros([k,k])
    count = np.zeros([k,k])
    for i in range(n):
        for j in range(n):
            if j > i: # because of the symmetry
                m = pi[i]  
                l = pi[j]
                count[m,l] += 1
                count[l,m] += 1
                if X[i,j] == 1:
                    gamma[m,l] += 1
                    gamma[l,m] += 1
    new_gamma = gamma/count
    new_gamma[np.isnan(new_gamma)] = 0
    print("new_gamma = ", new_gamma)
    return new_gamma
    
def SWC(G, k, pi, gamma):        
    G1 = G.copy()
    for e in G.edges(): # remove edges
        qe = gamma[pi[e[0]], pi[e[1]]]
        if pi[e[0]] == pi[e[1]] and np.random.choice(2, p = [qe, 1-qe]):
            G1.remove_edge(*e)
    
    V = [None] * k #store CP in each cluster
    CP = list() # store CP as a whole
    for i in range(k):
        G_temp = G1.subgraph(np.array([v for v in G1.nodes()])[pi==i]) # extract subgraph in the same cluster 
        V[i] = [e for e in nx.connected_components(G_temp)] # CP in each cluster
        CP = CP + V[i] # construct connected components
                            
    # select a random CP
    idx = np.random.choice(len(CP))
    V0 = CP[idx]
    
    # propose probability
    q = np.ones(k)
    for i in range(k):
        pi_temp = pi.copy()
        pi_temp[list(V0)] = i
        # data likelihood
        for s in range(n):
            for t in range(n):
                if t > s:
                    temp = gamma[pi_temp[s],pi_temp[t]]**X[s,t] * (1-gamma[pi_temp[s],pi_temp[t]])**(1-X[s,t])
                    q[i] *= temp
    
#    q = np.exp(q)
    q1 = q/sum(q) 
    
    #print("q_true = ", q)            
    pi_new = pi.copy()
    label_old = pi[list(V0)[0]]

    label_new = np.random.choice(k, p = q1)
    pi_new[list(V0)] = label_new
    #print("V=", V)
    #print("V0=", V0)
#    print("alpha= ", alpha)
    #print("label_old=", label_old)
    #print("label_new=", label_new)
    #print(pi_new == pi)    
    return pi_new, q[label_new]

#### simulation set up####
#k = 3 # number of clusters
#p_t = np.array([0.2, 0.5, 0.3])
#gamma_t = np.array([(0.7, 0.01, 0.01), (0.01, 0.7, 0.02), (0.01, 0.02, 0.7)])
#X = np.zeros([50, 50])
#n = len(X)
#pi_t = np.random.choice(k, size=n, p=p_t) 
#for i in range(n):
#    for j in range(n):
#        if j > i: # because of the symmetry
#            temp = gamma_t[pi_t[i], pi_t[j]]
#            X[i,j] = np.random.choice(2, p = [1-temp, temp])

###############################################################
X = np.loadtxt('SBM.txt')
pi_t = np.loadtxt("true_label.txt")
pi_t = pi_t-1
pi_t=np.array(list(map(int, pi_t)))
n = len(X)
k = 3
G = nx.from_numpy_matrix(X)

# initialize cluster 
pi = np.random.choice(k, n)
# initialize connectivity
gamma = np.array([(0.5, 0.1, 0.1), (0.1, 0.5, 0.1), (0.1, 0.1, 0.5)])

for i in range(500):
    pi, prob = SWC(G, k, pi, gamma)
    gamma = para_update(X, pi, k)
    
rr0 = 0
rr1 = 0
rr2 = 0

for i in range(len(pi)):
    if pi_t[i] == 0:
        rr0 += (pi[i] == 0)
    if pi_t[i] == 1:
        rr1 += (pi[i] == 2)
    if pi_t[i] == 2:
        rr2 += (pi[i] == 1)

rr0/sum(pi_t == 0)    
rr1/sum(pi_t == 1)       
rr2/sum(pi_t == 2)

sum(pi==0)/len(pi_t) #pi_0
sum(pi==2)/len(pi_t) #pi_1
sum(pi==1)/len(pi_t) #pi_2
