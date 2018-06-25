#### successful gaussian###
## with constant qe ####
import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import datasets
from itertools import chain
from sklearn import neighbors
from scipy.stats import multivariate_normal
var = multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
import os
os.chdir('C:/Users/Linfan/OneDrive - personalmicrosoftsoftware.ucla.edu/2018 Spring/201C/project/')

# update sigma:
def para_update(X, pi, mu, sigma, k):
    p = mu.shape[1]
    sigma_new = np.empty([k,p,p])
    mu_new = np.zeros((k, p))
    for i in range(k):
        mask = pi == i
        idx = np.arange(len(X))[mask]
        nl = sum(mask)
        # update mu
        for j in range(p):
            mu_new[i,j] = np.mean(X[mask, j])
        sigma_e = np.zeros([p,p])
        for j in idx:
            sigma_e += np.dot((X[j, ] -mu[i]).reshape((p,1)), (X[j, ] -mu[i]).reshape((1,p)))
        sigma_new[i] = sigma_e/nl
    return mu_new, sigma_new

def likelihood(mu, sigma):
    ll = list()
    for i in range(k):
        ll.append(multivariate_normal(mean=mu[i], cov=sigma[i]))  
    return ll
    
def SWC(G, k, pi, qe, ll):        
    G1 = G.copy()
    for e in G.edges(): # remove edges
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
    q = np.zeros(k)
    for i in range(k):
        pi_temp = pi.copy()
        pi_temp[list(V0)] = i
        # data likelihood
        for j in range(n):
            q[i] += np.log(ll[pi_temp[j]].pdf(X[j, ]))
    q = np.exp(q)
    q = q/sum(q)   
      
    pi_new = pi.copy()
    label_new = np.random.choice(k, p= q)
    pi_new[list(V0)] = label_new
    #print("V=", V)
    #print("V0=", V0)
#    print("alpha= ", alpha)
    #print("label_old=", pi[list(V0)[0]])
    #print("label_new=", label_new)
    #print(pi_new == pi)
    
    return pi_new

###load data###   
iris = datasets.load_iris()
X = iris.data
n = len(X)

k = 3 # number of clusters
Nit = 100
temp = neighbors.kneighbors_graph(X, 7, mode='distance')
G = nx.from_numpy_matrix(temp.toarray())

# initialize cluster 
pi = np.random.choice(np.arange(k), n)
# initialize data likelihood function:
mu_temp = [np.mean(X[:,0]), np.mean(X[:,1]), np.mean(X[:,2]), np.mean(X[:,3])]
mu = np.array([mu_temp, mu_temp, mu_temp])
sigma = np.array([np.diag([1,1,1,1]), np.diag([1,1,1,1]), np.diag([1,1,1,1])])
ll = likelihood(mu, sigma)

for i in range(1000):
    pi = SWC(G, k, pi, 0.5, ll)
    mu, sigma = para_update(X, pi, mu, sigma, k)
    ll = likelihood(mu, sigma)
    
plt.plot(X[pi == 0, 0], X[pi == 0,1],'ro', color="green")
plt.plot(X[pi == 1, 0], X[pi == 1,1],'ro', color="blue")
plt.plot(X[pi == 2, 0], X[pi == 2,1],'ro', color="red")
plt.show()

#pi_true = iris.target
#
#plt.plot(X[pi_true == 0, 0], X[pi_true == 0,1],'ro', color="red")
#plt.plot(X[pi_true == 1, 0], X[pi_true == 1,1],'ro', color="blue")
#plt.plot(X[pi_true == 2, 0], X[pi_true == 2,1],'ro', color="green")   
#plt.show()

