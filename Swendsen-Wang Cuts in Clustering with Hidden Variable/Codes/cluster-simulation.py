#### successful gaussian###
## with constant qe ####
import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
from itertools import chain
from sklearn import neighbors
from scipy.stats import multivariate_normal
import os
os.chdir('C:/Users/Linfan/OneDrive - personalmicrosoftsoftware.ucla.edu/2018 Spring/201C/project/')

# update parameter:
def para_update(X, pi, mu, sigma, k):
    sigma_new = np.empty([k,2,2])
    mu_new = np.zeros((k, 2))
    for i in range(k):
        mask = pi == i
        idx = np.arange(len(X))[mask]
        nl = sum(mask)
        mu_new[i] = [np.mean(X[mask, 0]), np.mean(X[mask, 1])]
        sigma_e = np.zeros([2,2])
        for j in idx:
            sigma_e += np.dot((X[j, ] -mu[i]).reshape((2,1)), (X[j, ] -mu[i]).reshape((1,2)))
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
    label_new = np.random.choice(k, p=q)
    pi_new[list(V0)] = label_new
    return pi_new, q[label_new]


qe = 0.5
X = np.loadtxt('Dataset-1.txt')
n = len(X)

k = 3 # number of clusters
temp = neighbors.kneighbors_graph(X, 4, mode='distance')
G = nx.from_numpy_matrix(temp.toarray())

# initialize cluster 
pi = np.random.choice(np.arange(k), n)
# initialize data likelihood function:
mu = np.array([(10,3), (2,2), (3,1)])
sigma = np.array([np.diag([1,1]), np.diag([1,1]), np.diag([1,1])])
ll = likelihood(mu, sigma)

pi_max = np.array([])
prob_max = 0
for l in range(50):
    for i in range(100):
        pi, prob = SWC(G, k, pi, 0.8, ll)
        
        mu, sigma = para_update(X, pi, mu, sigma, k)
        ll = likelihood(mu, sigma)
    
    if prob > prob_max:
            prob_max = prob
            pi_max = pi

fig = plt.figure()   
plt.plot(X[pi_max == 0, 0], X[pi_max == 0,1],'ro', color="red")
plt.plot(X[pi_max == 1, 0], X[pi_max == 1,1],'ro', color="blue")
plt.plot(X[pi_max == 2, 0], X[pi_max == 2,1],'ro', color="green")
plt.xlabel('y1')
plt.ylabel('y2')
plt.title('Clusetered Data Points with Poor Initials (SWC)')
fig.savefig('2-SWC-cluster.png')

    

