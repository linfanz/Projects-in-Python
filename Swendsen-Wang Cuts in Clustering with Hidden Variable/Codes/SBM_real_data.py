## version 1###
### success ###
import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
from itertools import chain
from sklearn import neighbors
from matplotlib import colors
import time
from statistics import mode
import matplotlib.cm as cm
import os
os.chdir('C:/Users/Linfan/OneDrive - personalmicrosoftsoftware.ucla.edu/2018 Spring/201C/project/')

def SWC(G, k, pi, gamma):        
    G1 = G.copy()
    for e in G.edges(): # remove edges
        qe = gamma[pi[e[0]], pi[e[1]]]
        if pi[e[0]] == pi[e[1]] and np.random.choice(2, p = [qe, 1-qe]):
            G1.remove_edge(*e)
    
    V = [None] * k #store CP in each cluster
    CP = list() #store CP as a whole
    for i in range(k):
        G_temp = G1.subgraph(np.array([v for v in G1.nodes()])[pi==i]) # extract subgraph in the same cluster 
        V[i] = [e for e in nx.connected_components(G_temp)] # CP in each cluster
        CP = CP + V[i] # construct connected components
                            
    # select a random CP
    idx = np.random.choice(len(CP))
    V0 = CP[idx]
    label_old = pi[list(V0)[0]]
    # propose a new cluster to V0
    label_new = np.random.choice(k)
         
    q = 1
    pi_new = pi.copy()
    pi_new[list(V0)] = label_new
                                
    # proposal
    for s in list(V0):
        for t in range(n):
            p_new = gamma[pi_new[s],pi_new[t]]**X[s,t] * (1-gamma[pi_new[s],pi_new[t]])**(1-X[s,t])
            p_old = gamma[pi[s],pi[t]]**X[s,t] * (1-gamma[pi[s],pi[t]])**(1-X[s,t])
            q *= (p_new/p_old)
                                
    alpha = min(1,q)
    
    if alpha > np.random.random(1):  
        return pi_new
    else:
        return pi            
           
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
    #print("new_gamma = ", new_gamma)
    return new_gamma

#### load data ####
k = 42 # number of clusters
email = np.loadtxt('email-Eu-core.txt', dtype=np.int16)
true_labels = np.loadtxt('email-Eu-core-department-labels.txt', dtype=np.int16)
n = len(true_labels)
G = nx.Graph()
for i in range(len(email)):
    G.add_edge(email[i,0], email[i,1])

X = nx.to_numpy_matrix(G)
np.allclose(X, X.T)
np.fill_diagonal(X, 0)

#np.savetxt('email_matrix.txt', X, fmt="%i")

G = nx.from_numpy_matrix(X)
# initialize cluster 
pi = np.random.choice(k, n)
# initialize connectivity
gamma = 0.01*np.ones((n,n))
np.fill_diagonal(gamma, 0.5)

start_time = time.time()
for i in range(500):
    pi = SWC(G, k, pi, gamma)
    gamma = para_update(X, pi, k)
print("--- %s seconds ---" % (time.time() - start_time))
    
#np.savetxt('output_email.txt', pi, fmt="%i")

est_labels = np.loadtxt('output_email.txt', dtype=np.int16)



nx.draw_networkx(G, with_labels=False, nodelist = list(np.arange(1005)[true_labels[:,1] == mode(true_labels[:,1])]))

nx.draw_networkx(G, with_labels=False, nodelist = list(np.arange(1005)[est_labels == mode(est_labels)]))

mode(true_labels[:,1])
mode(est_labels)

rr = 0
for i in range(1005):
    if true_labels[:,1][i] == 4 and est_labels[i] == 41:
        rr += 1
rr/sum(true_labels[:,1] == 4)

nx.draw_networkx(G, with_labels=False, node_color = true_labels[:,1], node_size = 100)

nx.draw_networkx(G, with_labels=False, node_color = est_labels, node_size = 100)

nx.draw_networkx(G, with_labels=False, node_size = 100, edge_color = "black")

G1 = G.subgraph(list(np.arange(1005)[true_labels[:,1] == 4]))
fig = plt.figure()   
nx.draw_networkx(G1)
plt.title('Largest group (true)')
fig.savefig('5-SWC-true group.png')

G2 = G.subgraph(list(np.arange(1005)[est_labels == 14]))
fig = plt.figure()
nx.draw_networkx(G2)
plt.title('Largest group (estimated)')
fig.savefig('6-SWC-estimated.png')

nx.draw_networkx(G, nodelist = list([0, 1, 2, 3, 4]))
