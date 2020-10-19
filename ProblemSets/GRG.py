import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.stats import pareto, poisson
from collections import Counter
from statsmodels.distributions.empirical_distribution import ECDF


# Generate n i.i.d weights from a pareto distribution
# w = pareto.rvs(b, size=n)


def generate_GRG(n, w):
    """"
    Generates a GRGn graph from a weight sequence w
    :param n = the number of vertices in the graph
    :param w = the weights of the vertices
    :returns the degree corresponding to every vertex
    """
    sum_w = sum(w)
    adj = np.zeros((n, n))

    # Fill the adjacency matrix according to correct edge probability
    for i in range(n):
        for j in range(n):
            edge_prob = (w[i] * w[j]) / (sum_w + w[i] * w[j])
            s = np.random.uniform(0, 1)
            if edge_prob >= s and i < j:
                adj[i, j] = 1

    # Generate the graph from the adjacency matrix
    G = nx.from_numpy_matrix(adj)
    return list(dict(G.degree()).values())


def sim_GRG(n, w, m):
    """"
    This method creates an GRGn graph m times and applies appropriate analysis to the resulting degrees
    Note that in this simulation, the weights are given as an argument and are thus fixed
    """
    deglist = np.zeros((m, n))  # store the degrees for each simulation in a row of the deglist matrix
    for sim in range(m):
        deg = generate_GRG(n, w)
        deglist[sim, :] = deg

    return deglist


# -----
n = 200
m = 100
b = 5
w = pareto.rvs(b, size=n)

# Question 3 --------------------------------------------------------------------------------
ind_max, _ = max(enumerate(w), key=operator.itemgetter(1))  # vertex label with max weight value
ind_min, _ = min(enumerate(w), key=operator.itemgetter(1))  # vertex label with min weight value

deglist = sim_GRG(n, w, m)  # run the simulation

deg_vmax = deglist[:, ind_max]  # degrees corresponding to vertex with max weight value
deg_vmin = deglist[:, ind_min]  # degrees corresponding to vertex with min weight value

ecdf_max = ECDF(deg_vmax)
ecdf_min = ECDF(deg_vmin)

rv_vmin = poisson(w[ind_min])  # random variable ~Poisson(w_i)
rv_vmax = poisson(w[ind_max])

x_vmin = np.linspace(min(deg_vmin), max(deg_vmin))
x_vmax = np.linspace(min(deg_vmax), max(deg_vmax))
cdf_vmin = rv_vmin.cdf(x_vmin)
cdf_vmax = rv_vmax.cdf(x_vmax)

# Plots
fig1, axs = plt.subplots(1, 2)
fig1.suptitle('ECDF compared to Poisson CDF for n = %d' % n)
axs[0].plot(ecdf_max.x, ecdf_max.y, label='ECDF')
axs[0].plot(x_vmax, cdf_vmax, label='Poisson(w_i)')
axs[0].set_title('Vertex with max degree')
axs[0].legend()

axs[1].plot(ecdf_min.x, ecdf_min.y, label='ECDF')
axs[1].plot(x_vmin, cdf_vmin, label='Poisson(w_i)')
axs[1].set_title('Vertex with min degree')
axs[1].legend()
for ax in axs.flat:
    ax.set(xlabel='Degrees', ylabel='ECDF')

# Question 4:
deglist = sim_GRG(n, w, m)
deglist_sorted = deglist[:, np.argsort(w)]

fig2, ax = plt.subplots()
ax.plot(np.arange(n), np.sort(w), 'bo', label='vertex weight')
label_added = False
for sim in range(m):
    if not label_added:
        ax.plot(np.arange(n), deglist_sorted[sim, :], 'ro', label='Degrees')
        label_added = True
    else:
        ax.plot(np.arange(n), deglist_sorted[sim, :], 'ro')
ax.legend()
plt.show()
