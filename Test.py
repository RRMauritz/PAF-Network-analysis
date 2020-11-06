import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import symbols, summation, Array
import networkx as nx
import numpy as np
from random import choices, seed
from scipy.stats import binom


def paf_graph(n, Q):
    """"
    Implementation of the Albert Barabási graph with fitness
    :param n = the total vertices after the network simulation is over
    :param Q = a vector containing of the probabilities corresponding to each fitness value (ascending-> 1, 2, 3...)
    """

    # Start with a single vertex and self-loop on it
    G = nx.MultiGraph()
    G.add_edges_from(zip([0], [0]))
    # Sample n fitness parameters corresponding to the (future) vertices
    # Do this according to a probability distribution Q = [p1, p2, p3,...]
    fitnesses = choices(np.arange(1, len(Q) + 1), weights=Q, k=n)  # len(Q) +1 as right boundary not included

    # list of scaled degrees acting as prob. distr., 2*fitness value of first vertex (self loop -> deg x2)
    sc_deg = [2 * fitnesses[0]]
    # We start with the target being vertex 0
    target = 0
    # The new entering vertex, starting with label m0 (as Python is 0-based)
    source = 1
    while source < n:
        # Add edges from the source to the the m targets
        G.add_edge(source, target)
        # Increment the sc_deg list for the position of the target and source vertex
        sc_deg[target] += fitnesses[target]
        sc_deg.append(fitnesses[source])
        # Make a new list that transfer sc_deg to a probability distribution
        prob = np.array(sc_deg) / sum(sc_deg)
        # Sample m target nodes from the existing vertices with sc_deg as distribution
        target = np.random.choice(np.arange(len(sc_deg)), p=prob)
        source += 1
    return G, fitnesses


def competition_compare_PAF(n, Q, plot=False):
    # Create an instance of the PAF graph
    G, fitness = paf_graph(n, Q)

    # Compute lamb_0 based on Q => solve equation
    Q = Array(Q)
    l, j = symbols('l,j', real=True)
    eq = summation(j * Q[j - 1] / (l - j), (j, 1, len(Q)))  # The end term is included in the summation
    s = solve(eq - 1, l)
    lamb_0 = max(s)
    print("Lambda_0 = ", lamb_0)

    # For each fitness value, store the total degree
    degrees = list(dict(G.degree()).values())
    fit_link = {}
    for i in range(len(degrees)):
        if fitness[i] in fit_link:
            fit_link[fitness[i]] += degrees[i]
        else:
            fit_link[fitness[i]] = degrees[i]
    # Make a list of the fitness values and the corresponding counts
    link_k = list(fit_link.keys())
    link_v = list(fit_link.values())
    link_k, link_v = zip(*sorted(zip(link_k, link_v)))  # sort ascending
    link_v = [l / n for l in link_v]  # scale by n
    nu = [lamb_0 * Q[j - 1] / (lamb_0 - j) for j in range(1, len(Q) + 1)]

    # Make a bar plot
    if plot:
        plt.plot([i for i in range(1, len(Q) + 1)], nu, 'ro', label='Nu sequence')
        plt.bar(link_k, link_v, label='Scaled link count')
        plt.title('Test')
        plt.xlabel('Fitness value')
        plt.ylabel('Scaled link count')
        plt.legend()
        plt.show()


n = 10000
# rv = binom(6, 0.3)
# Q = rv.pmf([i for i in range(6)])

Q = [0.2, 0.4, 0.2, 0.1, 0.1]
competition_compare_PAF(n, Q, plot=True)
