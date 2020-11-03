from ProjectPAF.Graph_Generation import paf_graph
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sympy.solvers import solve
from sympy import symbols, summation, Array
from scipy.stats import binom


def deg_compare_PAF(n, Q, plot=False):
    """
    Creates a PAF network and computes its degree sequence
    :param n: the number of vertices in the PAF graph
    :param plot: if True, then the degree sequence will be plotted via a bar plot
    """
    G, fitness = paf_graph(n, Q)

    degrees = list(dict(G.degree()).values())  # list of degree values for all vertices
    fit_deg = {}
    for j in range(1, len(Q) + 1):
        degrees_j = [degrees[k] for k in range(len(degrees)) if fitness[k] == j]
        deg_count_j = Counter(degrees_j)
        deg_k_j = list(deg_count_j.keys())
        deg_v_j = list(deg_count_j.values())
        deg_k_j, deg_v_j = zip(*sorted(zip(deg_k_j, deg_v_j)))
        deg_v_j = [v / len(degrees_j) for v in
                   deg_v_j]  # TODO: good that we divide by len(degrees_j) instead of n? Probably not!
        fit_deg[j - 1] = (deg_k_j, deg_v_j)

    deg_count = Counter(degrees)  # Count each occurrence
    deg_k = list(deg_count.keys())  # unique degrees (key)
    deg_v = list(deg_count.values())  # number of occurrences of each unique degree (value)

    deg_k, deg_v = zip(*sorted(zip(deg_k, deg_v)))  # sort the two together based on deg_k ascending
    deg_v = [v / n for v in deg_v]  # normalize the bars

    # First calculate the nu_sequence
    Q = Array(Q)
    l, j = symbols('l,j', real=True)
    eq = summation(j * Q[j - 1] / (l - j), (j, 1, len(Q)))  # The end term is included in the summation
    lamb_0 = max(solve(eq - 1, l))

    nu = [lamb_0 * Q[j - 1] / (lamb_0 - j) for j in range(1, len(Q) + 1)]
    # Then calculate the eta-sequence
    eta = {}
    for j in range(len(Q)):
        eta[j] = [nu[j] * (1 / k) * np.product(np.array([l / (l + lamb_0 * (1 / (j + 1))) for l in range(2, k + 1)]))
                  for k in deg_k]
    # Degree distribution for each fitnessvalue

    if plot:
        ncol = 2
        nrow = 2
        fig, axs = plt.subplots(nrow, ncol)
        axs = axs.ravel()
        for j in range(1, len(Q) + 1):
            axs[j - 1].bar(fit_deg[j - 1][0], fit_deg[j - 1][1])
            axs[j - 1].plot(deg_k, eta[j - 1], 'r')
            axs[j - 1].set_title('Fitness = %i' % j)
            axs[j - 1].set_xlim([0, 20])
        plt.show()
    return eta


def competition_compare_PAF(n, Q, plot=False):
    # Create an instance of the PAF graph
    G, fitness = paf_graph(n, Q)

    # Compute lamb_0 based on Q
    Q = Array(Q)
    l, j = symbols('l,j', real=True)
    eq = summation(j * Q[j - 1] / (l - j), (j, 1, len(Q)))  # The end term is included in the summation
    lamb_0 = max(solve(eq - 1, l))
    print("Lambda_0 = ", lamb_0)

    # For each fitness value, store the total degree
    degrees = list(dict(G.degree()).values())
    fit_link = {}
    # THIS IS CORRECT:
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
        plt.title('Scaled link count per fitness value')
        plt.xlabel('Fitness value')
        plt.ylabel('Scaled link count')
        plt.legend()
        plt.show()


def show_PAF(n, Q):
    G, fitness = paf_graph(n, Q)

    pos = nx.spring_layout(G)
    fitness_labels = dict(zip([i for i in range(len(fitness))], fitness))
    nx.draw(G, pos, node_color=fitness, cmap=plt.cm.Reds_r)
    nx.draw_networkx_labels(G, pos, fitness_labels)
    plt.show()


# ---------------------------------------------------------
# rv = binom(6, 0.3)
# Q = rv.pmf([i for i in range(6)])
Q = [0.4, 0.3, 0.2, 0.1]

n = 30000
eta = deg_compare_PAF(n, Q, plot=True)

# show_PAF(n,Q)
# competition_compare_PAF(n, Q, plot=True)
# Q = [0.1, 0.3, 0.2, 0.3, 0.1]
# Q = [0.6, 0.19, 0.21]  # TODO: problems with imaginary numbers -> this isn't mentioned in the paper
