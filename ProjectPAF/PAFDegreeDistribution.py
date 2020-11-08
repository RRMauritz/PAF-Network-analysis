from ProjectPAF.Graph_Generation import paf_graph, show_PAF
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sympy.solvers import solve
from sympy import symbols, summation, Array, Abs
from scipy.stats import kurtosis


def deg_compare_PAF(n, Q, plot=False):
    """
    Creates a PAF network and computes its degree sequence for each fitness-value
    :param n: the number of vertices in the PAF graph
    :param plot: if True, then the degree sequence will be plotted via a bar plot
    """
    G, fitness = paf_graph(n, Q)
    degrees = list(dict(G.degree()).values())  # list of degree values for all vertices
    fit_deg = {}
    for j in range(1, len(Q) + 1):  # because fitness-value start at 1
        # For each item in the degree list, store it in this j-list if the corresponding fitness value equals j
        degrees_j = [degrees[k] for k in range(len(degrees)) if fitness[k] == j]
        deg_count_j = Counter(degrees_j)
        deg_k_j = list(deg_count_j.keys())
        deg_v_j = list(deg_count_j.values())
        deg_k_j, deg_v_j = zip(*sorted(zip(deg_k_j, deg_v_j)))
        deg_v_j = [v / n for v in deg_v_j]  # TODO: good that we divide by n?
        fit_deg[j] = (deg_k_j, deg_v_j)  # store it with key = j

    # First calculate the nu_sequence
    Qs = Array(Q)
    l, j = symbols('l,j')
    # The end term is included in the summation, the first entry of Q corresponds to j = 1
    eq = summation(j * Qs[j - 1] / (l - j), (j, 1, len(Q)))
    lamb_0 = max(np.abs(solve(eq - 1, l)))

    nu = [lamb_0 * Q[j - 1] / (lamb_0 - j) for j in range(1, len(Q) + 1)]

    # Then calculate the eta-sequence, store it in a dict with the fv being the key
    eta = {}
    for j in range(1, len(Q) + 1):  # because the fitness-value starts at 1
        eta[j] = [nu[j - 1] * (1 / k) * np.product(np.array([l / (l + lamb_0 * (1 / j)) for l in range(2, k + 1)]))
                  for k in fit_deg[j][0]]

    # Degree distribution for each fitnessvalue

    if plot:
        ncol = 3
        nrow = 3
        fig, axs = plt.subplots(nrow, ncol)
        axs = axs.ravel()
        for j in range(1, len(Q) + 1):
            axs[j - 1].bar(fit_deg[j][0], fit_deg[j][1])
            axs[j - 1].plot(fit_deg[j][0], eta[j], 'r')
            axs[j - 1].set_title('Fitness = %i' % j)
            axs[j - 1].set_xlim([0, 20])
        plt.show()
    return eta


def competition_compare_PAF(n, Q, lamb_0=None, plot=False):
    """"
    Optional to give lamb_0 to the function so that it is not recalculated each time
    """
    # Create an instance of the PAF graph
    G, fitness = paf_graph(n, Q)

    # Compute lamb_0 based on Q => solve equation
    if lamb_0 is None:
        Qs = Array(Q)
        l, j = symbols('l,j')
        eq = summation(j * Qs[j - 1] / (l - j), (j, 1, len(Q)))  # The end term is included in the summation
        lamb_0 = max(np.abs(solve(eq - 1, l)))  # TODO: not entirely sure if taking the abs value is valid

    # For each fitness value, store the total degree
    degrees = list(dict(G.degree()).values())
    fit_link = {el: 0 for el in range(1, len(Q) + 1)}  # for each fitness-value, pre-set it's value to 0
    for i in range(len(degrees)):
        fit_link[fitness[i]] += degrees[i]

    # Make a list of the fitness values and the corresponding counts
    link_k = list(fit_link.keys())
    link_v = list(fit_link.values())
    link_k, link_v = zip(*sorted(zip(link_k, link_v)))  # sort ascending
    link_v = [l / n for l in link_v]  # scale by n
    nu = [lamb_0 * Q[j - 1] / (lamb_0 - j) for j in range(1, len(Q) + 1)]

    abs_difference = np.abs(np.array(nu) - np.array(link_v))
    ABS = sum(abs_difference)
    # Make a bar plot
    if plot:
        plt.plot([i for i in range(1, len(Q) + 1)], nu, 'ro', label='Nu sequence')
        plt.bar(link_k, link_v, label='Scaled link count')
        plt.title('Scaled link count per fitness value, n = %i, ABS = %5.4f' % (n, ABS))
        plt.xlabel('Fitness value')
        plt.ylabel('Scaled link count')
        plt.legend()
        plt.show()
    return ABS


def competition_compare_PAF_sim(n, Q, I):
    # Solve for lamb_0 and give it as argument to competition_compare_PAF
    # so that we do not have to recalculate it each time
    Qs = Array(Q)
    l, j = symbols('l,j', real=True)
    eq = summation(j * Qs[j - 1] / (l - j), (j, 1, len(Q)))  # The end term is included in the summation
    lamb_0 = max(solve(eq - 1, l))

    abs_values = np.zeros(I)
    for i in range(I):
        abs_values[i] = competition_compare_PAF(n, Q, lamb_0, False)
    mean_abs = np.mean(abs_values)
    std_abs = np.std(abs_values)
    return mean_abs, std_abs

