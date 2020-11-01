from ProjectPAF.Graph_Generation import paf_graph
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx

from sympy.solvers import solve
from sympy import symbols, summation, Array
from scipy.stats import binom


def deg_compare_PAF(n, plot=False):
    """
    Creates a PAF network and computes its degree sequence
    :param n: the number of vertices in the PAF graph
    :param plot: if True, then the degree sequence will be plotted via a bar plot
    """
    G, fitness, _ = paf_graph(n)

    degrees = list(dict(G.degree()).values())  # list of degree values for all vertices
    deg_count = Counter(degrees)  # Count each occurrence
    deg_k = list(deg_count.keys())  # unique degrees (key)
    deg_v = list(deg_count.values())  # number of occurrences of each unique degree (value)

    deg_k, deg_v = zip(*sorted(zip(deg_k, deg_v)))  # sort the two together based on deg_k ascending
    deg_v = [v / n for v in deg_v]  # normalize the bars

    if plot:
        plt.bar(deg_k, deg_v, label='Degree frequency realization')
        plt.title('Degree sequence of PA network')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    return G, fitness


def competition_compare_PAF(n, plot=False):
    # Create an instance of the PAF graph
    G, fitness, Q = paf_graph(n)

    # Compute lamb_0 based on Q
    Q = Array(Q)
    l, j = symbols('l,j')
    eq = summation(j * Q[j] / (l - j), (j, 0, 10))
    lamb_0 = max(solve(eq - 1, l))

    print(lamb_0)
    # For each fitness value, store the total degree
    degrees = list(dict(G.degree()).values())
    fit_deg = {}
    # THIS IS CORRECT:
    for i in range(len(degrees)):
        if fitness[i] in fit_deg:
            fit_deg[fitness[i]] += degrees[i]
        else:
            fit_deg[fitness[i]] = degrees[i]
    # Make a list of the fitness values and the corresponding counts
    link_k = list(fit_deg.keys())
    link_v = list(fit_deg.values())
    link_k, link_v = zip(*sorted(zip(link_k, link_v)))  # sort ascending
    link_v = [l / n for l in link_v]  # scale by n
    nu = [lamb_0 * Q[j] / (lamb_0 - j) for j in range(len(Q))]
    # Make a bar plot
    if plot:
        plt.plot([i for i in range(1, len(Q) + 1)], nu, 'ro', label='Nu sequence')
        plt.bar(link_k, link_v, label='Scaled link count')
        plt.title('Scaled link count per fitness value')
        plt.xlabel('Fitness value')
        plt.ylabel('Scaled link count')
        plt.legend()
        plt.show()


def show_PAF(n):
    G, fitness, Q = paf_graph(n)

    s = sum([j * Q[j] / (10.000119450901292805590966427892 - j) for j in range(0, 11)])
    print(s)

    pos = nx.spring_layout(G)
    fitness_labels = dict(zip([i for i in range(len(fitness))], fitness))
    nx.draw(G, pos, node_color=fitness, cmap=plt.cm.Reds_r)
    nx.draw_networkx_labels(G, pos, fitness_labels)
    plt.show()


n = 10000
# show_PAF(n)
competition_compare_PAF(n, plot=True)
# deg_compare_PAF(n,plot=True)
