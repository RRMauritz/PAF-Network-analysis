import matplotlib.pyplot as plt
from ProjectPAF.Graph_Generation import *
from collections import Counter
from scipy.spatial.distance import jensenshannon


def deg_compare_PA(n, m, plot=False):
    """"
    Compares the empirical degree distribution of the PA(n,m) model with the power law sequence
    Returns the JSD between the two and executes a plot depending on the parameter 'plot'
    :param n: the number of elements in the PA network
    :param m: the number of connections each new vertex makes
    :param plot: if True, this method will plot the degree and power law sequence
    """
    G = ba_graph_oi(n, m)  # Bollobás & Riordan 2004 Model!

    degrees = list(dict(G.degree()).values())  # list of degree values for all vertices
    deg_count = Counter(degrees)  # Count each occurrence
    deg_k = list(deg_count.keys())  # unique degrees (key)
    deg_v = list(deg_count.values())  # number of occurrences of each unique degree (value)

    deg_k, deg_v = zip(*sorted(zip(deg_k, deg_v)))  # sort the two together based on deg_k ascending
    deg_v = [v / n for v in deg_v]  # normalize the bars

    power = [(2 * m * (m + 1)) / (k * (k + 1) * (k + 2)) for k in
             deg_k]  # power law sequence (theoretical limiting distr.)

    jsd = jensenshannon(deg_v, power)  # Compute the Jenshen-Shannon divergence between the two

    if plot:
        plt.bar(deg_k, deg_v, label='Degree frequency realization')
        plt.plot(deg_k, power, 'r', label='Power law sequence')
        plt.xlim([m - 1, 50])
        plt.title('Degree sequence of PA network; n = %d, m = %d, JSD = %5.4f' % (n, m, jsd))
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    return jsd


n = 1000
m = 10

jsd = deg_compare_PA(n, m, False)


