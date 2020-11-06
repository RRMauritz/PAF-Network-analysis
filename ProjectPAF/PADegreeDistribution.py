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


def deg_compare_PA_sim(n, m, I):
    """
    This method generates a PA(n,m) network and computes the JSD distance between its empirical degree distribution
    and the known power-law sequence. It does this I times and returns the mean JSD value
    together with the standard error
    :param n: the number of elements in the PA network
    :param m: the number of connections each new vertex makes
    :param I: the number of simulations
    """
    jsd_values = np.zeros(I)
    for i in range(I):
        print(i)
        jsd_values[i] = deg_compare_PA(n, m, False)
    mean_jsd = np.mean(jsd_values)
    std_jsd = np.std(jsd_values)
    return mean_jsd, std_jsd


n = 50
m = 6
I = 100

mean, std = deg_compare_PA_sim(n, m, I)

print(np.round(mean, 3), np.round(std, 4))
