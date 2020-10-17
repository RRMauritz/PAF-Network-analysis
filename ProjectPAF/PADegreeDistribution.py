from ProjectPAF.PAF_Generation import *
import matplotlib.pyplot as plt
from collections import Counter

n = 1000
m = 10
m0 = 3
I = 50


# G = ba_graph(n, m)
#
# degrees = list(dict(G.degree()).values())  # degree values for all vertices
#
# deg_count = Counter(degrees)
# deg_k = list(deg_count.keys())  # unique degrees
# deg_v = list(deg_count.values())  # number of occurences of each unique degree
# deg_v = [v / n for v in deg_v]  # normalize the bars
#
# power = [(2 * m * (m + 1)) / (k * (k + 1) * (k + 2)) for k in deg_k]
#
# plt.bar(deg_k, deg_v)
# plt.plot(deg_k, power, 'ro')
# plt.xlim([0, 30])
# plt.show()

def sim_deg_distr_PA(n, m, I):
    """"
    Simulate the degree distribution of the PA graph #I times and plots all the degree sequences over each other
    """
    # For each iteration we need to store the degrees and its counts
    D = {}
    for i in range(I):
        G = ba_graph_oi(n, m)
        degrees = list(dict(G.degree()).values())
        deg_count = Counter(degrees)
        deg_k = list(deg_count.keys())
        deg_v = list(deg_count.values())
        D[i] = (deg_k, deg_v)

    for i in range(I):
        deg_k, deg_v = zip(*sorted(zip(D[i][0], D[i][1])))
        plt.plot(deg_k, deg_v)
    plt.xlim([0, 50])
    plt.xlabel('Degree')
    plt.ylabel('Proportion of vertices with that degree')
    plt.title('Degree sequences of the Preferential Attachment model')
    plt.show()


sim_deg_distr_PA(n, m, I)
