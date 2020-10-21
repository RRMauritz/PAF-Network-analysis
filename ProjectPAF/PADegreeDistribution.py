import matplotlib.pyplot as plt
from ProjectPAF.PAF_Generation import *
from collections import Counter
from scipy.spatial.distance import jensenshannon


n = 10000
m = 10
m0 = 3
I = 50

G = ba_graph_oi(n, m)

degrees = list(dict(G.degree()).values())  # degree values for all vertices
deg_count = Counter(degrees)
deg_k = list(deg_count.keys())  # unique degrees
deg_v = list(deg_count.values())  # number of occurences of each unique degree

deg_k, deg_v = zip(*sorted(zip(deg_k, deg_v)))  # sort the two together
deg_v = [v / n for v in deg_v]  # normalize the bars

power = [(2 * m * (m + 1)) / (k * (k + 1) * (k + 2)) for k in deg_k]

jsd = jensenshannon(deg_v, power)

plt.bar(deg_k, deg_v, label='Degree frequency realization')
plt.plot(deg_k, power, 'r', label='Power law sequence')
plt.xlim([m - 1, 50])
plt.title('Degree sequence of PA network; n = %d, m = %d, JSD = %5.4f' % (n, m, jsd))
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.legend()
plt.show()


