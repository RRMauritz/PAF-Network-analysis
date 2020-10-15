from ProjectPAF.PAF_Generation import *
import matplotlib.pyplot as plt
from collections import Counter

n = 1000
m = 2  # TODO: currently degree distr. only converges to k-formula for n = 1 -> page 255/256 CN book
m0 = 3

G = ba_graph_oi(n, m)

degrees = list(dict(G.degree()).values())  # degree values for all vertices

deg_count = Counter(degrees)
deg_k = list(deg_count.keys())  # unique degrees
deg_v = list(deg_count.values())  # number of occurences of each unique degree
deg_v = [v / n for v in deg_v]  # normalize the bars

power = [2*(m+1) / (k * (k + 1) * (k + 2)) for k in deg_k]

plt.bar(deg_k, deg_v)
plt.plot(deg_k, power, 'ro')

plt.show()
