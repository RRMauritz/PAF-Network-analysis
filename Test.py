import matplotlib.pyplot as plt
from ProjectPAF.PAF_Generation import paf_graph
import numpy as np

sc_deg = np.array([i for i in range(10)])
m = 5
d = 3

sc_deg = np.append(sc_deg, m / d)
sc_deg[2] = 0

print(sc_deg)