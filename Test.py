import matplotlib.pyplot as plt
from ProjectPAF.PAF_Generation import paf_graph
import numpy as np

fitnesses = np.random.binomial(10, 0.3, size=5) + 1

print(fitnesses)