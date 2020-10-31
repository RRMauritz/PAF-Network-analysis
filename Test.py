from sympy import *
import operator as op
import numpy as np
from functools import reduce
from ProjectPAF.Graph_Generation import paf_graph
from scipy.stats import binom
from math import factorial

def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom  # or / in Python 2


paf_graph(4)
# J = 10
# p = 0.3
#
# rv = binom(10, 0.3)
# Q = rv.pmf([i for i in range(10)])
#
# j, l = symbols('j,l')
# print(Sum((j * p) / (l - j), (j, 0, J)).doit())


