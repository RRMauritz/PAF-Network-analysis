import numpy as np


def BP(n, p, T):
    """"
    Generates a branching process with i.i.d. Bin(n,p) offspring distribution
    :param n denotes the maximum number of children for each individual
    :param p denotes the success rate of getting a child
    :param T denotes the number of generations for which the simulation is run
    """
    H = np.zeros(T)
    H[0] = 1
    ext = False
    for t in range(1, T):
        z_prev = H[t - 1]
        H[t] = sum(np.random.binomial(n, p, int(z_prev)))
        # print("t = ", t, "z_prev = ", z_prev, " offspring = ", H[t])
        if H[t] == 0:
            ext = True
            break
    mu = n * p
    return mu, H, ext


def est_mu(n, p, T, N):
    """"
    Estimates the extinction parameter in a BP with bin(n,p) offspring
    It does so by looking for an extinction up to generation T and repeats this process N times
    """
    n_ext = 0
    for k in range(N):
        print(k)
        _, _, ext = BP(n, p, T)
        if ext:
            n_ext += 1
    return (n_ext / N) * 100


n_sub = est_mu(2, 0.40, 60, 3000)
n_super = est_mu(2, 0.55, 60, 3000)
print('Subcritical: mu = ', n_sub, '%')
print('Supercritical: mu = ', n_super, '%')
