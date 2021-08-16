import numpy as np
from pricer.constants import (
    FILTER_OFF,
    FILTER_ON,
    SINC_METHOD,
    EXPONENTIAL_FILTER,
    CALL,
    PUT
)
from pricer.fourier.utility_functions import (
    binomial_1d as binomial,
    gibbs_filter,
    inv_hilbert_trans, acc_epsilon
)


def spitzer_lb(q, H, hilbert_type, option_type):
    m = len(q)
    n = len(H)
    h = np.multiply(np.transpose(np.tile(q, (n, 1))), np.tile(H, (m, 1)))

    iL = -np.log(1-h)
    iHL = inv_hilbert_trans(iL, n, hilbert_type)
    Lm = np.exp((iL-iHL)/2)
    Lp = np.exp((iL+iHL)/2)

    index = int(n/2)
    if option_type == CALL:
        LM = np.tile(Lm[:, index], (n, 1))
        F_0 = np.multiply(LM.T, Lp)
    else:
        LP = np.tile(Lp[:, index], (n, 1))
        F_0 = np.multiply(LP.T, Lm)

    return F_0


def izt_aw(H, ht_type, ndates, euler="ON", option_type="CALl"):
    izt_gamma = 6
    nE = 12
    mE = 20
    rho = np.power(10, -izt_gamma/ndates)
    if ndates <= nE+mE or euler == "OFF":
        q = rho * np.concatenate(([1], [np.exp((-1j*np.arange(1, ndates)*np.pi)/ndates)], [- 1]), axis=None)
        c = np.multiply(np.power(-1, np.arange(0, ndates+1)),
                        np.concatenate(([0.5], [np.ones(ndates-1)], [0.5]), axis=None))
        ftilde = spitzer_lb(q, H, ht_type, option_type)
        fn = np.sum(np.multiply(np.reshape(c, (len(c), 1)), ftilde), axis=0)
    else:
        q = rho * np.concatenate(([1], [np.exp((-1j*np.arange(1, nE+mE+1)*np.pi)/ndates)]), axis=None)
        c = np.multiply(np.power(-1, np.arange(0, nE+mE+1)), np.concatenate(([0.5], [np.ones(nE+mE)]), axis=None))
        ftilde = spitzer_lb(q, H, ht_type, option_type)
        ps = np.cumsum(np.multiply(np.reshape(c, (len(c), 1)), ftilde), axis=0)
        y = binomial(mE)
        fn = np.sum(np.multiply(np.reshape(y, (len(y), 1)), ps[nE-1:nE+mE, :]), axis=0) / np.power(2, mE)
    f = np.divide(fn, (np.multiply(ndates, np.power(rho, ndates))))
    return f


def izt_cavers_acc(H, ht_type, ndates, option_type="CALl"):
    r = 0.99
    n = np.arange(0, ndates+1)
    q = np.divide(1.0, np.multiply(r, np.exp(1j*np.multiply(np.pi, n)/ndates)))
    c = np.exp(np.divide(1j*np.pi*ndates*n, ndates))*np.concatenate(([0.5], [np.ones(ndates-1)], [0.5]), axis=None)
    ftilde = spitzer_lb(q, H, ht_type, option_type)
    ps = np.cumsum(np.multiply(np.reshape(c, (len(c), 1)), ftilde), axis=0)
    f = acc_epsilon(ps)
    f = np.divide(np.multiply(np.power(r, ndates), np.real(f[-1, :]).T), ndates)
    return f
