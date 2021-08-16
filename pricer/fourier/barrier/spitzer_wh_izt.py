import numpy as np
import copy as copy_

from pricer.constants import (
    FILTER_OFF,
    FILTER_ON,
    SINC_METHOD,
    EXPONENTIAL_FILTER,
)
from pricer.fourier.utility_functions import (
    binomial_1d as binomial,
    gibbs_filter,
    inv_hilbert_trans,
    acc_epsilon,
    acc_shanks
)


def spitzer_wh_levy(q, K, G, down_barrier, up_barrier, barrier_level, ht_type=SINC_METHOD, filter_=FILTER_OFF,
                    g_filter=EXPONENTIAL_FILTER):
    m = len(q)
    n = len(K)
    xi = np.pi/barrier_level * np.arange(-n/2, n/2)

    gf = 0
    gf_mat = []
    if filter_ == FILTER_ON:
        gf = gibbs_filter(xi, 12, g_filter)
        K = np.multiply(K, gf)
        gf_mat = np.tile(gf, (m, 1))

    Karr = np.tile(K, (m, 1))
    Larr = np.transpose(np.tile(q, (n, 1)))
    L = 1 - np.multiply(Larr, Karr)

    # Factorise L = (1-qK) with respect to zero
    iHt = inv_hilbert_trans(np.log(L), n, ht_type)
    Lm = np.exp(np.divide(np.log(L) - iHt, 2))
    Lp = np.exp(np.divide(np.log(L) + iHt, 2))

    if -barrier_level < down_barrier and up_barrier == barrier_level:
        # Decompose P with respect to the down barrier
        dxi = np.tile(np.exp(1j*down_barrier*xi), (m, 1))
        if filter_ == FILTER_ON:
            P = np.divide(np.multiply(gf_mat, Karr), np.multiply(Lm, dxi))
        else:
            P = np.divide(Karr, np.multiply(Lm, dxi))
        Pp = (P+inv_hilbert_trans(P, n, ht_type))/2
        F = np.divide(np.multiply(Karr, np.multiply(np.tile(np.conj(G), (m, 1)), np.multiply(dxi, Pp))), Lp)
        return F

    elif -barrier_level == down_barrier and up_barrier < barrier_level:
        # Decompose Q with respect to the up barrier
        uxi = np.tile(np.exp(1j*up_barrier*xi), (m, 1))
        if filter_ == FILTER_ON:
            P = np.divide(np.multiply(gf_mat, Karr), np.multiply(Lm, uxi))
        else:
            P = np.divide(Karr, np.multiply(Lm, uxi))
        Pp = (P + inv_hilbert_trans(P, n, ht_type)) / 2
        F = np.divide(np.multiply(Karr, np.multiply(np.tile(np.conj(G), (m, 1)), np.multiply(uxi, Pp))), Lp)
        return F

    else:
        max_iterations = 6
        tol = 1e-15
        F = np.zeros((m, n), dtype=np.complex_)

        for i in range(0, m):
            iteration_number = 0
            x = Lm[i, :]
            Ldm = np.multiply(Lm[i, :], np.exp(1j*down_barrier*xi))
            Lup = np.multiply(Lp[i, :], np.exp(1j*up_barrier*xi))
            Jup = np.zeros(n, dtype=np.complex_)
            while True:
                if filter_ == FILTER_ON:
                    P = np.multiply(gf, (K-Jup))
                else:
                    P = K-Jup
                Jdm = (P-np.multiply(Ldm, inv_hilbert_trans(np.divide(P, Ldm), n, ht_type)))/2

                if filter_ == FILTER_ON:
                    Q = np.multiply(gf, (K-Jdm))
                else:
                    Q = K-Jdm
                Jup = (Q+np.multiply(Lup, inv_hilbert_trans(np.divide(Q, Lup), n, ht_type)))/2

                F_old = copy_.copy(F[i, :])
                # F0(i,:) = K.*(K./L(i,:)-eidxi.*Pm./Lp(i,:)-eiuxi.*Qp./Lm(i,:));
                F[i, :] = np.divide(np.multiply(K, np.multiply(np.conj(G), (K-Jdm-Jup))), L[i, :])
                err = np.linalg.norm(F[i, :]-F_old, np.inf)
                iteration_number += 1
                if err < tol or iteration_number == max_iterations:
                    break

    return F


def spitzer_wh_stochvol(P, G, ndates, ngrid_v, ngrid_x, r, dt, tau, dx, euler="OFF"):
    """
    This function is to find price of discrete mornitored barrier option
    using Z-transform and WH technique.
    We use ndates-1 for the z-transform as Z-transform is the process to make
    time collups but at initial date, we don't need to apply barrier
    condition, so there are ndate-1 time steps need to take Z-transform.
    P: characteristic function, n_v cell with (n_x,n_v) matrixs.
    ndates: number of mornitoring dates.
    tau: every ith row is weight_j*p_ij.
    P0: characteristic function with v0 is initial v, fixed and only one, so
    is a (n_x,n_v) matrixs.
    tau0: weight_j*p_v0j, vector.
    gg: alpha shifted payoff for all grid points of x. use zero to replace,see
    who quick.
    G: Fourier transformed payoff
    """
    # P0=P[0], tau0=tau[0]
    tau_d = tau[1:]-np.diag(np.diag(tau[1:]))

    izt_gamma = 6
    nE = 12
    mE = 20
    rho = np.power(10, -izt_gamma/(ndates-1))
    if euler == "OFF":
        q = rho * np.concatenate(([1], [np.exp((-1j*np.arange(1, ndates-1)*np.pi)/(ndates-1))], [- 1]), axis=None)
        c = np.multiply(np.power(-1, np.arange(0, ndates)),
                        np.concatenate(([0.5], [np.ones(ndates-1-1)], [0.5]), axis=None))
    else:
        q = rho * np.concatenate(([1], [np.exp((-1j*np.arange(1, nE+mE+1)*np.pi)/ndates)]), axis=None)
        # is a column vector with size(nE+mE+1,1)
        c = np.multiply(np.power(-1, np.arange(0, nE+mE+1)), np.concatenate(([0.5], [np.ones(nE+mE)]), axis=None))
    c1 = c[:, np.newaxis][:, :, np.newaxis]

    FC = np.zeros((len(q), ngrid_x, ngrid_v), dtype=np.complex_)
    tol = 1e-7  # enough, 1e-15 produce same ans

    L = np.zeros((ngrid_x, len(q), ngrid_v), dtype=np.complex_)
    for i in range(0, ngrid_v):
        Li = 1-q[np.newaxis, :]*P[i+1][:, i][:, np.newaxis]*np.exp(-r*dt)*tau[i+1, i]  # (n_x,len(q))
        L[:, :, i] = np.copy(Li)

    G = np.transpose(G)
    tau_d = np.transpose(tau_d)
    FC_old = 0

    for j in range(0, len(q)):
        if j == 0:
            FC_old = np.copy(FC[j, :, :])

        # Factorise L, fixed in while loop.
        lL = np.log(L[:, j, :].T)  # (n_v,n_x)
        iHlL = inv_hilbert_trans(lL, 0, SINC_METHOD)  # 0 or len(P)?
        lLm = (lL-iHlL)/2
        lLp = (lL+iHlL)/2
        Lm = np.exp(lLm).T
        Lp = np.exp(lLp).T  # (ngrid_x,ngrid_v)

        # Fixed point algorithm
        iteration = 0
        C_m = np.copy(FC_old)
        m_max = 100
        while iteration != m_max+1:
            for i in range(0, ngrid_v):
                temp = (q[j]*np.exp(-r*dt)*np.dot(P[i+1]*C_m, tau_d[:, i])+G)/Lm[:, i]  # (n_x,1)
                # test=(inv_hilbert_trans(temp.T,0,SINC_METHOD).T+temp)/2
                C_m[:, i] = 0.5*(inv_hilbert_trans(temp.T, 0, SINC_METHOD).T+temp)/Lp[:, i]

            err = np.linalg.norm(C_m-FC_old, np.inf, axis=0)  # (1,n_v)
            if max(err) < tol:
                # fprintf('Iter %d, Error %f\n',iteration,max(err))
                break
            iteration = iteration+1
            FC_old = np.copy(C_m)

        FC[j, :, :] = np.copy(C_m)
        FC_old = np.copy(C_m)

    if euler == "OFF":
        f = sum(np.tile(c1, (1, ngrid_x, ngrid_v))*FC, 0)
    else:
        ps = np.cumsum(np.tile(c1, (1, ngrid_x, ngrid_v))*FC, 1)
        f = sum(np.transpose(binomial(mE))*ps[nE-1:nE+mE, :, :], 0)/2**mE

    f = np.divide(f, ((ndates-1)*rho**(ndates-1)))
    Sol = np.matmul(f*P[0], np.transpose(tau[0]))*np.exp(-r*dt)  # (n_x,1)
    sol = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Sol, 0), axis=0), 0))/(dx*ngrid_x)
    return sol


def izt_aw(H, G, lower_barrier, upper_barrier, bound, ht_type, filter_, g_filter, ndates, euler="ON"):
    izt_gamma = 6
    nE = 12
    mE = 20
    rho = np.power(10, -izt_gamma/ndates)
    if euler == "OFF":
        q = rho * np.concatenate(([1], [np.exp((-1j*np.arange(1, ndates)*np.pi)/ndates)], [- 1]), axis=None)
        c = np.multiply(np.power(-1, np.arange(0, ndates+1)),
                        np.concatenate(([0.5], [np.ones(ndates-1)], [0.5]), axis=None))
        ftilde = spitzer_wh_levy(q, np.conj(H), G, lower_barrier, upper_barrier, bound, ht_type, filter_)
        fn = np.real(np.sum(np.multiply(np.reshape(c, (len(c), 1)), ftilde), axis=0))
    else:
        q = rho * np.concatenate(([1], [np.exp((-1j*np.arange(1, nE+mE+1)*np.pi)/ndates)]), axis=None)
        c = np.multiply(np.power(-1, np.arange(0, nE+mE+1)), np.concatenate(([0.5], [np.ones(nE+mE)]), axis=None))
        ftilde = spitzer_wh_levy(q, np.conj(H), G, lower_barrier, upper_barrier, bound, ht_type, filter_, g_filter)
        ps = np.cumsum(np.multiply(np.reshape(c, (len(c), 1)), ftilde), axis=0)
        y = binomial(mE)
        fn = np.real(np.sum(np.multiply(np.reshape(y, (len(y), 1)), ps[nE-1:nE+mE, :]), axis=0)) / np.power(2, mE)
    f = np.divide(fn, (np.multiply(ndates, np.power(rho, ndates))))
    return f


def izt_c(H, G, lower_barrier, upper_barrier, bound, ht_type, filter_, g_filter, ndates, r):
    N = ndates+2048
    n = np.arange(0, N)
    q = np.divide(1.0, np.multiply(r, np.exp(-1j*2*np.multiply(np.pi, n)/N)))
    ftilde = spitzer_wh_levy(q, np.conj(H), G, lower_barrier, upper_barrier, bound, ht_type, filter_, g_filter)
    temp = np.transpose(ftilde)
    fn = np.real(np.multiply(np.reshape(np.power(r, n), (1, N)), np.fft.ifft(np.conj(temp), N, 1)))
    f = fn[:, ndates]
    return f


def izt_c_sum(H, G, lower_barrier, upper_barrier, bound, ht_type, filter_, g_filter, ndates, r):
    N = ndates*10
    n = np.arange(0, N+1)
    q = np.divide(1.0, np.multiply(r, np.exp(1j*np.multiply(np.pi, n)/N)))
    c = np.multiply(np.exp(np.divide(1j*np.pi*ndates*n, N)), np.concatenate(([0.5], [np.ones(N-1)], [0.5]), axis=None))
    ftilde = spitzer_wh_levy(q, np.conj(H), G, lower_barrier, upper_barrier, bound, ht_type, filter_, g_filter)
    f = np.multiply(np.power(r, ndates), np.sum(np.real(np.multiply(c, np.transpose(ftilde))), axis=0))/N
    return f


def izt_c_acc_s(H, G, lower_barrier, upper_barrier, bound, ht_type, filter_, g_filter, ndates, r):
    n = np.arange(0, ndates+1)
    q = np.conj(np.divide(1.0, np.multiply(r, np.exp(-1j*np.multiply(np.pi, n)/ndates))))
    c = np.exp(np.divide(-1j*np.pi*ndates*n, ndates)) * np.concatenate(([0.5], [np.ones(ndates)]), axis=None)
    ftilde = spitzer_wh_levy(q, np.conj(H), G, lower_barrier, upper_barrier, bound, ht_type, filter_, g_filter)
    ps = np.cumsum(np.multiply(np.reshape(c, (len(c), 1)), ftilde), axis=0)
    f = acc_shanks(ps)
    f = np.divide(np.multiply(np.power(r, ndates), np.real(f[-1, :]).T), ndates)
    return f


def izt_c_acc_e(H, G, lower_barrier, upper_barrier, bound, ht_type, filter_, g_filter, ndates, r):
    n = np.arange(0, ndates+1)
    q = np.conj(np.divide(1.0, np.multiply(r, np.exp(-1j*np.multiply(np.pi, n)/ndates))))
    c = np.exp(np.divide(-1j*np.pi*ndates*n, ndates)) * np.concatenate(([0.5], [np.ones(ndates-1)], [0.5]), axis=None)
    ftilde = spitzer_wh_levy(q, np.conj(H), G, lower_barrier, upper_barrier, bound, ht_type, filter_, g_filter)
    ps = np.cumsum(np.multiply(np.reshape(c, (len(c), 1)), ftilde), axis=0)
    f = acc_epsilon(ps)
    f = np.divide(np.multiply(np.power(r, ndates), np.real(f[-1, :]).T), ndates)
    return f

