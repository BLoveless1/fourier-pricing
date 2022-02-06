import numpy as np

from pricer.core.constants import *
from pricer.fourier.utility_functions import (
    gibbs_filter,
    inv_hilbert_trans
)


def feng_linetsky_levy(H, solution, down_barrier, up_barrier, bound, num_dates, ht_type=SINC_METHOD, filter_=FILTER_ON,
                       g_filter=EXPONENTIAL_FILTER):
    n = np.size(H)
    xi = np.pi/bound*np.arange(-n/2, n/2)
    dxi = np.exp(np.multiply(1j * down_barrier, xi))
    uxi = np.exp(np.multiply(1j * up_barrier, xi))
    if filter_ == FILTER_OFF:

        if -bound < down_barrier and up_barrier == bound:  # down-and-out
            for i in range(num_dates, 1, -1):   # backward induction in Fourier space
                solution = np.multiply(solution, H)
                solution = \
                    np.multiply(0.5, solution+np.multiply(dxi, inv_hilbert_trans(np.divide(solution, dxi), n, ht_type)))

        elif -bound == down_barrier and up_barrier < bound:    # up-and-out
            for i in range(num_dates, 1, -1):   # backward induction in Fourier space
                solution = np.multiply(solution, H)
                solution = \
                    np.multiply(0.5, solution+np.multiply(uxi, inv_hilbert_trans(np.divide(solution, uxi), n, ht_type)))

        else:   # double barrier
            for i in range(num_dates, 1, -1):  # backward induction in Fourier space
                solution = np.multiply(solution, H)
                sol1 = np.multiply(dxi, inv_hilbert_trans(np.divide(solution, dxi), n, ht_type))
                sol2 = np.multiply(uxi, inv_hilbert_trans(np.divide(solution, uxi), n, ht_type))
                solution = np.multiply(0.5, sol1 - sol2)

        solution = np.multiply(solution, H)
        Sol = np.divide(np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(solution)))), 2*bound)
        return Sol

    else:

        gf = gibbs_filter(xi, 12, g_filter)
        if -bound < down_barrier and up_barrier == bound:  # down-and-out
            for i in range(num_dates, 1, -1):  # backward induction in Fourier space
                solution = np.multiply(gf, np.multiply(solution, H))
                solution = \
                    np.multiply(0.5,
                                solution+np.multiply(dxi, inv_hilbert_trans(np.divide(solution, dxi), n, ht_type)))

        elif -bound == down_barrier and up_barrier < bound:  # up-and-out
            for i in range(num_dates, 1, -1):  # backward induction in Fourier space
                solution = np.multiply(gf, np.multiply(solution, H))
                solution = \
                    np.multiply(0.5,
                                solution - np.multiply(uxi, inv_hilbert_trans(np.divide(solution, uxi), n, ht_type)))

        else:   # double barrier
            for i in range(num_dates, 1, -1):  # backward induction in Fourier space
                solution = np.multiply(solution, H)
                sol1 = np.multiply(dxi, inv_hilbert_trans(np.divide(np.multiply(gf, solution), dxi), n, ht_type))
                sol2 = np.multiply(uxi, inv_hilbert_trans(np.divide(np.multiply(gf, solution), uxi), n, ht_type))
                solution = np.multiply(0.5, sol1 - sol2)

        solution = np.multiply(solution, H)
        Sol = np.divide(np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(np.multiply(gf, solution))))), 2*bound)
        return Sol


def feng_linetsky_stochvol(xi, H, Sol, down_barrier, up_barrier, bound, ndates, grid_v, rf, dt, tau, ht_type=SINC_METHOD):
    """
    L. Feng, V. Linetsky, Math. Finance 18, 337-384 (2008)
    H is 3D matrix with shape (n_v+1, n_x, n_v)
    Sol is payoff after F.T. and repmat with size (grid_x,grid_v)
    tau is (n_v+1, n_x, n_v)
    """
    H_alpha = np.ones(grid_v)
    dxi = np.exp(np.multiply(1j*down_barrier, xi))
    uxi = np.exp(np.multiply(1j*up_barrier, xi))

    if -bound < down_barrier and up_barrier == bound:  # down-and-out option
        for i in range(ndates, 1, -1):  # backward induction in Fourier space
            solution_t = np.copy(Sol)
            for j in range(1, grid_v+1):  # put H0 as index 0
                solution = np.matmul(solution_t*(H[j]/H_alpha[j-1]), np.transpose(tau[j, :]))  # (n_x, 1)
                solution = \
                    np.multiply(np.exp(-rf*dt)*0.5*H_alpha[j-1],
                                (np.transpose(solution)+dxi*inv_hilbert_trans(np.transpose(solution)/dxi, 0, ht_type)))
                Sol[:, j-1] = np.transpose(solution)

    elif -bound == down_barrier and up_barrier < bound:  # up-and-out option
        for i in range(ndates, 1, -1):
            solution_t = np.copy(Sol)
            for j in range(1, grid_v+1):
                solution = np.matmul(solution_t*(H[j]/H_alpha[j-1]), np.transpose(tau[j, :]))
                solution = \
                    np.multiply(np.exp(-rf*dt)*0.5*H_alpha[j-1],
                                (np.transpose(solution)-uxi*inv_hilbert_trans(np.transpose(solution)/uxi, 0, ht_type)))
                Sol[:, j-1] = np.transpose(solution)

    else:  # double-barrier option
        for i in range(ndates, 1, -1):
            solution_t = np.copy(Sol)
            for j in range(1, grid_v+1):
                solution = np.matmul(solution_t*(H[j]/H_alpha[j-1]), np.transpose(tau[j, :]))
                solution1 = np.matmul(dxi, inv_hilbert_trans(np.transpose(solution)/dxi, 0, ht_type))
                solution2 = np.matmul(uxi, inv_hilbert_trans(np.transpose(solution)/uxi, 0, ht_type))
                solution = np.multiply(np.multiply(0.5*np.exp(-rf*dt), H_alpha[j-1]), (solution1-solution2))
                Sol[:, j-1] = np.transpose(solution)

    # Eqs. (5.14) and (6.38): final step, from Fourier space to normal space
    Sol = np.matmul(Sol*H[0], tau[0])*np.exp(-rf*dt)
    sol = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Sol, 0), axis=0), 0))/(2*bound)
    return sol
