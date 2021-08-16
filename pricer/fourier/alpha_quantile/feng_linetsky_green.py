import numpy as np

from pricer.fourier.utility_functions import inv_hilbert_trans


def feng_linetsky_green(H, hilbert_type, ndates, option_type):
    n = len(H)
    Sol = np.ones(n)

    if option_type == "CALL":
        for i in range(ndates-1, 0, -1):
            Sol = np.multiply(Sol, H)
            Sol = 0.5 * (Sol + inv_hilbert_trans(Sol, n, hilbert_type))
            Sol = Sol+1-Sol[int(len(Sol)/2)]

    else:
        for i in range(ndates-1, 0, -1):
            Sol = np.multiply(Sol, H)
            Sol = 0.5 * (Sol-inv_hilbert_trans(Sol, n, hilbert_type))
            Sol = Sol+1-Sol[int(len(Sol)/2)]

    return Sol
