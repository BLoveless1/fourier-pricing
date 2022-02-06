import math
from scipy.special import ive

from pricer.fourier.models import *
from ..core.constants import *

np.seterr(divide='ignore', invalid='ignore')


class UtilitiesError(Exception):
    pass


def binomial_coefficient(n, k):
    result = 1
    for i in range(1, k+1):
        result = result*(n-i+1)//i
    return result


def binomial_1d(n):
    pascal = [[binomial_coefficient(i+j, i) for j in range(n+1)] for i in range(n+1)]
    return np.diag(np.fliplr(pascal))


def characteristic_function(model, u, flag=BACKWARD_PROBLEM):
    mean_correction = (model.r-model.q)*model.dt-np.log(model.get_char_function(-1j))
    F = np.multiply(model.get_char_function(u), np.exp(1j*mean_correction*u))
    if flag == BACKWARD_PROBLEM:
        F = np.conj(F)
    return F


def bound_lower_tail_levy(model, x, max_num_moments, direction_flag):
    min_low = 1
    for i in range(1, max_num_moments):
        bound = np.real(characteristic_function(model, i*1j, direction_flag)*np.exp(-x*i))
        min_low = min(min_low, bound)
    return min_low


def bound_upper_tail_levy(model, x, max_num_moments, direction_flag):
    min_up = 1
    for i in range(1, max_num_moments):
        bound = np.real(characteristic_function(model, -i*1j, direction_flag)*np.exp(-x*i))
        min_up = min(min_up, bound)
    return min_up


def deterministic_bounds(model, direction_flag):
    max_num_moments = 11
    factor_lower_barrier = 5
    factor_upper_barrier = 0
    tol = 1e-08

    cumulant_1, cumulant_2 = model.get_cumulant()
    if math.isnan(cumulant_1) or math.isnan(cumulant_2):
        raise UtilitiesError("Incorrect model parameters giving NaN cumulant: cumulant_1, cumulant_2 = " +
                             cumulant_1+", "+cumulant_2)

    lower_bound = cumulant_1-factor_lower_barrier*np.sqrt(cumulant_2)
    bound = bound_lower_tail_levy(model, -lower_bound, max_num_moments, direction_flag)
    while bound > tol:
        factor_lower_barrier = factor_lower_barrier+1
        lower_bound = cumulant_1-factor_lower_barrier*np.sqrt(cumulant_2)
        bound = bound_lower_tail_levy(model, -lower_bound, max_num_moments, direction_flag)

    upper_bound = cumulant_1+factor_upper_barrier*np.sqrt(cumulant_2)
    bound = bound_upper_tail_levy(model, upper_bound, max_num_moments, direction_flag)
    while bound > tol:
        factor_upper_barrier = factor_upper_barrier+1
        upper_bound = cumulant_1+factor_upper_barrier*np.sqrt(cumulant_2)
        bound = bound_upper_tail_levy(model, upper_bound, max_num_moments, direction_flag)

    return lower_bound, upper_bound


def stochastic_vol_bounds(model, T, set_gamma=7, v_bar=0.0001):
    if model.a is np.NaN and model.b is np.NaN:
        raise UtilitiesError("Model must be stochastic to use stochastic volatility bounds")
    kappa, theta = model.a, model.b
    t = T/2
    v_mu = np.exp(-kappa*t)*model.v0+theta*(1-np.exp(-kappa*t))
    v_var = (model.sigma**2/kappa)*model.v0 *\
            (np.exp(-kappa*t) - np.exp(-2*kappa*t))+(theta*model.sigma**2/kappa/2)*pow(1-np.exp(-kappa*t), 2)
    lower_bound = max(v_bar, v_mu-set_gamma*np.sqrt(v_var))
    upper_bound = v_mu+set_gamma*np.sqrt(v_var)
    return lower_bound, upper_bound


def set_alpha(model, lower_barrier, upper_barrier, barrier_level, option_type=CALL):
    if -barrier_level < lower_barrier and upper_barrier == barrier_level:  # down-and-out option
        if option_type in [CALL]:
            payoff_type = VANILLA
        elif option_type in [PUT]:
            payoff_type = TRUNCATED
        else:
            raise UtilitiesError("Incorrect option type:"+option_type)
    elif -barrier_level == lower_barrier and upper_barrier < barrier_level:  # up-and-out option
        if option_type in [CALL]:
            payoff_type = TRUNCATED
        elif option_type in [PUT]:
            payoff_type = VANILLA
        else:
            raise UtilitiesError("Incorrect option type in set_alpha :"+option_type)
    else:  # double barrier option
        payoff_type = TRUNCATED

    lm = model.lambda_m
    lp = model.lambda_p
    if lm == 0 or lp == 0:
        if payoff_type == VANILLA:
            alpha = -10
        else:
            alpha = 0
    else:
        if payoff_type == VANILLA:
            alpha = (lm-1)/2
        else:
            alpha = (lp+lm)/2

    return alpha


def kernel(model, ngrid, xmin, xmax, truncation_alpha=0, disc="ON", flag=BACKWARD_PROBLEM):
    N = ngrid/2
    dx = (xmax-xmin)/ngrid
    x = dx*np.arange(-N, N)
    dw = 2*np.pi/(xmax-xmin)
    w = dw*np.arange(-N, N)
    H = characteristic_function(model, w+1j*truncation_alpha, flag)
    if disc == "ON":  # discount factor in the density
        H = H*np.exp(-model.r*model.dt)
    h = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(H))))/(xmax-xmin)
    return x, h, w, H


def stochastic_grid(ngrid_x, xmax, xmin):
    N_x = ngrid_x/2
    dx = (xmax-xmin)/ngrid_x
    x = dx*np.arange(-N_x, N_x)
    dxi_x = 2*np.pi/(dx*ngrid_x)
    xi_x = dxi_x*np.arange(-N_x, N_x)
    return x, xi_x, dx


def transition_prob_matrix(model, x, y):
    """
    Find transition probability matrix of CIR model
    dv=a(b-v)dt+sigma*np.sqrt(v)dW_t
    x is v(t) is a scalar. y is v(t+dt) is a vector.
    """
    a, b, sigma, dt = model.a, model.b, model.sigma, model.dt
    c = 2*a/((1-np.exp(-a*dt))*sigma**2)
    q = 2*a*b/(sigma**2)-1
    u = c*x*np.exp(-a*dt)
    v = c*y
    P1 = np.log(c)-u-v+np.log(np.power((v/u), (q/2)))+np.log(ive(q, 2*np.sqrt(u*v)))+2*np.sqrt(u*v)
    P = np.exp(P1)
    return P


def payoff(x, xi, truncation_alpha, K, L, U, C, option_type=CALL):
    S = C*np.exp(x)
    if option_type in [CALL]:
        g = np.multiply(
            np.multiply(np.multiply(np.exp(truncation_alpha*x), np.where((S-K) > 0, S-K, 0)), np.less_equal(S, U)*1),
            np.greater_equal(S, L)*1)
    elif option_type in [PUT]:
        g = np.multiply(
            np.multiply(np.multiply(np.exp(truncation_alpha*x), np.where((K-S) > 0, K-S, 0)), np.less_equal(S, U)*1),
            np.greater_equal(S, L)*1)
    else:
        raise UtilitiesError("Incorrect option type in payoff: "+option_type)

    b1 = b2 = 0
    if option_type in [CALL]:
        b1 = np.log(max(L, K)/C)
        b2 = np.log(U/C)
    elif option_type in [PUT]:
        b1 = np.log(min(U, K)/C)
        b2 = np.log(L/C)

    xi_adjust = truncation_alpha+1j*xi
    K_adjust = np.log(K/C)
    G = np.multiply(C, ((np.exp(np.multiply(b2, (1+xi_adjust)))-np.exp(np.multiply(b1, (1+xi_adjust))))/(1+xi_adjust)-(
            np.exp(K_adjust+np.multiply(b2, xi_adjust))-np.exp(K_adjust+np.multiply(b1, xi_adjust)))/xi_adjust))

    if truncation_alpha == 0:
        G[math.floor(np.size(G)/2)] = C*(np.exp(b2)-np.exp(b1)-np.exp(K_adjust)*(b2-b1))

    if truncation_alpha == -1:
        G[math.floor(np.size(G)/2)] = C*b2-b1+np.exp(b2)-np.exp(K_adjust-b2)-np.exp(K_adjust-b1)

    return S, g, G


def payoff_wh_stochvol(x, xi, truncation_alpha, K, L, U, C, option_type=CALL):
    S = C*np.exp(x)
    l = np.log(L/C)
    u = np.log(U/C)
    k = np.log(K/C)
    if option_type in [CALL]:
        g = np.multiply(
            np.multiply(np.multiply(np.exp(truncation_alpha*x), np.where((C*np.exp(x+l)-K) > 0, C*np.exp(x+l)-K, 0)),
                        np.less_equal(x, u-l)*1), np.greater_equal(x, 0)*1)
    elif option_type in [PUT]:
        g = np.multiply(
            np.multiply(np.multiply(np.exp(truncation_alpha*x), np.where((K-C*np.exp(x+l)) > 0, K-C*np.exp(x+l), 0)),
                        np.less_equal(x, u-l)*1), np.greater_equal(x, 0)*1)
    else:
        raise UtilitiesError("Incorrect option type in payoff: "+option_type)

    b1 = b2 = 0
    if option_type in [CALL]:
        b1 = max(0, k-l)
        b2 = u-l
    elif option_type in [PUT]:
        b1 = min(u-l, k-l)
        b2 = 0

    xi_adjust = truncation_alpha+1j*xi
    K_adjust = np.log(K/C)
    G = np.multiply(C,
                    ((np.exp(l+np.multiply(b2, (1+xi_adjust)))-np.exp(l+np.multiply(b1, (1+xi_adjust))))/(1+xi_adjust)-(
                            np.exp(K_adjust+np.multiply(b2, xi_adjust)) -
                            np.exp(K_adjust+np.multiply(b1, xi_adjust)))/xi_adjust))

    if truncation_alpha == 0:
        G[math.floor(np.size(G)/2)+1] = np.multiply(C, np.exp(b2)-np.exp(b1)-np.multiply(np.exp(K_adjust), (b2-b1)))

    if truncation_alpha == -1:
        G[math.floor(np.size(G)/2)+1] = np.multiply(C, b2-b1+np.exp(b2)-np.exp(K_adjust-b2)-np.exp(K_adjust-b1))

    return S, g, G


def gibbs_filter(xi, parameter, g_filter=EXPONENTIAL_FILTER):
    xi_max = np.amax(xi)
    xi_norm = np.divide(xi, xi_max)

    if g_filter == EXPONENTIAL_FILTER:  # param used to set filter order (must be even) 2-10 typical
        machine_err = pow(10.0, -16)
        alpha = -np.log(machine_err)
        sigma_f = np.exp(-alpha*pow(xi_norm, parameter))

    elif g_filter == PLANCK_TAPER_WINDOW:  # param used to set shape parameter E, 0<E<0.5, larger = narrower
        N = np.size(xi)
        n_p = np.arange(1, math.floor(np.multiply(parameter, (N-1)))-(math.floor(np.multiply(parameter, (N-1)))
                                                                      == np.multiply(parameter, (N-1))))
        n_m = np.arange(math.ceil(np.multiply((1-parameter), (N-1)))+(math.ceil(np.multiply(parameter, (N-1)))
                                                                      == np.multiply(parameter, (N-1))), N-2)
        zp = np.multiply(np.multiply(parameter, (N-1)),
                         np.divide(1.0, np)+np.divide(1.0, (np-np.multiply(parameter, (N-1)))))
        zm = np.multiply(np.multiply(parameter, (N-1)),
                         np.divide(1.0, N-1-n_m)+np.divide(1.0, (np.multiply((1-parameter), (N-1))-n_m)))
        sigma_f = np.concat(
            [0, 1.0/(np.exp(zp)+1), np.ones(N-np.size(n_p)-np.size(n_m)-2), np.divide(1.0, np.exp(zm)+1), 0])

    else:
        sigma_f = np.ones(np.size(xi))

    return sigma_f


def inv_hilbert_trans(F, P, ht_type=SINC_METHOD):
    if np.ndim(F) == 1:
        M = 1
        N = np.size(F)
    else:
        M = np.size(F, 0)
        N = np.size(F, 1)
    Q = N+P
    vec = []
    old_type = ""
    if ht_type == "":
        ht_type = SINC_METHOD
    if ht_type != old_type:

        if ht_type == RIGHT_CONT_SIGN:
            vec = np.concat((-np.ones((M, math.floor(Q/2)), dtype=int), np.ones((M, math.floor(Q/2)), dtype=int)),
                            axis=None)
        elif ht_type == LEFT_CONT_SIGN:
            vec = np.concat((np.ones(M, 1), -np.ones((M, math.floor(Q/2)), dtype=int),
                             np.ones((M, math.floor(Q/2)-1), dtype=int)), axis=None)
        elif ht_type == SYMMETRIC_SIGN:
            vec = np.concat((np.zeros(M, 1), -np.ones((M, math.floor(Q/2)-1), dtype=int), np.zeros(M, 1),
                             np.ones((M, math.floor(Q/2)-1), dtype=int)), axis=None)
        elif ht_type == SINC_METHOD:
            t = np.nan_to_num(np.divide(1-pow(-1, np.arange(-Q/2, Q/2)), np.multiply(np.pi, np.arange(-Q/2, Q/2))))
            vec = np.tile(np.imag(np.fft.fft(np.fft.ifftshift(t))), (M, 1))
        else:
            raise UtilitiesError("Unknown Hilbert calculation type: "+str(ht_type))

    # Compute the Hilbert transform times the np.imaginary unit
    if M == 1:
        iHf = np.fft.fft(np.multiply(vec, np.fft.ifft(F, Q)))
        return iHf[0][0:N]
    else:
        iHf = np.fft.fft(np.multiply(vec, np.fft.ifft(F, Q, 1)), None, 1)
        return iHf[:, 0:N]


def quadrature_nodes(N, a, b):
    """
    This script is for computing definite integrals using Legendre-Gauss
    Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
    [a,b] with truncation order N.
    The outputs are nodes and corresponding weights
    """
    N = N-1
    N1 = N+1
    N2 = N+2
    xu = np.transpose(np.linspace(-1, 1, N1))
    y = np.cos((2*np.transpose(np.arange(0, N+1))+1)*np.pi/(2*N+2))+(0.27/N1)*np.sin(np.pi*xu*N/N2)
    L = np.zeros((N1, N2))
    Lp = np.zeros((N1, N2))
    y0 = 2
    Lp[:, 1] = 1

    while max(np.absolute(y-y0)) > np.finfo(float).eps:
        L[:, 0] = 1
        L[:, 1] = y
        for k in range(2, N1+1):
            L[:, k] = ((2*k-1)*y*L[:, k-1]-(k-1)*L[:, k-2])/k

        x = L[:, N1-1]-y*L[:, N2-1]
        Lp = N2*(L[:, N1-1]-y*L[:, N2-1])/(1-np.power(y, 2))
        y0 = y
        y = y0-L[:, N2-1]/Lp

    x = (a*(1-y)+b*(1+y))/2
    x = x[::-1]
    w = (b-a)/((1-np.power(y, 2))*np.power(Lp, 2))*np.power((N2/N1), 2)
    w = w[::-1]
    return x, w


def get_log_heston_char_function(model, xi, v, prev_v, n_x):
    """
       This function is used to calculate the characteristic function of np.log(S) of Heston model at time t, given v
       and prev_v.
       Input variables:
       v: (n_x,n_v)
       xi: (n_x,n_v), ngrid_v copys of column vector of xi_x .
       tau: T-t, time to maturity. Here default tau=dt.
       dS_t=r*Stdt+np.sqrt(v)S_tdW_1,t;
       dv=kappa(theta-v)dt+sigma*np.sqrt(v)dW_2,t.
       Output variable:
       phi: output of characteristic function at time t.
       """
    # Need to make sure the input of xi is matrix and v is vector!

    kappa, theta, tau = model.a, model.b, model.dt
    temp_A = model.r*tau+model.rho/model.sigma*(v-prev_v-kappa*theta*tau)  # (1, n_v)
    A = np.exp(1j*xi*np.tile(temp_A, (n_x, 1)))  # (n_x,n_v)

    a = xi*(kappa*model.rho/model.sigma-0.5)+0.5*1j*pow(xi, 2)*(1-model.rho**2)  # (n_x,n_v)
    g = np.sqrt(pow(kappa, 2)-2*pow(model.sigma, 2)*1j*a)  # (n_x,n_v) same for all n_v
    b = 1-np.exp(-g*tau)  # (n_x,n_v) same for all n_vm
    c = 1-np.exp(-kappa*tau)
    d = np.tile(np.sqrt(prev_v*v), (n_x, 1))  # (n_x,n_v) same for all n_x, n_x copies of np.sqrt(v0*v)

    B = g*np.exp(-0.5*(g-kappa)*tau)*c/(kappa*b)  # (n_x,n_v) same for all n_v
    temp = kappa*(1+np.exp(-kappa*tau))/c
    temp2 = g*(1+np.exp(-g*tau))/b  # (n_x,n_v) same for all n_v

    C = np.exp(np.tile((v+prev_v), (n_x, 1))/pow(model.sigma, 2)*(temp-temp2))

    nu = 2*theta*kappa/(pow(model.sigma, 2))-1
    z1 = 4*g*np.exp(-0.5*g*tau)/(pow(model.sigma, 2)*b)  # (n_x,n_v) same for all n_v
    z2 = 4*kappa*np.exp(-0.5*kappa*tau)/(pow(model.sigma, 2)*c)
    temp1 = np.log(ive(nu, d*z1))+abs(np.real(d*z1))
    temp2 = np.log(ive(nu, d*z2))+abs(np.real(d*z2))
    D = np.exp(temp1-temp2)

    phi = A*B*C*D

    return phi


def calculation_map_barriers(requests):
    requests_map = {0: {  # for error reference in plots
        "method": "FengLinestky",
        "distribution": requests[0]["distribution"],
        "filter": "FILTER_ON",
        "g_filter": "EXPONENTIAL_FILTER",
        "Hilbert": "right_hand_side_cont_func",
        "density_disc_factor": "ON",
        "Hilbert_type": "SINC_METHOD",
        "direction": "BACKWARD_PROBLEM"
    }}
    length = len(requests)
    for i in range(1, length+1):
        requests_map[i] = requests[i-1]
    if len(requests_map) > 12:
        raise UtilitiesError("Too many requests for plotting options:"+str(len(requests_map))+" maximum is 12")
    return requests_map


def calculation_map_aq(requests):
    requests_map = {}
    length = len(requests)
    for i in range(0, length):
        requests_map[i] = requests[i]
    if len(requests_map) > 12:
        raise UtilitiesError("Too many requests for plotting options:"+str(len(requests_map))+" maximum is 12")
    return requests_map


def get_model(distribution, dt, rf, q):
    if distribution == "NormalDist":
        model = NormalDist(dt, rf, q)
        cavers = 1.1
        cavers_sum = 1.01
        cavers_acc = 1.2
    elif distribution == "NormalInverseGaussian":
        model = NormalInverseGaussian(dt, rf, q)
        cavers = 1.1
        cavers_sum = 1.04
        cavers_acc = 1.2
    elif distribution == "VarianceGamma":
        model = VarianceGamma(dt, rf, q)
        cavers = 1.1
        cavers_sum = 1.01
        cavers_acc = 1.2
    elif distribution == "KouDE":
        model = KouDE(dt, rf, q)
        cavers = 1.1
        cavers_sum = 1.01
        cavers_acc = 1.2
    elif distribution == "MertonJD":
        model = MertonJD(dt, rf, q)
        cavers = 1.15
        cavers_sum = 1.14
        cavers_acc = 1.2
    elif distribution == "Heston":
        model = Heston(dt, rf, q)
        cavers = 0
        cavers_sum = 0
        cavers_acc = 0
    else:
        raise UtilitiesError("Unavailable distribution provided in settings file: "+distribution)

    cavers_r = {
        'cavers': cavers,
        'cavers_sum': cavers_sum,
        'cavers_acc': cavers_acc
    }
    return model, cavers_r


def acc_epsilon(ps):
    ls = len(ps)
    f_epsilon = ps
    k = ls - 2
    for n in range(0, ls+3-k):
        f_epsilon[k, n] = f_epsilon[k-2, n+1]+np.divide(1.0, f_epsilon[k-1, n+1]-f_epsilon[k-1, n])

    return f_epsilon


def acc_shanks(ps):
    ps_s = ps
    for i in range(1, 10):
        ps_3 = ps_s[3:-1, :]
        ps_2 = ps_s[2:-1-1, :]
        ps_1 = ps_s[1:-1-2, :]

        numerator = np.power(ps_3-ps_2, 2)
        denominator = (ps_3-ps_2) - (ps_2-ps_1)
        denominator[denominator == 0] = 1

        ps_s = ps_3-np.divide(numerator, denominator)

    return ps_s