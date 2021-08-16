from scipy.interpolate import (
    CubicSpline as spline,
    interp1d
)

from pricer.fourier.utility_functions import *
from pricer.fourier.barrier.spitzer_wh_izt import *
from pricer.fourier.barrier.feng_linetsky import *
import time


class FourierHandlerError(Exception):
    pass


def price_fourier_barrier_trades(data, fourier_requests_file):
    T = data["expiry_T"]
    ndates = data["num_dates"]
    dt = T/ndates
    rf = data["risk_free_rate"]
    q = data["dividend_rate"]
    S = data["spot_price"]
    K = data["strike_price"]
    option_type = data["option_type"]
    lower_barrier = data["lower_barrier"]
    upper_barrier = data["upper_barrier"]
    scale_factor = S
    direction = data["direction"]

    request_map = calculation_map_barriers(fourier_requests_file)
    max_loop_num = data['iteration_number']
    results = {}
    grid = 1
    for request in request_map:
        results[request], grid = price_handler(request_map[request], max_loop_num, K, q, dt, rf, T, upper_barrier,
                                               lower_barrier, ndates, scale_factor, S, option_type, direction)

    return results, grid


def price_handler(request, max_loop_num, K, q, dt, rf, T, upper_barrier, lower_barrier, ndates, scale_factor, spot,
                  option_type, direction=BACKWARD_PROBLEM):
    model, cavers_r = get_model(request["distribution"], dt, rf, q)
    lower_bound, upper_bound = deterministic_bounds(model, direction)
    upper_bound = max(abs(lower_bound), upper_bound)
    truncation = spot*np.exp(upper_bound)
    bound = np.log(truncation/scale_factor)  # upper bound of support
    truncation_alpha = set_alpha(model, np.log(lower_barrier/scale_factor), np.log(upper_barrier/scale_factor), bound,
                                 option_type)
    grid_v, up_vol, low_vol = 0, 0, 0
    if request["distribution"] == 'Heston':
        low_vol, up_vol = stochastic_vol_bounds(model, T)
        grid_v = 40

    stochastic_pricing_variables = np.array([grid_v, up_vol, low_vol])
    if all(stochastic_pricing_variables == 0):
        results, grid = price_array_levy(request, model, max_loop_num, truncation_alpha, bound, K, upper_barrier,
                                         lower_barrier, ndates, scale_factor, spot, option_type, cavers_r)
    else:
        results, grid = price_array_stochastic_vol(request, model, max_loop_num, 0.5, bound, K, upper_barrier,
                                                   lower_barrier, ndates, spot, option_type,
                                                   stochastic_pricing_variables)
    return results, grid


def price_array_levy(request, model, max_loop_num, truncation_alpha, bound, K, upper_barrier, lower_barrier,
                     ndates, scale_factor, spot, option_type, cavers_r):
    result_map = {}
    grid_array = np.ones(max_loop_num-1)
    start = 1
    if "data_generation" in request and request["data_generation"] == "ON":
        start = max_loop_num-1
    for i in range(start, max_loop_num):
        grid = round(pow(2, i+4))
        grid_array[i-1] = grid
        # Compute the grid and the discounted kernel for Fourier-based methods
        x, h, xi, H = kernel(model, grid, -bound, bound, truncation_alpha, request["density_disc_factor"],
                             request["direction"])

        # Compute the scale, payoff and its FT for Fourier-based methods
        s, g, G = payoff(x, xi, truncation_alpha, K, lower_barrier, upper_barrier, scale_factor, option_type)

        request_string = request['method'].split("_")
        if request["filter"] == "FILTER_ON":
            key_string = request['method']+"_FilterOn"+"_"+request["distribution"]
        else:
            key_string = request['method']+"_FilterOff"+"_"+request["distribution"]

        if request_string[0] == 'FengLinestky':
            t = time.time()
            sol = feng_linetsky_levy(H, G, np.log(lower_barrier/scale_factor), np.log(upper_barrier/scale_factor),
                                     bound, ndates, request["Hilbert_type"], request["filter"])
            sol = np.multiply(sol, np.exp(-truncation_alpha*x))
            try:
                price = spline(s, sol)
            except ValueError:
                price = 0.0
            price = np.longdouble(price(spot))
            cpu_time = time.time()-t
            result_map[key_string, grid] = [price, cpu_time]

        elif request_string[0] == 'WeinerHopfSpitzer':

            t = time.time()
            if request_string[1] == 'AbateWhitt':
                Sol = izt_aw(H, G, np.log(lower_barrier/scale_factor), np.log(upper_barrier/scale_factor), bound,
                             request["Hilbert_type"], request["filter"], request["g_filter"], ndates-2,
                             request["Euler_acceleration"])

            elif request_string[1] == 'Cavers':
                Sol = izt_c(H, G, np.log(lower_barrier/scale_factor), np.log(upper_barrier/scale_factor), bound,
                            request["Hilbert_type"], request["filter"], request["g_filter"], ndates-2,
                            cavers_r["cavers"])

            elif request_string[1] == 'CaversSum':
                Sol = izt_c_sum(H, G, np.log(lower_barrier/scale_factor), np.log(upper_barrier/scale_factor), bound,
                                request["Hilbert_type"], request["filter"], request["g_filter"], ndates-2,
                                cavers_r["cavers_sum"])

            elif request_string[1] == "CaversAcceleratedShanks":
                Sol = izt_c_acc_s(H, G, np.log(lower_barrier/scale_factor), np.log(upper_barrier/scale_factor), bound,
                                  request["Hilbert_type"], request["filter"], request["g_filter"], ndates-2,
                                  cavers_r["cavers_acc"])

            elif request_string[1] == "CaversAcceleratedEpsilon":
                Sol = izt_c_acc_e(H, G, np.log(lower_barrier/scale_factor), np.log(upper_barrier/scale_factor), bound,
                                  request["Hilbert_type"], request["filter"], request["g_filter"], ndates-2,
                                  cavers_r["cavers_acc"])

            else:
                raise FourierHandlerError("Invalid inverse Z-transform in request: "+request_string[1])

            sol = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Sol))))/(2*bound)
            index = int(len(sol)/2)
            price = sol[index]
            cpu_time = time.time()-t
            result_map[key_string, grid] = [price, cpu_time]

        else:
            raise FourierHandlerError("Invalid pricing method requested: "+request_string[0])

    if "data_generation" in request and request["data_generation"] == "ON":
        return list(result_map.values())[0]
    else:
        return result_map, grid_array


def price_array_stochastic_vol(request, model, max_loop_num, truncation_alpha, bound, K, upper_barrier,
                               lower_barrier, ndates, spot, option_type, stochastic_variables):
    result_map = {}
    grid_array = np.ones(max_loop_num-1)
    start = 1
    if "data_generation" in request and request["data_generation"] == "ON":
        start = max_loop_num-1
    for i in range(start, max_loop_num):
        grid_x = round(pow(2, i+4))
        grid_array[i-1] = grid_x
        [grid_v, up_vol, low_vol] = stochastic_variables
        grid_v = int(grid_v)
        x, xi_x, dx = stochastic_grid(grid_x, bound, -bound)
        v, tau_w = quadrature_nodes(grid_v, low_vol, up_vol)

        # transition matrix of variance
        tau_p = np.zeros((grid_v, grid_v))

        for vol in range(grid_v):
            tau_p[vol, :] = transition_prob_matrix(model, v[vol], v)

        tau_p0 = transition_prob_matrix(model, model.v0, v)
        tau = tau_w*tau_p
        tau_0 = tau_w*tau_p0

        # [[tau_0],[tau]] size:(n_v+1,n_v)
        tau = np.append(tau_0[np.newaxis, :], tau, axis=0)

        # Characteristic function
        xi_X = np.transpose(np.tile(xi_x, (grid_v, 1)))
        xi_X_alpha = -xi_X+1j*truncation_alpha*np.ones((grid_x, grid_v))

        P = []  # Here P is a list with size: n_v
        P0 = get_log_heston_char_function(model, xi_X_alpha, v, model.v0, grid_x)
        P.append(P0)
        for vol in range(grid_v):
            P.append(np.get_np.log_heston_char_function(model, xi_X_alpha, v, v[vol], grid_x))

        request_string = request['method'].split("_")
        if request["filter"] == "FILTER_ON":
            key_string = request['method']+"_FilterOn"+"_"+request["distribution"]
        else:
            key_string = request['method']+"_FilterOff"+"_"+request["distribution"]

        if request_string[0] == 'FengLinestky':
            s, g, G = payoff(x, xi_x, truncation_alpha, K, lower_barrier, upper_barrier, spot, option_type)
            t = time.time()
            Sol = np.transpose(np.tile(G, (grid_v, 1)))
            # why is there two bounds to this function
            sol = feng_linetsky_stochvol(xi_x, P, Sol, np.log(lower_barrier/spot), bound, bound, ndates, grid_v,
                                         model.r, model.dt, tau, ht_type=SINC_METHOD)
            sol = np.multiply(sol, np.exp(-truncation_alpha*x))
            price = sol[int(grid_x/2)]
            cpu_time = time.time()-t
            result_map[key_string, grid_x] = [price, cpu_time]

        elif request_string[0] == 'WeinerHopfSpitzer':
            upper_barrier = spot*np.exp(bound)
            s, g, G = payoff_wh_stochvol(x, xi_x, truncation_alpha, K, lower_barrier, upper_barrier, spot, option_type)
            t = time.time()
            sol = spitzer_wh_stochvol(P, G, ndates, grid_v, grid_x, model.r, model.dt, tau, dx,
                                      request["Euler_acceleration"])
            sol = sol*np.exp(-truncation_alpha*x)
            l = np.log(lower_barrier/spot)
            func = interp1d(x, sol, kind='cubic')
            price = np.longdouble(func(-l))
            cpu_time = time.time()-t
            result_map[key_string, grid_x] = [price, cpu_time]

        else:
            raise FourierHandlerError("Invalid pricing method requested: "+request_string[0])

    if "data_generation" in request and request["data_generation"] == "ON":
        return list(result_map.values())[0]
    else:
        return result_map, grid_array
