from pricer.fourier.utility_functions import *
from pricer.fourier.alpha_quantile.spitzer_wh_izt import *
from pricer.fourier.alpha_quantile.feng_linetsky_green import *
import time


class FourierHandlerError(Exception):
    pass


def price_fourier_aq_trades(data, requests):

    T = data["expiry_T"]
    ndates = data["num_dates"]
    dt = T/ndates
    rf = data["risk_free_rate"]
    q = data["dividend_rate"]
    S = data["spot_price"]
    K = data["strike_price"]
    option_type = data["option_type"]
    scale_factor = S
    direction = data["direction"]

    request_map = calculation_map_aq(requests)
    max_loop_num = data['iteration_number']
    results = {}
    grid = 1
    for request in request_map:
        results[request], grid = price_handler(request_map[request], max_loop_num, K, q, dt, rf, T,
                                               ndates, scale_factor, S, option_type, direction)

    return results, grid


def price_handler(request, max_loop_num, K, q, dt, rf, T, ndates, scale_factor, spot, option_type,
                  direction=BACKWARD_PROBLEM):

    model, cavers_r = get_model(request["distribution"], dt, rf, q)
    lower_bound, upper_bound = deterministic_bounds(model, direction)
    upper_bound = max(abs(lower_bound), upper_bound)*1.5
    truncation_L = spot*np.exp(-upper_bound)
    truncation_U = spot*np.exp(upper_bound)  # upper bound of support
    truncation_alpha = 0
    alpha_threshold = request["alpha_date_threshold"]

    results, grid = price_array_levy(request, model, max_loop_num, truncation_alpha, upper_bound, truncation_L,
                                     truncation_U, K, alpha_threshold, ndates, scale_factor, T, option_type, cavers_r)

    return results, grid


def price_array_levy(request, model, max_loop_num, truncation_alpha, bound, truncation_L, truncation_U, K,
                     alpha_threshold, ndates, scale_factor, T, option_type, cavers_r):
    result_map = {}
    grid_array = np.ones(max_loop_num-1)
    start = 1
    if "data_generation" in request and request["data_generation"] == "ON":
        start = max_loop_num-1
    for i in range(start, max_loop_num):
        grid = round(pow(2, i + 4))
        grid_array[i-1] = grid
        # Compute the grid and the discounted kernel for Fourier-based methods
        x, h, xi, H = kernel(model, grid, -bound, bound, truncation_alpha, request["density_disc_factor"],
                             request["direction"])

        if request["filter"] == "FILTER_ON":
            key_string = request['method'] + "_FilterOn" + "_" + request["distribution"]
            gf = gibbs_filter(xi, 14, request["g_filter"])
            gf_1 = gibbs_filter(xi, 12, request["g_filter"])
        else:
            key_string = request['method'] + "_FilterOff" + "_" + request["distribution"]
            gf = np.ones(grid)
            gf_1 = np.ones(grid)
        H = np.multiply(H, gf)

        t = time.time()
        if request['method'] == 'AbateWhitt':
            Solmm = izt_aw(H, request["Hilbert_type"], ndates-alpha_threshold, request["Euler_acceleration"], "PUT")
            SolMp = izt_aw(H, request["Hilbert_type"], alpha_threshold, request["Euler_acceleration"], "CALL")

        elif request['method'] == 'FengLinetskyGreen':
            Solmm = feng_linetsky_green(H, request["Hilbert_type"], ndates-alpha_threshold, "PUT")
            SolMp = feng_linetsky_green(H, request["Hilbert_type"], alpha_threshold, "CALL")

        elif request['method'] == 'CaversAccelerated':
            Solmm = izt_cavers_acc(H, request["Hilbert_type"], ndates-alpha_threshold, "PUT")
            SolMp = izt_cavers_acc(H, request["Hilbert_type"], alpha_threshold, "CALL")

        else:
            raise FourierHandlerError("Invalid inverse Z-transform in request: " + request['method'])

        # Compute the scale, payoff and its FT for Fourier-based methods
        s, g, G = payoff(x, xi, truncation_alpha, K, truncation_L, truncation_U, scale_factor, option_type)
        Solmm_conv = 0.5*(Solmm+np.conj(np.roll(np.flip(Solmm), 1)))
        SolMp_conv = 0.5*(SolMp+np.conj(np.roll(np.flip(SolMp), 1)))
        Dist2 = np.multiply(Solmm_conv, SolMp_conv)

        sol = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(np.multiply(gf_1*np.conj(G), Dist2)))))/(2*bound)
        index = int(len(sol)/2)
        price = np.exp(-model.r*T) * sol[index]
        cpu_time = time.time()-t
        result_map[key_string, grid] = [price, cpu_time]

    if "data_generation" in request and request["data_generation"] == "ON":
        return list(result_map.values())[0]
    else:
        return result_map, grid_array
