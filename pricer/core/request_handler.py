from pricer.core.plots import *
from pricer.fourier.barrier_pricer import price_fourier_barrier_trades
from pricer.fourier.alpha_quantile_pricer import price_fourier_aq_trades
import csv


class RequestHandlerError(Exception):
    pass


def export_fourier_results(output, results):
    csv_file = csv.writer(output, delimiter=",", quoting=csv.QUOTE_MINIMAL)

    csv_file.writerow([
        'model',
        'grid',
        'price_error',
        'cpu_time'
    ])

    for result_dict in results.values():
        for [model, grid], [price, cpu_time] in result_dict.items():
            values = [
                model,
                grid,
                price,
                cpu_time
            ]
            csv_file.writerow(values)


def price_barrier_trades(requests_file, out=None):

    data = requests_file["data"]
    requests = requests_file["calculation_requests"]
    results, grid = price_fourier_barrier_trades(data, requests)
    error_plots(results, grid)

    if not results:
        return

    if out is None:
        return results
    export_fourier_results(out, results)
    return out


def price_alpha_quantile_trades(requests_file, out=None):

    data = requests_file["data"]
    requests = requests_file["calculation_requests"]
    results, grid = price_fourier_aq_trades(data, requests)

    if not results:
        return

    if out is None:
        return results
    export_fourier_results(out, results)
    return out
