import unittest
from numpy import (
    testing,
    array
)
from pricer.main import *


def get_default_settings():
    default_settings = {
        "type": "FOURIER",
        "num_dates": 64,
        "expiry_T": 1,
        "risk_free_rate": 0.05,
        "dividend_rate": 0.02,
        "spot_price": 100,
        "strike_price": 110,
        "option_type": "CALL",
        "lower_barrier": 85,
        "upper_barrier": 115,
        "direction": "BACKWARD_PROBLEM",
        "iteration_number": 11
    }
    return default_settings


def get_fl_request_map():
    test_request_map = {
        "calculation_requests": [
            # inputs FL with filtering by default
        ]
    }
    return test_request_map


def get_spitzer_request_map():
    test_request_map = {
        "calculation_requests": [
            {
                "method": "WeinerHopfSpitzer_AbateWhitt",
                "distribution": "NormalInverseGaussian",
                "filter": "FILTER_ON",
                "g_filter": "EXPONENTIAL_FILTER",
                "Hilbert": "right_hand_side_cont_func",
                "density_disc_factor": "ON",
                "Hilbert_type": "SINC_METHOD",
                "direction": "BACKWARD_PROBLEM",
                "Euler_acceleration": "ON"
            }
        ]
    }
    return test_request_map


def get_stochastic_vol_request_map():
    test_request_map = {
        "calculation_requests": [
            {
                "method": "FengLinestky",
                "distribution": "Heston",
                "filter": "FILTER_OFF",
                "g_filter": "EXPONENTIAL_FILTER",
                "Hilbert": "right_hand_side_cont_func",
                "density_disc_factor": "ON",
                "Hilbert_type": "SINC_METHOD",
                "direction": "BACKWARD_PROBLEM",
                "Euler_acceleration": "OFF"
            }
        ]
    }
    return test_request_map


def get_mertonjd_request_map():
    test_request_map = {
        "calculation_requests": [
            {
                "method": "WeinerHopfSpitzer_AbateWhitt",
                "distribution": "MertonJD",
                "filter": "FILTER_OFF",
                "g_filter": "EXPONENTIAL_FILTER",
                "Hilbert": "right_hand_side_cont_func",
                "density_disc_factor": "ON",
                "Hilbert_type": "SINC_METHOD",
                "direction": "BACKWARD_PROBLEM",
                "Euler_acceleration": "OFF"
            }
        ]
    }
    return test_request_map


class TestRequest(unittest.TestCase):

    def setUp(self):
        self.settings = get_default_settings()
        self.fl_request = get_fl_request_map()
        self.spitzer_request = get_spitzer_request_map()
        self.stochastic_vol_request = get_stochastic_vol_request_map()
        self.merton_jd_request = get_mertonjd_request_map()

    def test_fl_request(self):
        expected_results = array([
            0.05736132946439094,
            0.022146624595033422,
            0.0074309941193963225,
            0.000999819520401117,
            5.62903741593776e-05,
            1.0405800103946428e-06,
            1.501778407880794e-08,
            3.430804251802755e-12,
            8.049116928532385e-16,
            1e-16
        ])
        expected_cpu_times = array([
            0.01894688606262207,
            0.018912792205810547,
            0.019974946975708008,
            0.022945880889892578,
            0.0249326229095459,
            0.031914472579956055,
            0.04687309265136719,
            0.08078360557556152,
            0.17054438591003418,
            0.5465388298034668,
        ])
        prices = price_barrier_trades(self.settings, self.fl_request)
        results = 0
        cpu_times = 0
        for i in prices:
            results = [x[0] for x in list(prices[i].values())]
            cpu_times = [x[1] for x in list(prices[i].values())]
        testing.assert_array_equal(expected_results, array(results), "default FL request fails")
        testing.assert_array_almost_equal(expected_cpu_times, array(cpu_times), 1, "default FL request fails")

    def test_spitzer_request(self):
        expected_results = array([
            0.0577633359762099,
            0.02191141521730322,
            0.0072635739382554965,
            0.0009928210904975518,
            5.630961223870856e-05,
            1.0406255135028686e-06,
            1.506052754729481e-08,
            5.0642025295477566e-11,
            4.501925915390004e-11,
            4.277655313300599e-11,
        ])
        expected_cpu_times = array([
            0.03391218185424805,
            0.03590130805969238,
            0.03789782524108887,
            0.03491091728210449,
            0.042841434478759766,
            0.05987954139709473,
            0.09973859786987305,
            0.17546367645263672,
            0.39294958114624023,
            0.893608808517456,
        ])
        prices = price_barrier_trades(self.settings, self.spitzer_request)
        results = 0
        cpu_times = 0
        for i in prices:
            results = [x[0] for x in list(prices[i].values())]
            cpu_times = [x[1] for x in list(prices[i].values())]
        testing.assert_array_equal(expected_results, array(results), "default Spitzer request fails")
        testing.assert_array_almost_equal(expected_cpu_times, array(cpu_times), 1, "default Spitzer request fails")

    def test_stochastic_vol_request(self):
        expected_results = array([
            0.038795,
            0.131568,
            0.116228,
            0.116176,
            0.116176,
            0.116176,
            0.116176,
            0.116176,
            0.116176,
            0.116176
        ])
        expected_cpu_times = array([
            0.43284153938293457,
            0.4557840824127197,
            0.8683857917785645,
            1.0470123291015625,
            1.2730050086975098,
            1.8024682998657227,
            5.606551170349121,
            11.605890274047852,
            20.23047399520874,
            40.75491952896118,
        ])
        prices = price_barrier_trades(self.settings, self.stochastic_vol_request)
        results = 0
        cpu_times = 0
        for i in prices:
            results = [x[0] for x in list(prices[i].values())]
            cpu_times = [x[1] for x in list(prices[i].values())]
        testing.assert_array_almost_equal(expected_results, array(results), 1,
                                          "stochastic volatility with Heston request fails")
        testing.assert_array_almost_equal(expected_cpu_times, array(cpu_times), 1,
                                          "stochastic volatility with Heston request fails")

    def test_mertonjd_request(self):
        expected_results = array([
            0.01908438848908478,
            0.06358174798373069,
            0.08860828247084605,
            0.09224671150392279,
            0.09228648051761461,
            0.09228648077432271,
            0.09228648077432276,
            0.0922864807743231,
            0.09228648077432279,
            0.0922864807743229,
        ])
        expected_cpu_times = array([
            0.01998138427734375,
            0.01894855499267578,
            0.020943880081176758,
            0.022940397262573242,
            0.025933027267456055,
            0.03391122817993164,
            0.0478365421295166,
            0.08477401733398438,
            0.20342230796813965,
            0.5515594482421875,
        ])
        prices = price_barrier_trades(self.settings, self.merton_jd_request)
        results = 0
        cpu_times = 0
        for i in prices:
            results = [x[0] for x in list(prices[i].values())]
            cpu_times = [x[1] for x in list(prices[i].values())]
        testing.assert_array_equal(expected_results, array(results), "default Merton JD fails")
        testing.assert_array_almost_equal(expected_cpu_times, array(cpu_times), 1, "default Merton JD fails")
