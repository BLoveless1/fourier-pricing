import unittest
import numpy as np
from pricer.fourier.utility_functions import *


def get_normal_dist():
    dt = 0.04
    r = 0.05
    q = 0.0
    model = NormalDist(dt, r, q)
    return model


def get_inverse_gaussian():
    dt = 0.04
    r = 0.05
    q = 0.0
    model = NormalInverseGaussian(dt, r, q)
    return model


def get_heston():
    dt = 0.04
    r = 0.05
    q = 0.0
    model = Heston(dt, r, q)
    return model


class TestModels(unittest.TestCase):

    def setUp(self):
        self.tol = 6
        self.nd = get_normal_dist()           # normal distribution
        self.nig = get_inverse_gaussian()     # normal inverse gaussian
        self.h = get_heston()                 # Heston

    def test_normal_model(self):
        self.assertAlmostEqual(self.nd.stddev, 0.4*np.sqrt(0.04), self.tol, "normal model std dev error")

        cumulants = self.nd.get_cumulant()
        self.assertAlmostEqual(cumulants[0], 0.04*(0.05 - 0.5*pow(self.nd.stddev, 2)), self.tol,
                               "normal model cumulant_1 error")
        self.assertAlmostEqual(cumulants[1], 0.04*pow(self.nd.stddev, 2), self.tol, "normal model cumulant_2 error")

        self.assertAlmostEqual(self.nd.get_char_function(1), np.exp(1j*1*self.nd.mean - 0.5*pow(self.nd.stddev*1, 2)),
                               self.tol, "normal model characteristic_function error")

    def test_normal_inverse_gaussian_model(self):
        self.assertAlmostEqual(self.nig.lambda_m, self.nig.beta - self.nig.alpha, self.tol,
                               "normal inverse Gaussian parameter error")

        cumulants = self.nig.get_cumulant()
        c1 = np.divide(self.nig.beta - pow(self.nig.alpha, 2) + pow(self.nig.beta, 2),
                    np.sqrt(pow(self.nig.alpha, 2) - pow(self.nig.beta, 2)))
        c2 = np.sqrt(pow(self.nig.alpha, 2) - pow(self.nig.beta+1, 2))
        self.assertAlmostEqual(cumulants[0], 0.04*(0.05+(self.nig.delta/0.04)*(c1+c2)), self.tol,
                               "normal inverse Guassian cumulant_1 error")

        char_function = self.nig.get_char_function(1)
        cf1 = np.sqrt(pow(self.nig.alpha, 2)-pow(self.nig.beta+1j*1, 2))
        cf2 = np.sqrt(pow(self.nig.alpha, 2)-pow(self.nig.beta, 2))
        self.assertAlmostEqual(char_function, np.exp(-self.nig.delta*(cf1-cf2)), self.tol,
                               "normal inverse Guassian characteristic_function error")

    def test_heston_model(self):
        self.assertAlmostEqual(self.h.stddev, 0.035*np.sqrt(0.04), self.tol, "Heston model std dev error")

        cumulants = self.h.get_cumulant()
        self.assertAlmostEqual(cumulants[0], 0.04*(0.05-0.5*pow(self.h.stddev, 2)), self.tol,
                               "Heston model cumulant 1 error")
        self.assertAlmostEqual(cumulants[1], 0.04*pow(self.h.stddev, 2), self.tol, "Heston model cumulant 2 error")

        char_function = self.h.get_char_function(1)
        self.assertAlmostEqual(char_function, np.exp(1j*self.h.mean-0.5*pow(self.h.stddev, 2)), self.tol,
                               "Heston model characteristic function error")

    # continue with other models


class TestUtilities(unittest.TestCase):

    def setUp(self):
        self.tol = 5
        self.model = get_normal_dist()
        self.model1 = get_inverse_gaussian()
        self.stoch_model = get_heston()

    def test_binomial_function(self):
        expected_result = np.array([1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1])
        binomial_triangle = binomial_1d(10)
        np.testing.assert_array_equal(expected_result, binomial_triangle, "binomial function call error")

    def test_lower_bound_function(self):
        expected_result = 0.798675938
        lower_bound = bound_lower_tail_levy(self.model, 0.12, 3, "backward_problem")
        self.assertAlmostEqual(expected_result, lower_bound, self.tol, "lower bound function error")

    def test_upper_bound_function(self):
        expected_result = 0.79485147
        upper_bound = bound_upper_tail_levy(self.model, 0.12, 3, "backward_problem")
        self.assertAlmostEqual(expected_result, upper_bound, self.tol, "upper bound function error")

    def test_set_alpha_function(self):
        expected_result = -5.0
        alpha = set_alpha(self.model1, 80, 110, 85, "CALL")
        self.assertAlmostEqual(expected_result, alpha, self.tol, "set alpha function error")

    def test_kernel_function(self):
        expected_result = np.array([-1, -0.6, -0.2, 0.2, 0.6])
        kernel_ = kernel(self.model1, 5, -1, 1, 1.0, "ON", "BACKWARD_PROBLEM")[0]
        np.testing.assert_array_almost_equal(expected_result, kernel_, self.tol, "kernel function error")

    def test_gibbs_filter(self):
        expected_result = 0.00000000
        gibbs = gibbs_filter(np.pi/1.5, 12, "EXPONENTIAL_FILTER")
        self.assertAlmostEqual(expected_result, gibbs, self.tol, "gibbs filter function error")

    def test_stochastic_bounds(self):
        expected_lower_bound = 0.0171145
        expected_upper_bound = 0.0596521
        [lower_bound, upper_bound] = stochastic_vol_bounds(self.stoch_model, 1, 2, 0.01)
        np.testing.assert_almost_equal(expected_lower_bound, lower_bound, self.tol,
                                          "stochastic volatility bounds error")
        np.testing.assert_almost_equal(expected_upper_bound, upper_bound, self.tol,
                                          "stochastic volatility bounds error")

    def test_stochastic_grid(self):
        expected_result_x = np.array([5, 3, 1, -1, -3])
        expected_result_xi = np.array([1.57080, 0.94248, 0.31416, -0.31416, -0.94248])
        expected_result_dx = -2.0
        [x, xi, dx] = stochastic_grid(5, 0, 10)
        np.testing.assert_array_almost_equal(expected_result_x, x, self.tol, "stochastic grid error")
        np.testing.assert_array_almost_equal(expected_result_xi, xi, self.tol, "stochastic grid error")
        np.testing.assert_almost_equal(expected_result_dx, dx, self.tol, "stochastic grid error")

    def test_transition_prob_matrix(self):
        expected_result = np.array([2.3e-05, 2.3e-05])
        tp_mat = transition_prob_matrix(self.stoch_model, np.array([1, 1]), 1)
        np.testing.assert_array_almost_equal(expected_result, tp_mat, self.tol, "transition probability matrix error")

    def test_quadrature_nodes(self):
        expected_result_x = np.array([9.098386, 6.0, 2.901613])
        expected_result_w = np.array([-2.2222222, -3.555555, -2.222222])
        [x, w] = quadrature_nodes(3, 10, 2)
        np.testing.assert_array_almost_equal(expected_result_x, x, self.tol, "quadrature nodes error")
        np.testing.assert_array_almost_equal(expected_result_w, w, self.tol, "quadrature nodes error")
