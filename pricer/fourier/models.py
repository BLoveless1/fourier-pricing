import numpy as np


class FourierModel:
    def __init__(self, dt, r, q=0):
        self.dt = dt
        self.r = r
        self.q = q
        self.cumulant_1 = 0
        self.cumulant_2 = 0
        self.char_function = 0
        self.lambda_m = 0
        self.lambda_p = 0


class NormalDist(FourierModel):
    def __init__(self, dt, r, q):
        super().__init__(dt, r, q)
        self.mean = 0
        self.stddev = 0.4*np.sqrt(dt)
        self.lambda_m = 0
        self.lambda_p = 0

    def get_cumulant(self):
        self.cumulant_1 = self.dt*(self.r-0.5*pow(self.stddev, 2))
        self.cumulant_2 = self.dt*pow(self.stddev, 2)
        return self.cumulant_1, self.cumulant_2

    def get_char_function(self, u):
        return np.exp(1j*u*self.mean-0.5*pow(self.stddev*u, 2))


class NormalInverseGaussian(FourierModel):
    def __init__(self, dt, r, q, alpha=15, beta=-5, delta=0.5):
        super().__init__(dt, r, q)
        self.alpha = alpha
        self.beta = beta
        self.delta = delta*dt
        self.lambda_m = self.beta-self.alpha
        self.lambda_p = self.beta+self.alpha
        self.fenglinestky_c = delta
        self.fenglinestky_nu = 1

    def compute_volatility(self, expiry):
        c = pow(pow(self.alpha, 2)-pow(self.beta, 2), -1.5)
        return np.sqrt(pow(self.alpha, 2)*self.delta*c*np.expiry)

    def get_cumulant(self):
        delta = self.delta/self.dt

        c1 = np.divide(self.beta-pow(self.alpha, 2)+pow(self.beta, 2), np.sqrt(pow(self.alpha, 2)-pow(self.beta, 2)))
        c2 = np.sqrt(pow(self.alpha, 2)-pow(self.beta+1, 2))
        self.cumulant_1 = self.dt*(self.r+delta*(c1+c2))
        self.cumulant_2 = np.divide(self.dt*delta*pow(self.alpha, 2),
                                 pow(pow(self.alpha, 2)-pow(self.beta, 2), 1.5))
        return self.cumulant_1, self.cumulant_2

    def get_char_function(self, u):
        cf1 = np.sqrt(pow(self.alpha, 2)-pow(self.beta+1j*u, 2))
        cf2 = np.sqrt(pow(self.alpha, 2)-pow(self.beta, 2))
        return np.exp(-self.delta*(cf1-cf2))


class VarianceGamma(FourierModel):
    def __init__(self, dt, r, q, nu=(1/4), theta=(1/18-1/12)*4):
        super().__init__(dt, r, q)
        self.nu = nu/dt
        self.theta = theta*dt
        self.stddev = np.sqrt(2*4/(18*12))*np.sqrt(dt)
        self.lambda_m = -18
        self.lambda_p = 12

    def get_cumulant(self):
        nu = self.nu*self.dt
        theta = self.theta/self.dt
        s = self.stddev/np.sqrt(self.dt)

        c1 = np.log(pow(1-nu*(0.5*pow(s, 2)+theta), -1/nu))
        self.cumulant_1 = self.dt+(self.r+theta-c1)
        self.cumulant_2 = self.dt*(pow(s, 2)+nu*pow(theta, 2))

    def get_char_function(self, u):
        return pow(1-1j*u*self.theta*self.nu+0.5*self.nu*pow(self.stddev*u, 2), (-1/self.nu))


class Meixner(FourierModel):
    def __init__(self, dt, r, q, alpha=0.3977, beta=-1.4940, delta=0.3462):
        super().__init__(dt, r, q)
        self.alpha = alpha
        self.beta = beta
        self.delta = delta*dt
        self.s = self.alpha*np.sqrt(self.delta)/(4*np.cos(self.beta/2))

    def get_cumulant(self):
        delta = self.delta/self.dt

        c1 = np.log(pow(np.cos(self.beta/2)*1/np.cos((self.alpha+self.beta)/2), 2)*delta)
        self.cumulant_1 = self.dt*(self.r-c1+self.alpha*delta*np.tan(self.beta/2))
        self.cumulant_2 = self.dt*pow(self.alpha, 2)*delta*(pow((1/np.cos(self.beta/2)), 2))/2
        return self.cumulant_1, self.cumulant_2

    def get_char_function(self, u):
        return pow(np.divide(np.cos(self.beta/2), np.cosh(np.divide(self.alpha*u-1j*self.beta, 2))), 2*self.delta)


class CGMY(FourierModel):
    def __init__(self, dt, r, q, C=4, G=50, M=60, Y=0.7):
        super().__init__(dt, r, q)
        self.C = C*dt
        self.G = G
        self.M = M
        self.Y = Y
        self.lambda_m = -self.M
        self.lambda_p = self.G
        self.fenglinestky_c = 2*self.C*abs(np.gamma(-self.Y)*np.cos(np.pi*self.Y/2))
        self.fenglinestky_nu = self.Y

    def get_cumulant(self):
        C = self.C/self.dt

        c1 = pow(-self.G, (self.Y+1))*self.M-pow(self.G, self.Y)*self.M*self.Y
        c2 = (pow(self.G+1, self.Y)+pow(self.M-1, self.Y))*self.M-pow(self.M, self.Y+1)+pow(self.M, self.Y)*self.Y
        self.cumulant_1 = self.dt*(self.G*self.M*self.r-C*(c1+self.G*c2)*np.gamma(-self.Y))/(self.G*self.M)
        c3 = pow(self.G, self.Y)*pow(self.M, 2)+pow(self.G, 2)*pow(self.M, self.Y)
        self.cumulant_2 = self.dt*C*c3*(self.Y-1)*self.Y*np.gamma(-self.Y)/pow(self.G*self.M, 2)
        return self.cumulant_1, self.cumulant_2

    def get_char_function(self, u):
        cf = pow(self.M-1j*u, self.Y)-pow(self.M, self.Y)+pow(self.G+1j*u, self.Y)-pow(self.G, self.Y)
        return np.exp(self.C*np.gamma(-self.Y)*cf)


class KouDE(FourierModel):
    def __init__(self, dt, r, q, s=0.1, kappa=3, pigr=0.3, eta1=40, eta2=12):
        super().__init__(dt, r, q)
        self.s = s*np.sqrt(dt)
        self.kappa = kappa*dt
        self.pigr = pigr
        self.eta1 = eta1
        self.eta2 = eta2
        self.lambda_m = -self.eta1
        self.lambda_p = self.eta2
        self.fenglinestky_c = pow(self.s, 2)/2
        self.fenglinestky_nu = 2

    def get_cumulant(self):
        s = self.s/np.sqrt(self.dt)
        kappa = self.kappa/self.dt

        c1 = self.eta2*self.kappa*self.pigr+self.eta1*(kappa*(self.pigr-1)+self.eta2*self.r)
        c2 = np.divide(kappa*(1+self.eta1*(self.pigr-1)+self.eta2*self.pigr), (self.eta1-1)*(self.eta2+1))
        self.cumulant_1 = np.divide(self.dt*(c1-self.eta1*self.eta2*c2+0.5*pow(s, 2)), self.eta1*self.eta2)
        self.cumulant_2 = self.dt*(-kappa*2*(self.pigr-1)/pow(self.eta2, 2)-2*self.pigr/pow(self.eta1, 2)+pow(s, 2))
        return self.cumulant_1, self.cumulant_2

    def get_char_function(self, u):
        cf = (1-self.pigr)*np.divide(self.eta2, self.eta2+1j*u)+self.pigr*np.divide(self.eta1, self.eta1-1j*u)
        return np.exp(-0.5*pow(self.s*u, 2)+self.kappa*(cf-1))


class MertonJD(FourierModel):
    def __init__(self, dt, r, q, s=0.1, alpha=-0.5, kappa=0.17, delta=0.086):
        super().__init__(dt, r, q)
        self.s = s*np.sqrt(dt)
        self.alpha = alpha
        self.kappa = kappa*dt
        self.delta = delta
        self.lambda_m = 0
        self.lambda_p = 0
        self.fenglinestky_c = pow(s, 2)/2
        self.fenglinestky_nu = 2

    def get_volatility(self, expiry):
        return np.sqrt((pow(self.s, 2)+self.kappa*(pow(self.delta, 2))+pow(self.alpha, 2))*np.expiry)

    def get_cumulant(self):
        s = self.s/np.sqrt(self.dt)
        kappa = self.kappa/self.dt

        self.cumulant_1 = self.dt*((self.alpha+1-np.exp(self.alpha+0.5*pow(self.delta, 2)))*kappa+self.r-0.5*pow(s, 2))
        self.cumulant_2 = self.dt*(kappa*(pow(self.alpha, 2)+pow(self.delta, 2))+pow(s, 2))
        return self.cumulant_1, self.cumulant_2

    def get_char_function(self, u):
        cf = np.exp(1j*u*self.alpha-0.5*pow(self.delta*u, 2))
        return np.exp(-0.5*pow(self.s*u, 2)+self.kappa*(cf-1))


class Heston(FourierModel):
    def __init__(self, dt, r, q, a=4, b=0.035, rho=-0.6, sigma=0.15, v_0=0.06):
        super().__init__(dt, r, q)
        self.a = a  # mean reversion speed
        self.b = b  # mean reversion level
        self.rho = rho
        self.sigma = sigma  # volatility of the variance
        self.v0 = v_0  # initial variance
        self.mean = 0
        self.stddev = self.b*np.sqrt(dt)
        self.lambda_m = 0
        self.lambda_p = 0

    def get_cumulant(self):
        self.cumulant_1 = self.dt*(self.r-0.5*pow(self.stddev, 2))
        self.cumulant_2 = self.dt*pow(self.stddev, 2)
        return self.cumulant_1, self.cumulant_2

    def get_char_function(self, u):
        return np.exp(1j*u*self.mean-0.5*pow(self.stddev*u, 2))
