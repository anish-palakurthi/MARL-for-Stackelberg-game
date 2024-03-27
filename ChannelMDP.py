import numpy as np
import MATest.utils as util

alpha = 3.0
ref_loss = 0.001
ref_dis = 1.0

class Channel(object):
    def __init__(self, dis=100, n_t=1, n_r=4, rho=0.8, sigma2_noise=1e-9):
        self.dis = dis
        self.n_t = n_t
        self.n_r = n_r
        self.sigma2_noise = sigma2_noise
        self.path_loss = ref_loss * np.power(ref_dis / dis, alpha)
        self.sigma2 = self.path_loss
        self.rho = rho
        self.H = util.complexGaussian(self.n_t, self.n_r, scale=np.sqrt(self.sigma2))

    def getCh(self):
        return self.H

    def sampleCh(self):
        error_vector = util.complexGaussian(self.n_t, self.n_r, scale=np.sqrt(self.sigma2), amp=np.sqrt(1 - self.rho * self.rho))
        self.H = self.rho*self.H + error_vector
        return self.getCh()

    def calcChannelGain(self):
        return np.power(np.linalg.norm(self.H), 2)

