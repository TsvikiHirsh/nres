import numpy as np
import nres.utils as utils
from scipy.stats import exponnorm, norm, uniform
from scipy.signal import convolve
from scipy.ndimage import convolve1d

class Response:

    def __init__(self,kind="expo_gauss",L=10.49,tbin=1.5625e-9,nbins=700,eps=1e-6,**kwargs):
        self.tbin = tbin
        self.L = L
        grid = np.arange(-nbins,nbins+1,1)
        self.tgrid = grid*self.tbin
        self.eps = eps
        
        if kind == "expo_gauss":
            self.function = self.expogauss_response
        elif kind == "tof_calib":
            self.function = self.tof_calibration
        else:
            raise ValueError("kind can be either 'expo_gauss' or 'tof_calib'")
        

    def expogauss_response(self,tof,K=10,x0=0.,τ=5e-9,L0=1.,t0=0.):
        response = exponnorm.pdf(self.tgrid,K,x0,τ)
        return response

    
    def __call__(self,tof,**kwargs):
        return response