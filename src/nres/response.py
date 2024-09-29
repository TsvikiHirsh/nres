import numpy as np
import pandas as pd
import nres.utils as utils
from scipy.stats import exponnorm, norm, uniform
from scipy.signal import convolve
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import lmfit

class Response:

    def __init__(self,kind="expo_gauss",
                 vary:bool=False,eps:float=1.0e-6,
                 tstep=1.56255e-9,nbins=300):
        
        self.tstep = tstep
        self.grid = np.arange(-nbins,nbins+1,1)
        self.tgrid = self.grid*self.tstep
        
        self.eps = eps
        
        # choose response function
        if kind == "expo_gauss":
            self.function = self.expogauss_response
            self.params = lmfit.create_params(K=dict(value=1., min=0.0001,vary=vary),
                                            x0=dict(value=1e-9,vary=vary),
                                            τ=dict(value=1e-9,min=1e-10,vary=vary))
        elif not kind or kind=="none":
            self.function = self.empty_response
        else:
            raise NotImplementedError("kind error: Only supported response kind is 'expo_gauss' or 'none' ")
        
    def cut_array_symmetric(self, arr, threshold):
        """
        Symmetrically cut the array based on a threshold, ensuring the center element is preserved,
        and the resulting array has an odd number of elements.
        
        Parameters:
        arr (list or array): The input array.
        threshold (float): The threshold value for cutting the array.

        Returns:
        list or array: The symmetrically cut array with an odd number of elements.
        """
        # Ensure the array length is odd
        if len(arr) % 2 == 0:
            raise ValueError("Input array length must be odd.")

        # Find the center index of the array
        center_index = len(arr) // 2
        
        # Find the left index (first below threshold) starting from the center
        left_idx = center_index
        while left_idx >= 0 and arr[left_idx] >= threshold:
            left_idx -= 1
        
        # Find the right index (first below threshold) starting from the center
        right_idx = center_index
        while right_idx < len(arr) and arr[right_idx] >= threshold:
            right_idx += 1
        
        # Calculate the symmetric distance from the center
        left_dist = center_index - (left_idx + 1)  # distance to the left threshold
        right_dist = (right_idx - 1) - center_index  # distance to the right threshold
        dist = max(left_dist, right_dist)
        
        # Create symmetric bounds (odd number of elements)
        left_final = center_index - dist
        right_final = center_index + dist + 1  # +1 to include the right bound
        
        # Return the symmetrically sliced array with odd length
        return arr[left_final:right_final]


    def empty_response(self,**kwargs):
        return [0.,1.,0.]

    def expogauss_response(self,K=0.01,x0=0.,τ=1.0e-9,**kwargs):
        response = exponnorm.pdf(self.tgrid,K,x0,τ)
        response /= sum(response)
        return self.cut_array_symmetric(response,self.eps)
    
    def plot(self,params={},**kwargs):
        # plot the response function
        ax = kwargs.pop("ax",plt.gca())
        xlabel = kwargs.pop("xlabel","t [sec]")
        if not params:
            params = self.params
        y = self.function(**params)
        tof = np.arange(-len(y)//2+1,+len(y)//2+1,1)*self.tstep
        df = pd.Series(y,index=tof,name="Response")
        df.plot(ax=ax,xlabel=xlabel,**kwargs)



# background 
class Background:

    def __init__(self,kind="expo_norm",vary:bool=False):
        # choose background function

        if kind == "polynomial3":
            self.function = self.polynomial3_background
            self.params = lmfit.create_params(b0=dict(value=0.,vary=vary),
                                                         b1=dict(value=0.,vary=vary),
                                                         b2=dict(value=0.,vary=vary))
        elif kind == "polynomial5":
            self.function = self.polynomial5_background
            self.params = lmfit.create_params(b0=dict(value=0.,vary=vary),
                                                         b1=dict(value=0.,vary=vary),
                                                         b2=dict(value=0.,vary=vary),
                                                         b3=dict(value=0.,vary=vary),
                                                         b4=dict(value=0.,vary=vary))
        elif kind == "constant":            
            self.function = self.constant_background
            self.params = lmfit.create_params(b0=dict(value=0.,vary=vary))

        elif not kind or kind=="none":
            self.function = self.empty_background
            self.params = lmfit.Parameters() # empty

        else:
            raise NotImplementedError("kind error: Only supported background kinds are 'none', 'constant', 'polynomial3' or 'polynomial5'")
        
    def empty_background(self,E:np.ndarray,**kwargs) -> np.ndarray:
        # zero background
        return np.zeros_like(E)
    
    def constant_background(self,E:np.ndarray,b0:float=0.,**kwargs) -> np.ndarray:
        # Constant background
        return b0*np.ones_like(E)
    
    def polynomial3_background(self,E:np.ndarray,b0:float=0.,b1:float=1.,b2:float=0.,**kwargs) -> np.ndarray:
        # Background polynomial
        bg = b0 + b1 * np.sqrt(E) + b2 / np.sqrt(E)
        return bg

    def polynomial5_background(self,E:np.ndarray,b0:float=0.,b1:float=1.,
                               b2:float=0.,b3:float=0.,b4:float=0.,**kwargs) -> np.ndarray:
        # Background polynomial
        bg = b0 + b1 * np.sqrt(E) + b2 / np.sqrt(E) + b3 * np.exp(-b4/np.sqrt(E))
        return bg
    
    def plot(self,E:np.ndarray=[],params={},**kwargs):
        # plot the background
        ax = kwargs.pop("ax",plt.gca())
        ls = kwargs.pop("ls","--")
        color = kwargs.pop("color","0.5")
        label = kwargs.pop("label","Background")
        bgargs = {key:params[key].value for key in self.params if key in params}
        if len(E)==0:
            E = np.linspace(0,2e7,100)
        df = pd.Series(self.function(E,**bgargs),index=E,name="Background")
        df.plot(ax=ax,ls=ls,color=color,label=label,**kwargs)