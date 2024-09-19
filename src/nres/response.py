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
        
    def cut_array_symmetric(self, arr, threshold):
        # Find the center index of the array (since the length is odd)
        center_index = len(arr) // 2
        
        # Find the first index to the left of the center that falls below the threshold
        left_idx = center_index
        while left_idx >= 0 and arr[left_idx] >= threshold:
            left_idx -= 1
        
        # Find the first index to the right of the center that falls below the threshold
        right_idx = center_index
        while right_idx < len(arr) and arr[right_idx] >= threshold:
            right_idx += 1
        
        # Calculate distances from the center to the left and right thresholds
        left_dist = center_index - (left_idx + 1)  # distance to the left threshold (add 1 because we overshoot in the while loop)
        right_dist = (right_idx - 1) - center_index  # distance to the right threshold (subtract 1 because we overshoot in the while loop)
        
        # Use the larger distance (furthest index) for symmetric cutting
        dist = max(left_dist, right_dist)
        
        # Create symmetric slicing bounds
        left_final = center_index - dist
        right_final = center_index + dist + 1  # +1 to include the right bound
        
        # Return the symmetrically sliced array
        return arr[left_final:right_final]

    def expogauss_response(self,K=1,x0=0.,τ=1e-9,**kwargs):
        response = exponnorm.pdf(self.tgrid,K,x0,τ)
        response /= sum(response)
        return self.cut_array_symmetric(response,self.eps)
