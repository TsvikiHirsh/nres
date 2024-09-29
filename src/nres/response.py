import numpy as np
import pandas as pd
import nres.utils as utils
from scipy.stats import exponnorm, norm, uniform
from scipy.signal import convolve
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import lmfit

class Response:
    def __init__(self, kind="expo_gauss", vary: bool = False, eps: float = 1.0e-6,
                 tstep=1.56255e-9, nbins=300):
        """
        Initializes the Response object with specified parameters.

        Parameters:
        kind (str): The type of response function to use. Options are 'expo_gauss' or 'none'.
        vary (bool): If True, the parameters can vary during fitting. Default is False.
        eps (float): The threshold for cutting the response array symmetrically. Default is 1.0e-6.
        tstep (float): The time step for the response function. Default is 1.56255e-9 seconds.
        nbins (int): The number of bins for the response function. Default is 300.
        """
        self.tstep = tstep
        self.grid = np.arange(-nbins, nbins + 1, 1)
        self.tgrid = self.grid * self.tstep
        self.eps = eps

        # choose response function
        if kind == "expo_gauss":
            self.function = self.expogauss_response
            self.params = lmfit.create_params(
                K=dict(value=1., min=0.0001, vary=vary),
                x0=dict(value=1e-9, vary=vary),
                τ=dict(value=1e-9, min=1e-10, vary=vary)
            )
        elif not kind or kind == "none":
            self.function = self.empty_response
        else:
            raise NotImplementedError("kind error: Only supported response kind is 'expo_gauss' or 'none' ")

    def cut_array_symmetric(self, arr, threshold):
        """
        Symmetrically cuts the array based on a threshold, ensuring the center element is preserved,
        and the resulting array has an odd number of elements.

        Parameters:
        arr (list or np.ndarray): The input array to be cut.
        threshold (float): The threshold value for cutting the array.

        Returns:
        np.ndarray: The symmetrically cut array with an odd number of elements.

        Raises:
        ValueError: If the input array length is not odd.
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

    def empty_response(self, **kwargs):
        """
        Generates an empty response array.

        Returns:
        list: A list containing [0.0, 1.0, 0.0] as the response.
        """
        return [0., 1., 0.]

    def expogauss_response(self, K=0.01, x0=0., τ=1.0e-9, **kwargs):
        """
        Computes the exponential-Gaussian response function.

        Parameters:
        K (float): The shape parameter for the exponential function.
        x0 (float): The location parameter for the Gaussian function.
        τ (float): The scale parameter for the exponential function.

        Returns:
        np.ndarray: The normalized response array after cutting with the threshold defined by eps.
        """
        response = exponnorm.pdf(self.tgrid, K, x0, τ)
        response /= sum(response)
        return self.cut_array_symmetric(response, self.eps)

    def plot(self, params={}, **kwargs):
        """
        Plots the response function.

        Parameters:
        params (dict): The parameters for the response function. If empty, the default parameters are used.
        **kwargs: Additional keyword arguments passed to the plot function.

        Returns:
        None
        """
        # plot the response function
        ax = kwargs.pop("ax", plt.gca())
        xlabel = kwargs.pop("xlabel", "t [sec]")
        if not params:
            params = self.params
        y = self.function(**params)
        tof = np.arange(-len(y) // 2 + 1, +len(y) // 2 + 1, 1) * self.tstep
        df = pd.Series(y, index=tof, name="Response")
        df.plot(ax=ax, xlabel=xlabel, **kwargs)


# Background class
class Background:
    def __init__(self, kind="expo_norm", vary: bool = False):
        """
        Initializes the Background object with specified parameters.

        Parameters:
        kind (str): The type of background function to use. Options are 'none', 'constant', 'polynomial3', or 'polynomial5'.
        vary (bool): If True, the parameters can vary during fitting. Default is False.
        """
        # choose background function
        if kind == "polynomial3":
            self.function = self.polynomial3_background
            self.params = lmfit.create_params(
                b0=dict(value=0., vary=vary),
                b1=dict(value=0., vary=vary),
                b2=dict(value=0., vary=vary)
            )
        elif kind == "polynomial5":
            self.function = self.polynomial5_background
            self.params = lmfit.create_params(
                b0=dict(value=0., vary=vary),
                b1=dict(value=0., vary=vary),
                b2=dict(value=0., vary=vary),
                b3=dict(value=0., vary=vary),
                b4=dict(value=0., vary=vary)
            )
        elif kind == "constant":
            self.function = self.constant_background
            self.params = lmfit.create_params(b0=dict(value=0., vary=vary))
        elif not kind or kind == "none":
            self.function = self.empty_background
            self.params = lmfit.Parameters()  # empty
        else:
            raise NotImplementedError("kind error: Only supported background kinds are 'none', 'constant', 'polynomial3' or 'polynomial5'")

    def empty_background(self, E: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generates a zero background array.

        Parameters:
        E (np.ndarray): The energy values (not used in this function).

        Returns:
        np.ndarray: An array of zeros with the same shape as E.
        """
        return np.zeros_like(E)

    def constant_background(self, E: np.ndarray, b0: float = 0., **kwargs) -> np.ndarray:
        """
        Generates a constant background array.

        Parameters:
        E (np.ndarray): The energy values for which the background is calculated.
        b0 (float): The constant value for the background.

        Returns:
        np.ndarray: An array of the constant value b0 with the same shape as E.
        """
        return b0 * np.ones_like(E)

    def polynomial3_background(self, E: np.ndarray, b0: float = 0., b1: float = 1., b2: float = 0., **kwargs) -> np.ndarray:
        """
        Computes a third-degree polynomial background.

        Parameters:
        E (np.ndarray): The energy values for which the background is calculated.
        b0 (float): Coefficient for the constant term.
        b1 (float): Coefficient for the first-degree term.
        b2 (float): Coefficient for the second-degree term.

        Returns:
        np.ndarray: The computed polynomial background values for the given energy values.
        """
        bg = b0 + b1 * np.sqrt(E) + b2 / np.sqrt(E)
        return bg

    def polynomial5_background(self, E: np.ndarray, b0: float = 0., b1: float = 1.,
                               b2: float = 0., b3: float = 0., b4: float = 0., **kwargs) -> np.ndarray:
        """
        Computes a fifth-degree polynomial background.

        Parameters:
        E (np.ndarray): The energy values for which the background is calculated.
        b0 (float): Coefficient for the constant term.
        b1 (float): Coefficient for the first-degree term.
        b2 (float): Coefficient for the second-degree term.
        b3 (float): Coefficient for the third-degree term.
        b4 (float): Coefficient for the fourth-degree term.

        Returns:
        np.ndarray: The computed polynomial background values for the given energy values.
        """
        bg = b0 + b1 * np.sqrt(E) + b2 / np.sqrt(E) + b3 * E + b4 * E**2
        return bg

    def plot(self, E: np.ndarray, params={}, **kwargs):
        """
        Plots the background function.

        Parameters:
        E (np.ndarray): The energy values for which the background is calculated.
        params (dict): The parameters for the background function. If empty, the default parameters are used.
        **kwargs: Additional keyword arguments passed to the plot function.

        Returns:
        None
        """
        # plot the background function
        ax = kwargs.pop("ax", plt.gca())
        if not params:
            params = self.params
        y = self.function(E, **params)
        df = pd.Series(y, index=E, name="Background")
        df.plot(ax=ax, **kwargs)
