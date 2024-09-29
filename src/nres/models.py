import lmfit
import numpy as np
import nres.utils as utils
from nres.response import Response, Background
from nres.cross_section import CrossSection
from nres.data import Data
import pandas
import matplotlib.pyplot as plt
from copy import deepcopy 
from typing import List, Optional


class TransmissionModel(lmfit.Model):
    def __init__(self, cross_section, 
                        response:str = "expo_gauss",
                        background:str = "polynomial3",
                        vary_weights:bool=None, 
                        vary_background:bool=None, 
                        vary_tof:bool=None,
                        vary_response:bool=None,
                        **kwargs):
        """
        Initialize the TransmissionModel, a subclass of lmfit.Model.

        Parameters:
        - cross_section: callable
            A function that takes energy (E) as input and returns the cross section.
        - vary_weights: bool, optional (default=False)
            If True, allows the isotope weights to vary during fitting.
        - vary_background: bool, optional (default=False)
            If True, allows the background parameters (b0, b1, b2) to vary during fitting.
        - vary_tof: bool, optional (default=False)
            If True, allows the tof parameters (L0, t0) to vary during fitting.
        - kwargs: dict
            Additional keyword arguments for background parameters, such as `b0`, `b1`, and `b2`.
        """
        super().__init__(self.transmission, **kwargs)

        self.cross_section = cross_section

        self.params = self.make_params()
        if vary_weights is not None:
            self.params += self._make_weight_params(vary=vary_weights)
        if vary_tof is not None:
            self.params += self._make_tof_params(vary=vary_tof,**kwargs)


        self.response = Response(kind=response,vary=vary_response,
                                 tstep=self.cross_section.tstep)
        if vary_response is not None:
            self.params += self.response.params


        self.background = Background(kind=background,vary=vary_background)
        if vary_background is not None:
            self.params += self.background.params


        # set the total atomic weight n [atoms/barn-cm]
        self.n = self.cross_section.n if self.cross_section else 0.01


        

    def transmission(self, E: np.ndarray, thickness: float=1, norm:float=1.,**kwargs):
        """
        Transmission function model with background components.

        Parameters:
        - E: array-like
            The energy values at which to calculate the transmission.
        - thickness: float, optional (default=1)
            The thickness of the material.
        - norm: float, optional (default=1.)


        Returns:
        - T: array-like
            The calculated transmission values.
        """
        E = self._tof_correction(E,**kwargs)

        response = self.response.function(**kwargs)

        weights = [kwargs.pop(key.replace("_",""),val) for key,val in self.cross_section.weights.items()]

        bg = self.background.function(E,**kwargs)

        n = self.n

        # Transmission function
        xs = self.cross_section(E,weights=weights,response=response)

        T = norm * np.exp(- xs * thickness * n) * (1 - bg) + bg
        return T

    def fit(self, data, params=None, emin=0.5e6, emax=20.e6, **kwargs):
        """
        Fit the model to the data.

        Parameters:
        - data: array-like
            The data to fit the model to.
        - params: Parameters object, optional
            The initial parameter values for the fit.
        - kwargs: dict
            Additional keyword arguments passed to the lmfit.Model.fit method.

        Returns:
        - TransmissionModelResult
            The result of the fit.
        """
        # self.cross_section.set_energy_range(emin,emax)
        if isinstance(data,pandas.DataFrame):
            data = data.query(f"{emin}<energy<{emax}")
            weights = kwargs.get("weights",1./data["err"].values)
            fit_result = super().fit(data["trans"].values, params=params or self.params, weights=weights, E=data["energy"].values, **kwargs)
        elif isinstance(data,Data):
            data = data.table.query(f"{emin}<energy<{emax}")
            weights = kwargs.get("weights",1./data["err"].values)
            fit_result = super().fit(data["trans"].values, params=params or self.params, weights=weights, E=data["energy"].values, **kwargs)
        else:
            # Perform the fit using the parent class's fit method
            fit_result = super().fit(data, params=params or self.params, **kwargs)
        self.fit_result = fit_result
        # switch method names
        fit_result.plot_results = deepcopy(fit_result.plot)
        fit_result.plot = self.plot

        # return TransmissionModelResult(fit_result, params or self.params)
        return fit_result
    
    def plot(self,plot_bg=True,**kwargs):
        fig, ax = plt.subplots(2,1,sharex=True,height_ratios=[3.5,1],figsize=(6,5))
        energy = self.fit_result.userkws["E"]
        data = self.fit_result.data
        err = 1./self.fit_result.weights
        best_fit = self.fit_result.best_fit
        residual = self.fit_result.residual
        color = kwargs.pop("color","seagreen")
        ecolor = kwargs.pop("ecolor","0.8")
        ms = kwargs.pop("ms",2)
        ax[0].errorbar(energy,data,err,marker="o",color=color,ms=ms,zorder=-1,ecolor=ecolor,label="Best fit")  
        ax[0].plot(energy,best_fit,color="0.2",label="Data") 
        ax[0].set_ylabel("Transmission")
        ax[0].set_title(self.cross_section.name)
        ax[1].plot(energy,residual,color=color)
        ax[1].set_ylabel("Residuals [1σ]")
        ax[1].set_xlabel("Energy [eV]")
        if plot_bg and self.background.params:
            self.background.plot(E=energy,ax=ax[0],params=self.params,**kwargs)
            ax[0].legend(["Best fit","Background","Data"], fontsize=9,reverse=True,title=f"χ$^2$: {self.fit_result.redchi:.2f}")
        else:
            ax[0].legend(["Best fit","Data"], fontsize=9,reverse=True,title=f"χ$^2$: {self.fit_result.redchi:.2f}")
        plt.subplots_adjust(hspace=0.05)
        
        return ax
    
    def _make_tof_params(self,vary:bool=False,t0:float=0.,L0:float=1.):
        """
        Creates lmfit parameters for tof calibration with scale of the flight path distance L0 and time delay t0
        
        Returns:
            Parameters: An lmfit Parameters object
        """
        params = lmfit.Parameters()
        params.add("L0", value=L0, min=0.5, max= 1.5, vary=vary)
        params.add("t0", value=t0, vary=vary)
        return params



    def _make_weight_params(self, vary: bool = False):
        """
        Creates lmfit parameters based on a pandas.Series of initial weights, ensuring the sum of weights is 1.
        
        Returns:
            Parameters: An lmfit Parameters object with weights normalized to sum to 1.
        """
        params = lmfit.Parameters()
        weight_series = deepcopy(self.cross_section.weights)
        weight_series.index = weight_series.index.str.replace("-", "")
        param_names = weight_series.index
        N = len(weight_series)

        # Normalize the input weights to sum to 1
        weights = np.array(weight_series / weight_series.sum(), dtype=np.float64)

        if N == 1:
            # Special case: if N=1, the weight is always 1
            params.add(f'{param_names[0]}', value=1., vary=False)
        else:

            last_weight = weights[-1]
            # Add (N-1) free parameters corresponding to the first (N-1) items
            for i, name in enumerate(param_names[:-1]):
                initial_value = weights[i]  # Use weight values
                params.add(f'p{i+1}',value=np.log(weights[i]/last_weight),min=-14,max=14) # limit to 1ppm
            
            # Define the normalization expression
            normalization_expr = ' + '.join([f'exp(p{i+1})' for i in range(N-1)])
            
            # Add weights based on the free parameters
            for i, name in enumerate(param_names[:-1]):
                params.add(f'{name}', expr=f'exp(p{i+1}) / (1 + {normalization_expr})')
            
            # The last weight is 1 minus the sum of the previous weights
            params.add(f'{param_names[-1]}', expr=f'1 / (1 + {normalization_expr})')

        return params

    def set_cross_section(self, xs: 'CrossSection', inplace: bool = True) -> 'TransmissionModel':
        """
        Sets a new cross-section for the transmission model.

        Parameters:
        - xs: CrossSection
            The new cross-section to apply.
        - inplace: bool, optional, default=True
            If True, modify the current object. If False, return a new modified object.

        Returns:
        - TransmissionModel: The updated model (self or new instance).
        """
        if inplace:
            self.cross_section = xs
            params = self._make_weight_params()
            self.params += params
            return self
        else:
            new_self = deepcopy(self)
            new_self.cross_section = xs
            params = new_self._make_weight_params()
            new_self.params += params
            return new_self

    def update_params(self, params: dict = {}, values_only: bool = True, inplace: bool = True):
        """
        Updates the parameters of the transmission model.

        Parameters:
        - params: dict
            A dictionary containing the new parameters to update.
        - values_only: bool, optional, default=True
            If True, update only the values of the parameters.
        - inplace: bool, optional, default=True
            If True, modify the current object. If False, return a new modified object.

        Returns:
        - TransmissionModel: The updated model (self or new instance).
        """
        if inplace:
            if values_only:
                for param in params:
                    self.params[param].set(value=params[param].value)
            else:
                self.params = params
        else:
            new_self = deepcopy(self)
            if values_only:
                for param in params:
                    new_self.params[param].set(value=params[param].value)
            else:
                new_self.params = params
            return new_self  # Ensure a return statement in the non-inplace scenario.

    def vary_all(self, vary: Optional[bool] = None, except_for: List[str] = []):
        """
        Toggles the 'vary' attribute for all parameters in the model.

        Parameters:
        - vary: bool, optional
            If provided, set this value for all parameters' 'vary' attribute.
        - except_for: list of str, optional
            A list of parameters that should be excluded from this operation.

        Returns:
        - None
        """
        if vary is not None:
            for param in self.params:
                if param not in except_for:
                    self.params[param].set(vary=vary)

    def _tof_correction(self, E, L0: float = 1.0, t0: float = 0.0, **kwargs):
        """
        Applies a time-of-flight correction to the energy based on the given parameters.

        Parameters:
        - E: energy (in arbitrary units)
            The energy value to be corrected.
        - L0: float, optional, default=1.0
            The reference length factor for correction.
        - t0: float, optional, default=0.0
            The time offset for the correction.
        - kwargs: additional arguments (currently unused)

        Returns:
        - E: Corrected energy value.
        """
        tof = utils.energy2time(E, self.cross_section.L)
        dtof = (1.0 - L0) * tof + t0
        E = utils.time2energy(tof + dtof, self.cross_section.L)
        return E

