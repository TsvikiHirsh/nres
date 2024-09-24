import lmfit
import numpy as np
import nres.utils as utils
from nres.response import Response
from nres.data import Data
import pandas
import matplotlib.pyplot as plt
from copy import copy


class TransmissionModel(lmfit.Model):
    def __init__(self, cross_section, 
                 vary_weights=False, 
                 vary_background=False, 
                 vary_tof=False,
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
        for isotope in self.cross_section.isotopes:
            self.params.add(isotope.replace("-", ""),
                            value=self.cross_section.isotopes[isotope],
                            min=0,
                            max=1,
                            vary=vary_weights)

        # Initialize background parameters with provided values or defaults
        bg_args = {"b0": kwargs.get("b0", 0.), "b1": kwargs.get("b1", 0.), "b2": kwargs.get("b2", 0.)}
        for b in bg_args:
            self.params.add(b, value=bg_args[b], vary=vary_background)

        # Initialize tof parameters with provided values or defaults
        tof_args = {"L0": kwargs.get("L0", 1.), "t0": kwargs.get("t0", 0.)}
        self.params.add("L0", value=tof_args["L0"], min=0.5, max= 1.5, vary=vary_tof)
        self.params.add("t0", value=tof_args["t0"], min=-5.0e-9, max= 5.0e-9, vary=vary_tof)

        # set the n parameter as fixed
        if self.cross_section.n:
            self.params.add("n", value=self.cross_section.n, vary=False)
        else:
            self.params.add("n", value=0.01, vary=False)

        self.response = Response()

    def transmission(self, E, thickness=1, n=0.01, norm=1., b0=0., b1=0., b2=0.,L0=1.,t0=0.,**response_kw):
        """
        Transmission function model with background components.

        Parameters:
        - E: array-like
            The energy values at which to calculate the transmission.
        - thickness: float, optional (default=1)
            The thickness of the material.
        - n: float, optional (default=0.01)
            The number density of the material. units [atoms/barn-cm]
        - norm: float, optional (default=1.)
            Normalization factor for the transmission.
        - b0: float, optional (default=0.)
            Background parameter (constant term).
        - b1: float, optional (default=0.)
            Background parameter (linear term).
        - b2: float, optional (default=0.)
            Background parameter (quadratic term).

        Returns:
        - T: array-like
            The calculated transmission values.
        """
        tof = utils.energy2time(E,self.cross_section.L)
        dtof = (1.-L0)*tof + t0
        E = utils.time2energy(tof+dtof,self.cross_section.L)

        # Background polynomial
        bg = b0 + b1 * np.sqrt(E) + b2 / np.sqrt(E)
        
        response = self.response.function(**response_kw)
        weights = [response_kw.get(key,self.cross_section.isotopes[key]) for key in self.cross_section.isotopes]
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
        fit_result.plot_results = copy(fit_result.plot)
        fit_result.plot = self.plot

        # return TransmissionModelResult(fit_result, params or self.params)
        return fit_result
    
    def plot(self,**kwargs):
        fig, ax = plt.subplots(2,1,sharex=True,height_ratios=[3.5,1],figsize=(6,5))
        energy = self.fit_result.userkws["E"]
        data = self.fit_result.data
        err = 1./self.fit_result.weights
        best_fit = self.fit_result.best_fit
        residual = self.fit_result.residual
        color = kwargs.pop("color","seagreen")
        ecolor = kwargs.pop("ecolor","0.8")
        ms = kwargs.pop("ms",2)
        ax[0].errorbar(energy,data,err,marker="o",color=color,ms=ms,zorder=-1,ecolor=ecolor)  
        ax[0].plot(energy,best_fit,color="0.2") 
        ax[0].set_ylabel("Transmission")
        ax[0].set_title(self.cross_section.name)
        ax[1].plot(energy,residual,color=color)
        ax[1].set_ylabel("residuals [1σ]")
        ax[1].set_xlabel("Energy [eV]")
        plt.subplots_adjust(hspace=0.05)
        ax[0].legend(["Best fit","Data"],title=f"χ$^2$: {self.fit_result.redchi:.2f}")
        return ax
    

