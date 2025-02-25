import lmfit
import numpy as np
import nres.utils as utils
from nres.response import Response, Background
from nres.cross_section import CrossSection
from nres.data import Data
import pandas
import matplotlib.pyplot as plt
from copy import deepcopy 
from typing import List, Optional, Union


class TransmissionModel(lmfit.Model):
    def __init__(self, cross_section, 
                        response: str = "expo_gauss",
                        background: str = "polynomial3",
                        tof_calibration: str = "linear",
                        vary_weights: bool = None, 
                        vary_background: bool = None, 
                        vary_tof: bool = None,
                        vary_response: bool = None,
                        **kwargs):
        """
        Initialize the TransmissionModel, a subclass of lmfit.Model.

        Parameters
        ----------
        cross_section : callable
            A function that takes energy (E) as input and returns the cross section.
        response : str, optional
            The type of response function to use, by default "expo_gauss".
        tof_calibration : str, optional
            The type of TOF calibration to use, by default "linear". other options are "full" to include the energy dependent corrections.
        background : str, optional
            The type of background function to use, by default "polynomial3".
        vary_weights : bool, optional
            If True, allows the isotope weights to vary during fitting.
        vary_background : bool, optional
            If True, allows the background parameters (b0, b1, b2) to vary during fitting.
        vary_tof : bool, optional
            If True, allows the TOF (time-of-flight) parameters (L0, t0) to vary during fitting.
        vary_response : bool, optional
            If True, allows the response parameters to vary during fitting.
        kwargs : dict, optional
            Additional keyword arguments for model and background parameters.

        Notes
        -----
        This model calculates the transmission function as a combination of 
        cross-section, response function, and background.
        """
        super().__init__(self.transmission, **kwargs)

        self.cross_section = CrossSection()

        for material in cross_section.materials:
            self.cross_section += CrossSection(**{material:cross_section.materials[material]},
            splitby=cross_section.materials[material]["splitby"])

        self.params = self.make_params()
        if vary_weights is not None:
            self.params += self._make_weight_params(vary=vary_weights)
        if vary_tof is not None:
            self.params += self._make_tof_params(vary=vary_tof,kind=tof_calibration,**kwargs)


        self.response = Response(kind=response,vary=vary_response,
                                 tstep=self.cross_section.tstep)
        if vary_response is not None:
            self.params += self.response.params


        self.background = Background(kind=background,vary=vary_background)
        if vary_background is not None:
            self.params += self.background.params


        # set the total atomic weight n [atoms/barn-cm]
        self.n = self.cross_section.n if self.cross_section else 0.01


        

    def transmission(self, E: np.ndarray, thickness: float = 1, norm: float = 1., **kwargs):
        """
        Transmission function model with background components.

        Parameters
        ----------
        E : np.ndarray
            The energy values at which to calculate the transmission.
        thickness : float, optional
            The thickness of the material (in cm), by default 1.
        norm : float, optional
            Normalization factor, by default 1.
        kwargs : dict, optional
            Additional arguments for background, response, or cross-section.

        Returns
        -------
        np.ndarray
            The calculated transmission values.

        Notes
        -----
        This function combines the cross-section with the response and background 
        models to compute the transmission, which is given by:

        .. math:: T(E) = \text{norm} \cdot e^{- \sigma \cdot \text{thickness} \cdot n} \cdot (1 - \text{bg}) + \text{bg}
        
        where `sigma` is the cross-section, `bg` is the background function, and `n` is the total atomic weight.
        """
        E = self._tof_correction(E,**kwargs)

        response = self.response.function(**kwargs)

        weights = deepcopy(self.cross_section.weights)
        weights = [kwargs.pop(key.replace("_",""),val) for key,val in weights.items()]

        bg = self.background.function(E,**kwargs)

        k = kwargs.get("k",1.) # background factor, relevant for some of the background models

        n = self.n

        # Transmission function
        xs = self.cross_section(E,weights=weights,response=response)

        T = norm * np.exp(- xs * thickness * n) * (1 - bg) + k*bg
        return T

    def fit(self, data, params=None, emin=0.5e6, emax=20.e6, **kwargs):
        """
        Fit the model to the data.

        Parameters
        ----------
        data : pandas.DataFrame or nres.data.Data
            The data to fit the model to.
        params : lmfit.Parameters, optional
            Initial parameter values for the fit. If None, the current model parameters will be used.
        emin : float, optional
            The minimum energy for fitting, by default 0.5e6.
        emax : float, optional
            The maximum energy for fitting, by default 20.e6.
        kwargs : dict, optional
            Additional arguments passed to the lmfit.Model.fit method.

        Returns
        -------
        lmfit.model.ModelResult
            The result of the fit.

        Notes
        -----
        This function applies energy filtering to the input data based on `emin` and `emax`,
        then fits the transmission model to the filtered data.
        """
        # self.cross_section.set_energy_range(emin,emax)
        params = deepcopy(params or self.params)
        if isinstance(data,pandas.DataFrame):
            data = data.query(f"{emin}<energy<{emax}")
            weights = kwargs.get("weights",1./data["err"].values)
            fit_result = super().fit(data["trans"].values, params=params, weights=weights, E=data["energy"].values, **kwargs)
        elif isinstance(data,Data):
            data = data.table.query(f"{emin}<energy<{emax}")
            weights = kwargs.get("weights",1./data["err"].values)
            fit_result = super().fit(data["trans"].values, params=params, weights=weights, E=data["energy"].values, **kwargs)
        else:
            # Perform the fit using the parent class's fit method
            fit_result = super().fit(data, params=params, **kwargs)
        self.fit_result = fit_result
        # switch method names
        fit_result.plot_results = deepcopy(fit_result.plot)
        fit_result.plot = self.plot
        fit_result.weighted_thickness = self.weighted_thickness

        # return TransmissionModelResult(fit_result, params or self.params)
        return fit_result
    
        
    def plot(self, data: "nres.Data" = None, plot_bg: bool = True, correct_tof: bool = True, **kwargs):
        """
        Plot the results of the fit or model.

        Parameters
        ----------
        data : nres.Data, optional
            Show data alongside the model (useful before performing the fit).
        plot_bg : bool, optional
            Whether to include the background in the plot, by default True.
        correct_tof : bool, optional
            Apply TOF correction if L0 and t0 parameters are present, by default True.
        kwargs : dict, optional
            Additional plot settings like color, marker size, etc.

        Returns
        -------
        matplotlib.axes.Axes
            The axes of the plot.
        """
        fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=[3.5, 1], figsize=(6, 5))
        data_object = data.table.dropna().copy() if data else None

        if hasattr(self, "fit_result"):
            # Use fit results
            energy = self.fit_result.userkws["E"]
            data = self.fit_result.data
            err = 1.0 / self.fit_result.weights
            best_fit = self.fit_result.best_fit
            residual = self.fit_result.residual
            params = self.fit_result.params
            chi2 = self.fit_result.redchi
            fit_label = "Best fit"
        else:
            # Use model (no fit yet)
            fit_label = "Model"
            params = self.params
            if data is not None:
                energy = data_object["energy"]
                data = data_object["trans"]
                err = data_object["err"]
                best_fit = self.eval(params=params, E=energy.values)
                residual = (data - best_fit) / err
                # Calculate chi2 for the model
                chi2 = np.sum(((data - best_fit) / err) ** 2) / (len(data) - len(params))
            else:
                energy = self.cross_section.table.dropna().index.values
                data = np.nan * np.ones_like(energy)
                err = np.nan * np.ones_like(energy)
                best_fit = self.eval(params=params, E=energy)
                residual = np.nan * np.ones_like(energy)
                chi2 = np.nan

        # Apply TOF correction if enabled and L0, t0 parameters are present
        if correct_tof and "L0" in params and "t0" in params:
            L0 = params["L0"].value
            t0 = params["t0"].value
            t1 = params["t1"].value if "t1" in params else 0.0
            t2 = params["t2"].value if "t2" in params else 0.0
            energy = self._tof_correction(energy, L0=L0, t0=t0, t1=t1, t2=t2)

        # Plot settings
        color = kwargs.pop("color", "seagreen")
        title = kwargs.pop("title", self.cross_section.name)
        ecolor = kwargs.pop("ecolor", "0.8")
        ms = kwargs.pop("ms", 2)

        # Plot data and best-fit/model
        ax[0].errorbar(energy, data, err, marker="o", color=color, ms=ms, zorder=-1, ecolor=ecolor, label="Data")
        ax[0].plot(energy, best_fit, color="0.2", label=fit_label)
        ax[0].set_ylabel("Transmission")
        ax[0].set_title(title)

        # Plot residuals
        ax[1].plot(energy, residual, color=color)
        ax[1].set_ylabel("Residuals [1σ]")
        ax[1].set_xlabel("Energy [eV]" )

        # Plot background if requested
        if plot_bg and self.background.params:
            self.background.plot(E=energy, ax=ax[0], params=params, **kwargs)
            legend_labels = [fit_label, "Background", "Data"]
        else:
            legend_labels = [fit_label, "Data"]

        # Set legend with chi2 value
        ax[0].legend(
            legend_labels,
            fontsize=9,
            reverse=True,
            title=f"χ$^2$: {chi2:.2f}"
        )

        plt.subplots_adjust(hspace=0.05)
        return ax

    def weighted_thickness(self,params=None):
        """Returns the weighted thickness in [cm]

        Args:
            params (lmfit.Parameters, optional): parameters object. Defaults to None.
        """

        weights = self.cross_section.weights
        if params:
            thickness = params["thickness"].value
        elif hasattr(self,"fit_result"):
            thickness = self.fit_result.values["thickness"]
        else:
            thickness = self.params["thickness"].value
        return thickness * weights

    
    def _make_tof_params(self, vary: bool = False, kind:str = "linear", L0: float = 1.,
                                 t0: float = 0.,t1: float = 0., t2: float = 0.):
        """
        Create time-of-flight (TOF) parameters for the model.

        Parameters
        ----------
        vary : bool, optional
            Whether to allow these parameters to vary during fitting, by default False.
        kind : str, optional
            The type of TOF correction to apply, by default "linear". other options are "full" to include the energy dependent corrections.
        L0 : float, optional
            Initial flight path distance scale parameter, by default 1.
        t0 : float, optional
            Initial time offset parameter, by default 0.
        t1 : float, optional
            Initial linear correction parameter, by default 0.
        t2 : float, optional
            Initial logarithmic correction parameter, by default 0.


        Returns
        -------
        lmfit.Parameters
            The TOF-related parameters.
        """
        params = lmfit.Parameters()
        params.add("L0", value=L0, min=0.5, max= 1.5, vary=vary)
        params.add("t0", value=t0, vary=vary)
        if kind == "full":
            params.add("t1", value=t1, vary=vary)
            params.add("t2", value=t2, vary=vary)
        return params



    def _make_weight_params(self, vary: bool = False):
        """
        Create lmfit parameters based on initial isotope weights.

        Parameters
        ----------
        vary : bool, optional
            Whether to allow weights to vary during fitting, by default False.

        Returns
        -------
        lmfit.Parameters
            The normalized weight parameters for the model.
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
                params.add(f'p{i+1}',value=np.log(weights[i]/last_weight),min=-14,max=14,vary=vary) # limit to 1ppm
            
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
        Set a new cross-section for the model.

        Parameters
        ----------
        xs : CrossSection
            The new cross-section to apply.
        inplace : bool, optional
            If True, modify the current object. If False, return a new modified object, 
            by default True.

        Returns
        -------
        TransmissionModel
            The updated model (either modified in place or a new instance).
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
        Update the parameters of the model.

        Parameters
        ----------
        params : dict
            Dictionary of new parameters to update.
        values_only : bool, optional
            If True, update only the values of the parameters, by default True.
        inplace : bool, optional
            If True, modify the current object. If False, return a new modified object, 
            by default True.
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
        Toggle the 'vary' attribute for all model parameters.

        Parameters
        ----------
        vary : bool, optional
            The value to set for all parameters' 'vary' attribute.
        except_for : list of str, optional
            List of parameter names to exclude from this operation, by default [].
        """
        if vary is not None:
            for param in self.params:
                if param not in except_for:
                    self.params[param].set(vary=vary)

    def _tof_correction(self, E, L0: float = 1.0, t0: float = 0.0,
                               t1: float = 0.0, t2: float = 0.0,   **kwargs):
        """
        Apply a time-of-flight (TOF) correction to the energy values.

        Parameters
        ----------
        E : float or array-like
            The energy values to correct.
        L0 : float, optional
            The scale factor for the flight path, by default 1.0.
        t0 : float, optional
            The time offset for the correction, by default 0.0.
        t1 : float, optional
            The linear correction factor, by default 0.0.
        t2 : float, optional
            The logarithmic correction factor, by default 0.0.
        kwargs : dict, optional
            Additional arguments (currently unused).

        Returns
        -------
        np.ndarray
            The corrected energy values.
        """
        tof = utils.energy2time(E, self.cross_section.L)
        dtof = (1.0 - L0) * tof + t0 + t1 * E + t2 * np.log(E)
        E = utils.time2energy(tof + dtof, self.cross_section.L)
        return E
    

    def manually_calibrate_tof(self,
                                inputs: Union[list, np.ndarray] = None,
                                references: Union[list, np.ndarray] = None,
                                input_type: str = 'tof',
                                reference_type: str = 'energy',
                                **kwargs):
        """
        Manually calibrate time-of-flight (TOF) correction parameters.

        Parameters
        ----------
        inputs : list or np.ndarray
            Input values for calibration (time-like values).
        references : list or np.ndarray
            Corresponding reference values for calibration (energy-like values).
        input_type : str, optional
            Type of input values. Options are:
            - 'tof': Direct time values in units of seconds
            - 'energy': Convert energy to time using utils.energy2time
            - 'slice': Convert slice indices to time by multiplying with tstep
            Default is 'tof'.
        reference_type : str, optional
            Type of reference values. Options are:
            - 'tof': Direct time values in units of seconds
            - 'energy': Convert energy to time using utils.energy2time
            - 'slice': Convert slice indices to time by multiplying with tstep
            Default is 'energy'.
        Returns
        -------
        lmfit ModelResult object
            Detailed linear regression result with fitting information
        """
        # Input validation
        if inputs is None or references is None:
            raise ValueError("Both inputs and references must be provided")
        
        # Convert inputs to numpy arrays
        inputs = np.array(inputs, dtype=float)
        references = np.array(references, dtype=float)
        
        # Validate input lengths
        if len(inputs) != len(references):
            raise ValueError("Input values and reference values must have the same length")
        
        # Convert input values based on input_type
        if input_type == 'energy':
            inputs = utils.energy2time(inputs, self.cross_section.L)
        elif input_type == 'slice':
            inputs = inputs * self.cross_section.tstep
        elif input_type != 'tof':
            raise ValueError("Invalid input_type. Must be 'tof', 'energy', or 'slice'")
        
        # Convert reference values based on input_type
        if reference_type == 'energy':
            references = utils.energy2time(references, self.cross_section.L)
        elif reference_type == 'slice':
            references = references * self.cross_section.tstep
        elif reference_type != 'tof':
            raise ValueError("Invalid reference_type. Must be 'tof', 'energy', or 'slice'")
        
        # Define the linear model using lmfit
        def linear_tof_correction(x, L0=1., t0=0.):
            return L0 * x + t0
        
        # Create the model
        model = lmfit.Model(linear_tof_correction)
        params = model.make_params()

        if len(inputs)==1:
            params["L0"].vary = False
        
        # Perform the fit
        result = model.fit(inputs, params=params,x=references)
        
        # Update self.params with the calibration results
        self.params.set(t0=dict(value=result.params['t0'].value, vary=False))
        self.params.set(L0=dict(value=result.params['L0'].value, vary=False))
        
        return result