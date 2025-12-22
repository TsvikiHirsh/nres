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
                        params: "lmfit.Parameters" = None,
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
        params : lmfit.Parameters, optional
            Initial parameter values from a previous fit. Only the parameter values
            will be updated; vary flags, bounds, and expressions remain as defined by
            the vary_* arguments. This is useful for using fit results as initial
            guesses for subsequent fits.
        kwargs : dict, optional
            Additional keyword arguments for model and background parameters.

        Notes
        -----
        This model calculates the transmission function as a combination of
        cross-section, response function, and background.

        Examples
        --------
        Using fit results as initial guesses for a new model:

        >>> # First fit
        >>> model1 = TransmissionModel(xs, vary_background=True, vary_tof=True)
        >>> result1 = model1.fit(data1)
        >>>
        >>> # Use result1 parameters as initial guesses for a new fit
        >>> model2 = TransmissionModel(xs, vary_background=True, params=result1.params)
        >>> result2 = model2.fit(data2)
        """
        # Extract params from kwargs if provided there (for backward compatibility)
        if params is None and 'params' in kwargs:
            params = kwargs.pop('params')

        super().__init__(self.transmission, **kwargs)

        self.cross_section = CrossSection()

        for material in cross_section.materials:
            self.cross_section += CrossSection(**{material:cross_section.materials[material]},
            splitby=cross_section.materials[material]["splitby"])

        self.params = self.make_params()

        # Add minimum bounds to basic parameters to prevent negative values
        if "thickness" in self.params:
            self.params["thickness"].set(min=0.)
        if "norm" in self.params:
            self.params["norm"].set(min=0.)
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

        # Initialize stages based on vary_* parameters
        self._stages = {}
        possible_stages = ["basic", "background", "tof", "response", "weights"]
        vary_flags = {
            "basic": True,  # Always include basic parameters
            "background": vary_background,
            "tof": vary_tof,
            "response": vary_response,
            "weights": vary_weights,
        }
        for stage in possible_stages:
            if vary_flags.get(stage, False) is True:
                self._stages[stage] = stage

        # Load parameter values from previous fit if provided
        # This only updates values, not vary flags, bounds, or expressions
        if params is not None:
            self._load_param_values(params)



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

        .. math:: T(E) = \\text{norm} \\cdot e^{- \\sigma \\cdot \\text{thickness} \\cdot n} \\cdot (1 - \\text{bg}) + \\text{bg}

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

    def _load_param_values(self, source_params: "lmfit.Parameters"):
        """
        Load parameter values from a source Parameters object.

        Only updates the values of existing parameters, preserving vary flags,
        bounds, and expressions as defined during model initialization.

        Parameters
        ----------
        source_params : lmfit.Parameters
            Source parameters (e.g., from a previous fit result) to load values from.

        Notes
        -----
        This method is called during __init__ if params argument is provided.
        It ensures that only parameters that exist in both source and target
        are updated, and only their values are changed.
        """
        for param_name in self.params:
            if param_name in source_params:
                # Only update the value, preserve vary, min, max, expr
                self.params[param_name].value = source_params[param_name].value

    @property
    def stages(self):
        """Get the current fitting stages."""
        return self._stages

    @stages.setter
    def stages(self, value):
        """
        Set the fitting stages.

        Parameters
        ----------
        value : str or dict
            If str, must be "all" to use all vary=True parameters.
            If dict, keys are stage names, values are stage definitions
            ("all", a valid group name, or a list of parameters/groups).
        """
        import re

        # Define valid group names from group_map
        group_map = {
            "basic": ["norm", "thickness"],
            "background": [p for p in self.params if re.compile(r"(b|bg)\d+").match(p) or p.startswith("b_")],
            "tof": [p for p in ["L0", "t0", "t1", "t2"] if p in self.params],
            "response": [p for p in self.params if self.response and p in self.response.params],
            "weights": [p for p in self.params if re.compile(r"p\d+").match(p)],
        }

        if isinstance(value, str):
            if value != "all":
                raise ValueError("If stages is a string, it must be 'all'")
            self._stages = {"all": "all"}
        elif isinstance(value, dict):
            # Validate stage definitions
            for stage_name, stage_def in value.items():
                if not isinstance(stage_name, str):
                    raise ValueError(f"Stage names must be strings, got {type(stage_name)}")
                if isinstance(stage_def, str):
                    if stage_def != "all" and stage_def not in group_map:
                        raise ValueError(f"Stage definition for '{stage_name}' must be 'all' or a valid group name, got '{stage_def}'")
                elif isinstance(stage_def, list):
                    for param in stage_def:
                        if not isinstance(param, str):
                            raise ValueError(f"Parameters in stage '{stage_name}' must be strings, got {type(param)}")
                else:
                    raise ValueError(f"Stage definition for '{stage_name}' must be 'all', a valid group name, or a list, got {type(stage_def)}")
            self._stages = value
        else:
            raise ValueError(f"Stages must be a string ('all') or dict, got {type(value)}")

    def fit(self, data, params=None, emin: float = 0.5e6, emax: float = 20.e6,
            method: str = "rietveld",
            xtol: float = None, ftol: float = None, gtol: float = None,
            verbose: bool = False,
            progress_bar: bool = True,
            param_groups: Optional[List[List[str]]] = None,
            **kwargs):
        """
        Fit the model to data.

        This method supports both:
        - **Standard single-stage fitting** (default)
        - **Rietveld-style staged refinement** (`method="rietveld"`) with accumulative parameter refinement with accumulative parameter refinement

        Parameters
        ----------
        data : pandas.DataFrame or Data or array-like
            The input data.  
            - For `pandas.DataFrame` or `Data`: must have columns `"energy"`, `"trans"`, and `"err"`.
            - For array-like: will be passed directly to `lmfit.Model.fit`.
        params : lmfit.Parameters, optional
            Parameters to use for fitting. If None, uses the model's default parameters.
        emin, emax : float, optional
            Minimum and maximum energy for fitting (ignored for array-like input and overridden per stage if
            `param_groups` specify `"emin=..."` or `"emax=..."` strings).
        method : str, optional
            Fitting method.
            - `"rietveld"` (default) will run staged refinement via `_rietveld_fit`.
            - `"least-squares"` or any method supported by `lmfit` for single-stage fitting.
        xtol, ftol, gtol : float, optional
            Convergence tolerances (passed to `lmfit`).
        verbose : bool, optional
            If True, prints detailed fitting information.
        progress_bar : bool, optional
            If True, shows a progress bar for fitting:
            - For `"rietveld"`: shows stage name, energy range, and reduced chi² per stage.
            - For regular fits: shows overall fit progress.
        param_groups : list, dict, or None, optional
            Used only for `"rietveld"`. Groups of parameters to fit in each stage.
            Groups may also contain `"emin=..."` and/or `"emax=..."` strings to override the energy
            fitting range for that specific stage. For example:

            ```python
            param_groups = {
                "Basic": ["basic"],
                "Background": ["background", "emin=3", "emax=8"],
                "Extinction": ["extinction"]
            }
            ```

            These per-stage overrides temporarily replace the global `emin`/`emax` only during the stage.
        **kwargs
            Additional keyword arguments passed to `lmfit.Model.fit`.

            For grouped data, additional parameters:
            - `n_jobs` (int): Number of parallel jobs (default: 10). Use -1 for all CPUs,
              but beware of memory issues. For threading, consider n_jobs=4 or less.
            - `max_nbytes` (str): Maximum memory per worker (default: '100M'). Prevents
              memory exhaustion. Increase for complex models or set to None to disable.

        Returns
        -------
        lmfit.model.ModelResult
            The fit result object, with extra methods:
            - `.plot()` — plot the fit result.
            - `.plot_stage_progression()`, `.plot_chi2_progression()` for advanced diagnostics.
            - `.stages_summary` (for `"rietveld"`).

        Examples
        --------
        **Basic fit:**
        ```python
        result = model.fit(data_df, emin=1.0, emax=5.0)
        result.plot()
        ```

        **Rietveld-style staged refinement with per-stage energy overrides:**
        ```python
        param_groups = {
            "Norm/Thick": ["norm", "thickness"],
            "Background": ["b0", "b1", "emin=3", "emax=8"],
            "Extinction": ["ext_l", "ext_Gg"]
        }
        result = model.fit(
            data_df, method="rietveld",
            param_groups=param_groups,
            progress_bar=True
        )
        print(result.stages_summary)
        ```
        """
        # Check if data is grouped and route to parallel fitting
        if hasattr(data, 'is_grouped') and data.is_grouped:
            n_jobs = kwargs.pop('n_jobs', 10)
            max_nbytes = kwargs.pop('max_nbytes', '100M')
            return self._fit_grouped(
                data, params, emin, emax,
                method=method,
                xtol=xtol, ftol=ftol, gtol=gtol,
                verbose=verbose,
                progress_bar=progress_bar,
                param_groups=param_groups,
                n_jobs=n_jobs,
                max_nbytes=max_nbytes,
                **kwargs
            )

        # Route to Rietveld if requested
        if method == "rietveld":
            return self._rietveld_fit(
                data, params, emin, emax,
                verbose=verbose,
                progress_bar=progress_bar,
                param_groups=param_groups,
                **kwargs
            )

        # Prepare fit kwargs
        fit_kws = kwargs.pop("fit_kws", {})
        if xtol is not None: fit_kws.setdefault("xtol", xtol)
        if ftol is not None: fit_kws.setdefault("ftol", ftol)
        if gtol is not None: fit_kws.setdefault("gtol", gtol)
        kwargs["fit_kws"] = fit_kws

        # Try tqdm for progress
        try:
            from tqdm.notebook import tqdm
        except ImportError:
            from tqdm.auto import tqdm

        # If progress_bar=True, wrap the fit in tqdm
        if progress_bar:
            pbar = tqdm(total=1, desc="Fitting", disable=not progress_bar)
        else:
            pbar = None

        # Prepare input data
        if isinstance(data, pandas.DataFrame):
            data = data.query(f"{emin} < energy < {emax}")
            weights = kwargs.get("weights", 1. / data["err"].values)
            fit_result = super().fit(
                data["trans"].values,
                params=params or self.params,
                weights=weights,
                E=data["energy"].values,
                method=method,
                **kwargs
            )

        elif isinstance(data, Data):
            data = data.table.query(f"{emin} < energy < {emax}")
            weights = kwargs.get("weights", 1. / data["err"].values)
            fit_result = super().fit(
                data["trans"].values,
                params=params or self.params,
                weights=weights,
                E=data["energy"].values,
                method=method,
                **kwargs
            )

        else:
            fit_result = super().fit(
                data,
                params=params or self.params,
                method=method,
                **kwargs
            )

        if pbar:
            pbar.set_postfix({"redchi": f"{fit_result.redchi:.4g}"})
            pbar.update(1)
            pbar.close()

        # Attach results
        self.fit_result = fit_result
        fit_result.plot = self.plot
        fit_result.show_available_params = self.show_available_params
        fit_result.save = lambda filename, include_model=True: self._save_result(fit_result, filename, include_model)
        fit_result.save = lambda filename, include_model=True: self._save_result(fit_result, filename, include_model)

        if self.response is not None:
            fit_result.response = self.response
            fit_result.response.params = fit_result.params
        if self.background is not None:
            fit_result.background = self.background

        return fit_result

    def _rietveld_fit(self, data, params: "lmfit.Parameters" = None, emin: float = 0.5e6, emax: float = 20.e6,
                    verbose=False, progress_bar=True,
                    param_groups=None,
                    **kwargs):
        """ Perform Rietveld-style staged fitting with accumulative parameter refinement.

        In this method, parameters accumulate across stages. When a new stage is added,
        all previously refined parameters remain vary=True, allowing for simultaneous
        refinement of all parameters introduced up to that stage.
        
        Parameters
        ----------
        data : pandas.DataFrame or Data
            The input data containing energy and transmission values.
        params : lmfit.Parameters, optional
            Initial parameters for the fit. If None, uses the model's default parameters.
        emin : float, optional default=0.5e6
            Default minimum energy for fitting.
        emax : float, optional default=20.e6
            Default maximum energy for fitting.
        verbose : bool, optional
            If True, prints detailed information about each fitting stage.
        progress_bar : bool, optional
            If True, shows a progress bar for each fitting stage.
        param_groups : list, dict, or None, optional - only used for Rietveld fitting
            Groups of parameters to fit in each stage. Can contain special keywords:
            - "emin=<value>" or "emax=<value>": override energy bounds for that stage
            - "pick-one" or "pick_one": enable pick-one mode for isotope selection. In this mode,
              the fit tries each cross-section material individually with weight=1 (others at 0),
              then selects the isotope with the best fit quality (lowest reduced chi-squared).
              The selected isotope is fixed at weight=1 with all others at weight=0.
        kwargs : dict, optional
            Additional keyword arguments for the fit method, such as weights, method, etc.

        Returns
        -------
        fit_result : lmfit.ModelResult
            The final fit result after all stages.

        fit_result.stages_summary : pandas.DataFrame
            Summary of each fitting stage, including parameter values and reduced chi-squared.
        """

        from copy import deepcopy
        import sys
        import warnings
        import re
        import fnmatch
        import pandas
        try:
            from tqdm.notebook import tqdm
        except ImportError:
            from tqdm.auto import tqdm
        import pickle

        # Use original params to determine which were set to vary
        original_params = params or self.params

        # User-friendly group name mapping - only include parameters that have vary=True
        group_map = {
            "basic": [p for p in ["norm", "thickness"] if p in original_params and original_params[p].vary],
            "background": [p for p in original_params if (re.compile(r"(b|bg)\d+").match(p) or p.startswith("b_")) and original_params[p].vary],
            "tof": [p for p in ["L0", "t0", "t1", "t2"] if p in original_params and original_params[p].vary],
            "response": [p for p in original_params if self.response and p in self.response.params and original_params[p].vary],
            "weights": [p for p in original_params if re.compile(r"p\d+").match(p) and original_params[p].vary],
        }

        def resolve_single_param_or_group(item):
            """Resolve a single parameter name or group name to a list of parameters."""
            if item in group_map:
                resolved = group_map[item]
                if verbose:
                    print(f"  Resolved group '{item}' to: {resolved}")
                return resolved
            elif item in self.params:
                if verbose:
                    print(f"  Found parameter: {item}")
                return [item]
            else:
                matching_params = [p for p in self.params.keys() if fnmatch.fnmatch(p, item)]
                if matching_params:
                    if verbose:
                        print(f"  Pattern '{item}' matched: {matching_params}")
                    return matching_params
                else:
                    warnings.warn(f"Unknown parameter or group: '{item}'. Available parameters: {list(self.params.keys())}")
                    return []

        def resolve_group(entry):
            """
            Resolve a group entry (string, list, or nested structure) to:
            - A flat list of parameters
            - A dict of overrides like {'emin': float, 'emax': float}
            """
            params_list = []
            overrides = {}

            def process_item(item):
                nonlocal params_list, overrides
                if isinstance(item, str):
                    if item.startswith("emin="):
                        try:
                            overrides['emin'] = float(item.split("=", 1)[1])
                            if verbose:
                                print(f"  Override emin detected: {overrides['emin']}")
                        except ValueError:
                            warnings.warn(f"Invalid emin value in group: {item}")
                    elif item.startswith("emax="):
                        try:
                            overrides['emax'] = float(item.split("=", 1)[1])
                            if verbose:
                                print(f"  Override emax detected: {overrides['emax']}")
                        except ValueError:
                            warnings.warn(f"Invalid emax value in group: {item}")
                    elif item == "pick-one" or item == "pick_one":
                        overrides['pick_one'] = True
                        if verbose:
                            print(f"  Pick-one mode detected: will test each isotope individually")
                    else:
                        params_list.extend(resolve_single_param_or_group(item))
                elif isinstance(item, list):
                    for subitem in item:
                        process_item(subitem)
                else:
                    warnings.warn(f"Unexpected item type in group: {type(item)} - {item}")

            process_item(entry)
            return params_list, overrides

        # Handle different input formats for param_groups and parse overrides
        stage_names = []
        resolved_param_groups = []
        stage_overrides = []

        if param_groups is None:
            # Default groups
            default_groups = [
                "basic", "background", "tof", "response",
                "weights",
            ]
            for group in default_groups:
                params_list, overrides = resolve_group(group)
                if params_list:
                    resolved_param_groups.append(params_list)
                    stage_overrides.append(overrides)
                    stage_names.append(f"Stage_{len(stage_names) + 1}")
                elif verbose:
                    print(f"Skipping empty default group: {group}")

        elif isinstance(param_groups, dict):
            stage_names = list(param_groups.keys())
            for stage in stage_names:
                params_list, overrides = resolve_group(param_groups[stage])
                if params_list:
                    resolved_param_groups.append(params_list)
                    stage_overrides.append(overrides)
                else:
                    if verbose:
                        print(f"Skipping empty group: {stage}")

        elif isinstance(param_groups, list):
            for i, group in enumerate(param_groups):
                params_list, overrides = resolve_group(group)
                if params_list:
                    resolved_param_groups.append(params_list)
                    stage_overrides.append(overrides)
                    stage_names.append(f"Stage_{i + 1}")
                else:
                    if verbose:
                        print(f"Skipping empty group at index {i}")

        else:
            raise ValueError("param_groups must be None, a list, or a dictionary")

        # Remove any empty groups that slipped through
        filtered = [(n, g, o) for n, g, o in zip(stage_names, resolved_param_groups, stage_overrides) if g]
        if not filtered:
            raise ValueError("No valid parameter groups found. Check your parameter names.")
        stage_names, resolved_param_groups, stage_overrides = zip(*filtered)

        if verbose:
            print(f"\nFitting stages with possible energy overrides:")
            for i, (name, group, ov) in enumerate(zip(stage_names, resolved_param_groups, stage_overrides)):
                print(f"  {name}: {group}  overrides: {ov}")

        # Store for summary or introspection
        self._stage_param_groups = resolved_param_groups
        self._stage_names = stage_names

        params = deepcopy(params or self.params)

        # Setup tqdm iterator
        try:
            from tqdm.notebook import tqdm as notebook_tqdm
            if 'ipykernel' in sys.modules:
                iterator = notebook_tqdm(
                    zip(stage_names, resolved_param_groups, stage_overrides),
                    desc="Rietveld Fit",
                    disable=not progress_bar,
                    total=len(stage_names)
                )
            else:
                iterator = tqdm(
                    zip(stage_names, resolved_param_groups, stage_overrides),
                    desc="Rietveld Fit",
                    disable=not progress_bar,
                    total=len(stage_names)
                )
        except ImportError:
            iterator = tqdm(
                zip(stage_names, resolved_param_groups, stage_overrides),
                desc="Rietveld Fit",
                disable=not progress_bar,
                total=len(stage_names)
            )

        stage_results = []
        stage_summaries = []
        # Lists to collect final stages (including pick-one isotope tests)
        final_stage_results = []
        final_stage_names = []
        final_resolved_param_groups = []
        cumulative_params = set()  # Track parameters that have been refined (accumulative Rietveld)

        def extract_pickleable_attributes(fit_result):
            safe_attrs = [
                'params', 'success', 'residual', 'chisqr', 'redchi', 'aic', 'bic',
                'nvarys', 'ndata', 'nfev', 'message', 'lmdif_message', 'cov_x',
                'method', 'flatchain', 'errorbars', 'ci_out'
            ]

            class PickleableResult:
                pass

            result = PickleableResult()

            for attr in safe_attrs:
                if hasattr(fit_result, attr):
                    try:
                        value = getattr(fit_result, attr)
                        pickle.dumps(value)
                        setattr(result, attr, value)
                    except (TypeError, ValueError, AttributeError):
                        if verbose:
                            print(f"Skipping non-pickleable attribute: {attr}")
                        continue

            return result

        for stage_idx, (stage_name, group, overrides) in enumerate(iterator):
            stage_num = stage_idx + 1

            # Use overrides or fallback to global emin, emax
            stage_emin = overrides.get('emin', emin)
            stage_emax = overrides.get('emax', emax)

            if verbose:
                print(f"\n{stage_name}: Fitting parameters {group} with energy range [{stage_emin}, {stage_emax}]")

            # Filter data for this stage
            if isinstance(data, pandas.DataFrame):
                stage_data = data.query(f"{stage_emin} < energy < {stage_emax}")
                energies = stage_data["energy"].values
                trans = stage_data["trans"].values
                weights = kwargs.get("weights", 1. / stage_data["err"].values)
            elif isinstance(data, Data):
                stage_data = data.table.query(f"{stage_emin} < energy < {stage_emax}")
                energies = stage_data["energy"].values
                trans = stage_data["trans"].values
                weights = kwargs.get("weights", 1. / stage_data["err"].values)
            else:
                raise ValueError("Rietveld fitting requires energy-based input data.")

            # Check if pick-one mode is enabled for this stage
            if overrides.get('pick_one', False):
                if verbose:
                    print(f"\n  Pick-one mode: Testing each isotope individually...")

                # Get isotope names from cross-section weights
                isotope_names = list(self.cross_section.weights.index)
                isotope_names = [name.replace("-", "") for name in isotope_names]

                # Get the p parameters (free weight parameters)
                p_params = [p for p in params.keys() if re.compile(r"p\d+").match(p)]

                if len(isotope_names) <= 1:
                    warnings.warn(f"Pick-one mode requires at least 2 isotopes, but found {len(isotope_names)}. Skipping pick-one.")
                elif not p_params:
                    warnings.warn(f"Pick-one mode requires weight parameters (p1, p2, ...), but none found. Skipping pick-one.")
                else:
                    # Store results for each isotope test
                    isotope_results = []

                    # Test each isotope
                    for iso_idx, isotope_name in enumerate(isotope_names):
                        if verbose:
                            print(f"    Testing {isotope_name}...")

                        # Create a copy of params for this test
                        test_params = deepcopy(params)

                        # Set this isotope to weight=1, others to weight=0
                        # For isotope i (i < N-1): set p_i = 14 (max), others = -14 (min)
                        # For isotope N-1 (last): set all p_j = -14 (min)
                        for j, p_name in enumerate(p_params):
                            if iso_idx < len(p_params):
                                # One of the first N-1 isotopes
                                if j == iso_idx:
                                    test_params[p_name].value = 14.0  # This isotope dominates
                                else:
                                    test_params[p_name].value = -14.0  # Others suppressed
                            else:
                                # Last isotope (N-1)
                                test_params[p_name].value = -14.0  # All p's minimal -> last weight = 1

                        # Set all parameters to not vary (fixed for this test)
                        for p in test_params.values():
                            p.vary = False

                        # Only vary the parameters specified in the stage (excluding weights)
                        non_weight_params = [p for p in group if p not in p_params and not re.compile(r"p\d+").match(p)]
                        for param_name in non_weight_params:
                            if param_name in test_params:
                                test_params[param_name].vary = True

                        # Perform test fit
                        # Filter out kwargs that lmfit doesn't understand
                        lmfit_kwargs = {k: v for k, v in kwargs.items()
                                       if k not in ['n_cores', 'n_jobs', 'max_nbytes', 'progress_bar']}
                        try:
                            test_fit = super().fit(
                                trans,
                                params=test_params,
                                E=energies,
                                weights=weights,
                                method="leastsq",
                                **lmfit_kwargs
                            )

                            isotope_results.append({
                                'isotope': isotope_name,
                                'index': iso_idx,
                                'redchi': test_fit.redchi,
                                'params': test_fit.params
                            })

                            if verbose:
                                print(f"      {isotope_name}: χ²/dof = {test_fit.redchi:.4f}")

                            # Add this isotope test as a separate stage in the final results
                            final_stage_results.append(test_fit)
                            final_resolved_param_groups.append(non_weight_params)
                            final_stage_names.append(f"{stage_name} (test: {isotope_name})")

                        except Exception as e:
                            warnings.warn(f"Fitting failed for {isotope_name}: {e}")
                            isotope_results.append({
                                'isotope': isotope_name,
                                'index': iso_idx,
                                'redchi': float('inf'),
                                'params': None
                            })

                    # Find the best isotope (lowest reduced chi-squared)
                    best_result = min(isotope_results, key=lambda x: x['redchi'])
                    best_isotope = best_result['isotope']
                    best_idx = best_result['index']

                    if verbose:
                        print(f"\n  Best fit: {best_isotope} with χ²/dof = {best_result['redchi']:.4f}")

                    # Update progress bar
                    iterator.set_postfix({"stage": stage_name, "best": best_isotope, "reduced χ²": f"{best_result['redchi']:.4g}"})

                    # Set the weights to fix the best isotope at weight=1
                    if best_idx < len(p_params):
                        # One of the first N-1 isotopes
                        for j, p_name in enumerate(p_params):
                            if j == best_idx:
                                params[p_name].value = 14.0
                            else:
                                params[p_name].value = -14.0
                    else:
                        # Last isotope
                        for p_name in p_params:
                            params[p_name].value = -14.0

                    # Copy other fitted parameters from the best result
                    if best_result['params'] is not None:
                        non_weight_params = [p for p in group if p not in p_params and not re.compile(r"p\d+").match(p)]
                        for param_name in non_weight_params:
                            if param_name in params and param_name in best_result['params']:
                                params[param_name].value = best_result['params'][param_name].value

                    # The weights are now fixed, so we continue to the next stage
                    # Add the stage to cumulative params (the non-weight params were fitted)
                    cumulative_params.update([p for p in group if not re.compile(r"p\d+").match(p)])

                    # Create a fake fit_result for consistency
                    class PickOneFitResult:
                        def __init__(self, params, redchi):
                            self.params = params
                            self.redchi = redchi
                            self.success = True
                            self.residual = None
                            self.chisqr = None
                            self.aic = None
                            self.bic = None
                            self.nvarys = 0
                            self.ndata = len(energies)
                            self.nfev = 0
                            self.message = f"Pick-one mode: selected {best_isotope}"
                            self.lmdif_message = self.message
                            self.cov_x = None
                            self.method = "pick-one"
                            self.flatchain = None
                            self.errorbars = False
                            self.ci_out = None

                    iterator.set_description(f"Stage {stage_num}/{len(stage_names)}")

                    if verbose:
                        print(f"  {stage_name} completed with pick-one. Selected {best_isotope}, χ²/dof = {best_result['redchi']:.4f}")

                    # Skip the normal fitting for this stage (isotope tests already added to final lists)
                    continue

            # Accumulate parameters across stages (True Rietveld approach)
            cumulative_params.update(group)

            # Freeze all parameters
            for p in params.values():
                p.vary = False

            # Unfreeze current group
            # Note: group_map already filters out parameters with vary=False
            # Unfreeze all parameters that have been introduced so far
            unfrozen_count = 0
            for name in cumulative_params:
                if name in params:
                    params[name].vary = True
                    unfrozen_count += 1
                    if verbose and name in group:
                        print(f"  New parameter: {name}")
                    elif verbose:
                        print(f"  Continuing: {name}")
                else:
                    if name in group:  # Only warn for new parameters
                        warnings.warn(f"Parameter '{name}' not found in params")

            if verbose:
                print(f"  Total active parameters: {unfrozen_count}")

            if unfrozen_count == 0:
                warnings.warn(f"No parameters were unfrozen in {stage_name}. Skipping this stage.")
                continue

            # Perform fitting
            # Filter out kwargs that lmfit doesn't understand
            lmfit_kwargs = {k: v for k, v in kwargs.items()
                           if k not in ['n_cores', 'n_jobs', 'max_nbytes', 'progress_bar']}
            try:
                with warnings.catch_warnings():
                    if not verbose:
                        # Suppress lmfit warnings when not verbose
                        warnings.filterwarnings('ignore', category=UserWarning, module='lmfit')

                    fit_result = super().fit(
                        trans,
                        params=params,
                        E=energies,
                        weights=weights,
                        method="leastsq",
                        **lmfit_kwargs
                    )
            except Exception as e:
                if verbose:
                    warnings.warn(f"Fitting failed in {stage_name}: {e}")
                continue

            # Extract pickleable part
            stripped_result = extract_pickleable_attributes(fit_result)

            stage_results.append(stripped_result)
            # Also add to final results (for stages_summary)
            final_stage_results.append(stripped_result)
            final_stage_names.append(stage_name)
            final_resolved_param_groups.append(group)

            # Build summary
            varied_params = list(cumulative_params)  # Track cumulative parameters
            varied_params = list(cumulative_params)  # Track cumulative parameters
            summary = {
                "stage": stage_num,
                "stage_name": stage_name,
                "fitted_params": group,
                "emin": stage_emin,
                "emax": stage_emax,
                "redchi": fit_result.redchi
            }
            for name, par in fit_result.params.items():
                summary[f"{name}_value"] = par.value
                summary[f"{name}_stderr"] = par.stderr
                summary[f"{name}_vary"] = name in varied_params  # Mark as vary if in cumulative set
            stage_summaries.append(summary)

            iterator.set_description(f"Stage {stage_num}/{len(stage_names)}")
            iterator.set_postfix({"stage": stage_name, "reduced χ²": f"{fit_result.redchi:.4g}"})

            # Update params for next stage
            params = fit_result.params

            if verbose:
                print(f"  {stage_name} completed. χ²/dof = {fit_result.redchi:.4f}")

        if not stage_results:
            raise RuntimeError("No successful fitting stages completed")

        self.fit_result = fit_result
        self.fit_stages = stage_results
        # Use final lists (which include pick-one isotope tests) for stages_summary
        self.stages_summary = self._create_stages_summary_table_enhanced(
            final_stage_results if final_stage_results else stage_results,
            final_resolved_param_groups if final_resolved_param_groups else resolved_param_groups,
            final_stage_names if final_stage_names else stage_names
        )

        # Attach plotting methods and other attributes
        fit_result.plot = self.plot
        fit_result.plot_stage_progression = self.plot_stage_progression
        fit_result.plot_chi2_progression = self.plot_chi2_progression
        if self.response is not None:
            fit_result.response = self.response
            fit_result.response.params = fit_result.params
        if self.background is not None:
            fit_result.background = self.background

        fit_result.stages_summary = self.stages_summary
        fit_result.show_available_params = self.show_available_params
        fit_result.save = lambda filename, include_model=True: self._save_result(fit_result, filename, include_model)
        return fit_result


    def _create_stages_summary_table_enhanced(self, stage_results, resolved_param_groups, stage_names=None, color=True):
        import pandas as pd
        import numpy as np

        # --- Build the DataFrame ---
        all_param_names = list(stage_results[-1].params.keys())
        stage_data = {}
        if stage_names is None:
            stage_names = [f"Stage_{i+1}" for i in range(len(stage_results))]

        cumulative_params = set()  # Track cumulative parameters for Rietveld method

        for stage_idx, stage_result in enumerate(stage_results):
            stage_col = stage_names[stage_idx] if stage_idx < len(stage_names) else f"Stage_{stage_idx + 1}"
            stage_data[stage_col] = {'value': {}, 'stderr': {}, 'vary': {}}

            # Accumulate parameters across stages
            cumulative_params.update(resolved_param_groups[stage_idx])
            varied_in_stage = cumulative_params.copy()

            for param_name in all_param_names:
                if param_name in stage_result.params:
                    param = stage_result.params[param_name]
                    stage_data[stage_col]['value'][param_name] = param.value
                    stage_data[stage_col]['stderr'][param_name] = param.stderr if param.stderr is not None else np.nan
                    stage_data[stage_col]['vary'][param_name] = param_name in varied_in_stage
                else:
                    stage_data[stage_col]['value'][param_name] = np.nan
                    stage_data[stage_col]['stderr'][param_name] = np.nan
                    stage_data[stage_col]['vary'][param_name] = False

            redchi = stage_result.redchi if hasattr(stage_result, 'redchi') else np.nan
            stage_data[stage_col]['value']['redchi'] = redchi
            stage_data[stage_col]['stderr']['redchi'] = np.nan
            stage_data[stage_col]['vary']['redchi'] = np.nan

        # Create DataFrame
        data_for_df = {}
        for stage_col in stage_data:
            for metric in ['value', 'stderr', 'vary']:
                data_for_df[(stage_col, metric)] = stage_data[stage_col][metric]

        df = pd.DataFrame(data_for_df)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Stage', 'Metric'])
        all_param_names_with_redchi = all_param_names + ['redchi']
        df = df.reindex(all_param_names_with_redchi)

        # --- Add initial values column ---
        initial_values = {}
        for param_name in all_param_names:
            initial_values[param_name] = self.params[param_name].value if param_name in self.params else np.nan
        initial_values['redchi'] = np.nan

        initial_df = pd.DataFrame({('Initial', 'value'): initial_values})
        df = pd.concat([initial_df, df], axis=1)

        if not color:
            return df

        styler = df.style

        # 1) Highlight vary=True cells (light green for accumulative Rietveld)
        vary_cols = [col for col in df.columns if col[1] == 'vary']
        def highlight_vary(s):
            return ['background-color: lightgreen' if v is True else '' for v in s]
        for col in vary_cols:
            styler = styler.apply(highlight_vary, subset=[col], axis=0)

        # 2) Highlight redchi row's value cells (moccasin)
        def highlight_redchi_row(row):
            if row.name == 'redchi':
                return ['background-color: moccasin' if col[1] == 'value' else '' for col in df.columns]
            return ['' for _ in df.columns]
        styler = styler.apply(highlight_redchi_row, axis=1)

        # 3) Highlight value cells by fractional change with red hues (ignore <1%)
        value_cols = [col for col in df.columns if col[1] == 'value']

        # Calculate % absolute change between consecutive columns (Initial → Stage1 → Stage2 ...)
        changes = pd.DataFrame(index=df.index, columns=value_cols, dtype=float)
        prev_col = None
        for col in value_cols:
            if prev_col is None:
                # No previous for initial column, so zero changes here
                changes[col] = 0.0
            else:
                prev_vals = df[prev_col].astype(float)
                curr_vals = df[col].astype(float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    pct_change = np.abs((curr_vals - prev_vals) / prev_vals) * 100
                pct_change = pct_change.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                changes[col] = pct_change
            prev_col = col

        max_change = changes.max().max()
        # Normalize by max change, to get values in [0,1]
        norm_changes = changes / max_change if max_change > 0 else changes

        def red_color(val):
            # Ignore changes less than 1%
            if pd.isna(val) or val < 1:
                return ''
            # val in [0,1], map to red intensity
            # 0 -> white (255,255,255)
            # 1 -> dark red (255,100,100)
            r = 255
            g = int(255 - 155 * val)
            b = int(255 - 155 * val)
            return f'background-color: rgb({r},{g},{b})'

        for col in value_cols:
            styler = styler.apply(lambda s: [red_color(v) for v in norm_changes[col]], subset=[col], axis=0)

        return styler

    def _fit_grouped(self, data, params=None, emin: float = 0.5e6, emax: float = 20.e6,
                     method: str = "rietveld",
                     xtol: float = None, ftol: float = None, gtol: float = None,
                     verbose: bool = False,
                     progress_bar: bool = True,
                     param_groups: Optional[List[List[str]]] = None,
                     n_jobs: int = 10,
                     max_nbytes: str = '100M',
                     **kwargs):
        """
        Fit model to grouped data in parallel.

        Parameters:
        -----------
        data : Data
            Grouped data object with is_grouped=True.
        params : lmfit.Parameters, optional
            Parameters to use for fitting.
        emin, emax : float
            Energy range for fitting.
        method : str
            Fitting method: "least-squares" or "rietveld".
        xtol, ftol, gtol : float, optional
            Convergence tolerances.
        verbose : bool
            Show progress for individual fits.
        progress_bar : bool
            Show overall progress bar.
        param_groups : list or dict, optional
            Fitting stages configuration for rietveld.
        n_jobs : int
            Number of parallel jobs (default: 10). Use -1 for all CPUs, but be aware
            this can cause memory issues with large datasets. For threading backend,
            consider n_jobs=4 or less for better performance.
        max_nbytes : str
            Maximum memory per worker (default: '100M'). Limits memory usage to prevent
            system freezes. Increase (e.g., '500M') for complex models, or set to None
            to disable memory limits.
        **kwargs
            Additional arguments passed to fit.

        Returns:
        --------
        GroupedFitResult
            Container with fit results for each group.
        """
        from joblib import Parallel, delayed
        from nres.grouped_fit import GroupedFitResult
        import time

        try:
            from tqdm.auto import tqdm
        except ImportError:
            from tqdm import tqdm

        # Prepare fit arguments
        fit_kwargs = {
            'params': params,
            'emin': emin,
            'emax': emax,
            'method': method,
            'xtol': xtol,
            'ftol': ftol,
            'gtol': gtol,
            'verbose': verbose if verbose else False,
            'progress_bar': verbose,  # Show individual progress bars when verbose=True
            'param_groups': param_groups,
            **kwargs
        }

        def fit_single_group(idx):
            """Fit a single group using threading."""
            from nres.data import Data
            group_data = Data()
            group_data.table = data.groups[idx]
            group_data.L = data.L
            group_data.tstep = data.tstep

            try:
                result = self.fit(group_data, **fit_kwargs)
            except Exception as e:
                if verbose:
                    print(f"Error fitting group {idx}: {e}")
                result = None
            return idx, result

        start_time = time.time()

        # Execute with threading (or multiprocessing if n_jobs != 1)
        backend = 'threading' if n_jobs > 0 else 'loky'

        # Warn about performance with high n_jobs in threading mode
        if backend == 'threading' and n_jobs > 4 and verbose:
            print(f"Warning: Using {n_jobs} threads. Consider n_jobs=4 or less for better performance.")
            print(f"         High thread counts can cause memory issues. Current limit: {max_nbytes} per worker.")

        # Execute parallel fitting with proper progress bar
        if progress_bar:
            import sys
            pbar = tqdm(
                total=len(data.indices),
                desc=f"Fitting {len(data.indices)} groups",
                mininterval=0.05,  # Update display every 50ms minimum
                maxinterval=1.0,   # Force update at least every second
                smoothing=0.05,    # Less smoothing for more responsive updates
                file=sys.stderr,   # Write to stderr (unbuffered)
                dynamic_ncols=True,  # Adjust to terminal width
                leave=True         # Keep the bar after completion
            )
            results = []
            for result in Parallel(
                n_jobs=n_jobs,
                backend=backend,
                verbose=5 if verbose else 0,
                return_as='generator',
                max_nbytes=max_nbytes
            )(delayed(fit_single_group)(idx) for idx in data.indices):
                results.append(result)
                pbar.update(1)
                # Force immediate display update
                pbar.refresh()
                sys.stderr.flush()
            pbar.close()
        else:
            results = Parallel(
                n_jobs=n_jobs,
                backend=backend,
                verbose=5 if verbose else 0,
                max_nbytes=max_nbytes
            )(delayed(fit_single_group)(idx) for idx in data.indices)

        elapsed = time.time() - start_time
        if verbose:
            print(f"Completed in {elapsed:.2f}s using '{backend}' backend | {elapsed/len(data.indices):.3f}s per fit")

        # Collect results
        grouped_result = GroupedFitResult(group_shape=data.group_shape)
        failed_indices = []
        for idx, result in results:
            if result is not None:
                grouped_result.add_result(idx, result)
            else:
                failed_indices.append(idx)

        if failed_indices and verbose:
            import warnings
            warnings.warn(f"Fitting failed for {len(failed_indices)}/{len(data.indices)} groups. "
                         f"Failed indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")

        return grouped_result

    def show_available_params(self, show_groups=True, show_params=True):
        """
        Display available parameter groups and individual parameters for Rietveld fitting.

        Parameters
        ----------
        show_groups : bool, optional
            If True, show predefined parameter groups
        show_params : bool, optional
            If True, show all individual parameters
        """
        import re

        if show_groups:
            print("Available parameter groups:")
            print("=" * 30)

            # Only show parameters that have vary=True
            group_map = {
                "basic": [p for p in ["norm", "thickness"] if p in self.params and self.params[p].vary],
                "background": [p for p in self.params if (re.compile(r"(b|bg)\d+").match(p) or p.startswith("b_")) and self.params[p].vary],
                "tof": [p for p in ["L0", "t0", "t1", "t2"] if p in self.params and self.params[p].vary],
                "response": [p for p in self.params if self.response and p in self.response.params and self.params[p].vary],
                "weights": [p for p in self.params if re.compile(r"p\d+").match(p) and self.params[p].vary],
            }
            
            for group_name, params in group_map.items():
                if params:  # Only show groups with available parameters
                    print(f"  '{group_name}': {params}")
            
        if show_params:
            if show_groups:
                print("\nAll individual parameters:")
                print("=" * 30)
            else:
                print("Available parameters:")
                print("=" * 20)
                
            for param_name, param in self.params.items():
                vary_status = "vary" if param.vary else "fixed"
                print(f"  {param_name}: {param.value:.6g} ({vary_status})")
                
        print("\nExample usage:")
        print("=" * 15)
        print("# Using predefined groups:")
        print('param_groups = ["basic", "background", "extinction"]')
        print("\n# Using individual parameters:")
        print('param_groups = [["norm", "thickness"], ["b0", "ext_l2"]]')
        print("\n# Using named stages:")
        print('param_groups = {"scale": ["norm"], "sample": ["thickness", "extinction"]}')
        print("\n# Mixed approach:")
        print('param_groups = ["basic", ["b0", "ext_l2"], "lattice"]')

    def plot(self, data: "nres.Data" = None, plot_bg: bool = True, correct_tof: bool = True, stage: int = None, index=None, **kwargs):
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
        stage: int, optional
            If provided, plot results from a specific Rietveld fitting stage (1-indexed).
            Only works if Rietveld fitting has been performed.
        index : int, tuple, or str, optional
            For grouped data, specify which group to plot.
            - For 2D grids: can use tuple (0, 0) or string "(0, 0)"
            - For 1D arrays: can use int 5 or string "5"
            - For named groups: use string "groupname"
            If None and data is grouped, raises an error.
        kwargs : dict, optional
            Additional plot settings like color, marker size, etc.

        Returns
        -------
        matplotlib.axes.Axes
            The axes of the plot.
        """
        # Handle grouped data
        if data is not None and hasattr(data, 'is_grouped') and data.is_grouped:
            if index is None:
                raise ValueError(
                    "Data is grouped. Please specify which group to plot using the 'index' parameter.\n"
                    f"Available indices: {data.indices}"
                )
            # Extract the specific group
            from nres.data import Data
            normalized_index = data._normalize_index(index)
            if normalized_index not in data.groups:
                raise ValueError(f"Index {index} not found. Available indices: {data.indices}")

            # Create a non-grouped Data object for this specific group
            group_data = Data()
            group_data.table = data.groups[normalized_index]
            group_data.L = data.L
            group_data.tstep = data.tstep
            group_data.is_grouped = False
            data = group_data

        fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=[3.5, 1], figsize=(6, 5))
        data_object = data.table.dropna().copy() if data else None

        if stage is not None and hasattr(self, "fit_stages") and self.fit_stages:
            # Use specific stage results
            if stage < 1 or stage > len(self.fit_stages):
                raise ValueError(f"Stage {stage} not available. Available stages: 1-{len(self.fit_stages)}")
            
            # Get stage results
            stage_result = self.fit_stages[stage - 1]  # Convert to 0-indexed
            
            # We need to reconstruct the fit data from the original fit
            if hasattr(self, "fit_result") and self.fit_result is not None:
                energy = self.fit_result.userkws["E"]    
                data_values = self.fit_result.data    
                err = 1. / self.fit_result.weights    
            else:
                raise ValueError("Cannot plot stage results without original fit data")
                
            # Use stage parameters to evaluate model
            params = stage_result.params
            best_fit = self.eval(params=params, E=energy)
            residual = (data_values - best_fit) / err
            chi2 = stage_result.redchi if hasattr(stage_result, 'redchi') else np.sum(residual**2) / (len(data_values) - len(params))
            fit_label = f"Stage {stage} fit"
            
        elif hasattr(self, "fit_result"):
            # Use final fit results
            energy = self.fit_result.userkws["E"]
            data_values = self.fit_result.data
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
                data_values = data_object["trans"]
                err = data_object["err"]
                best_fit = self.eval(params=params, E=energy.values)
                residual = (data_values - best_fit) / err
                # Calculate chi2 for the model
                chi2 = np.sum(((data_values - best_fit) / err) ** 2) / (len(data_values) - len(params))
            else:
                energy = self.cross_section.table.dropna().index.values
                data_values = np.nan * np.ones_like(energy)
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
        ax[0].errorbar(energy, data_values, err, marker="o", color=color, ms=ms, zorder=-1, ecolor=ecolor, label="Data")
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


    def plot_stage_progression(self, stages: list = None, **kwargs):
        """
        Plot the progression of Rietveld refinement stages showing how the fit improves.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not hasattr(self, "fit_stages") or not self.fit_stages:
            raise ValueError("No Rietveld stages available. Run fit with method='rietveld' first.")

        if stages is None:
            stages = list(range(1, len(self.fit_stages) + 1))

        # Original data
        if hasattr(self, "fit_result") and self.fit_result is not None:
            energy = self.fit_result.userkws["E"]
            data_values = self.fit_result.data
            err = 1. / self.fit_result.weights
        else:
            raise ValueError("Cannot plot stage progression without original fit data")

        fig, ax = plt.subplots(figsize=(6, 4))

        # Match style: light gray points for data
        ax.errorbar(energy, data_values, err,
                    marker="o", color="0.6", ms=2, alpha=0.7, zorder=-1,
                    ecolor="0.85", label="Data")

        # Use consistent style palette
        colors = plt.cm.plasma(np.linspace(0, 0.85, len(stages)))

        for i, stage in enumerate(stages):
            if stage < 1 or stage > len(self.fit_stages):
                continue

            stage_result = self.fit_stages[stage - 1]
            params = stage_result.params
            best_fit = self.eval(params=params, E=energy)
            chi2 = getattr(stage_result, "redchi", np.nan)

            # Get stage name if available
            stage_name = f"Stage {stage}"
            if hasattr(self, "stages_summary"):
                stage_col = f"Stage_{stage}"
                if (stage_col, "vary") in self.stages_summary.columns:
                    varied_params = self.stages_summary.loc[
                        self.stages_summary[(stage_col, "vary")] == True
                    ].index.tolist()
                    varied_params = [p for p in varied_params if p != "redchi"]
                    if varied_params:
                        stage_name = ", ".join(varied_params[:2]) + (
                            f" +{len(varied_params)-2}" if len(varied_params) > 2 else ""
                        )

            ax.plot(energy, best_fit,
                    color=colors[i], lw=1.2 + 0.4 * i,
                    alpha=0.8,
                    label=f"{stage_name} (χ²={chi2:.3f})" if not np.isnan(chi2) else stage_name)

        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("Transmission")
        ax.set_title("Rietveld Refinement Stage Progression")
        ax.legend(fontsize=8, frameon=False)

        plt.tight_layout()
        return ax


    def plot_chi2_progression(self, **kwargs):
        """
        Plot the χ² progression through Rietveld stages with stage names on x-axis.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not hasattr(self, "fit_stages") or not self.fit_stages:
            raise ValueError("No Rietveld stages available. Run fit with method='rietveld' first.")

        stages = list(range(1, len(self.fit_stages) + 1))
        chi2_values = []
        stage_labels = []

        for stage in stages:
            stage_result = self.fit_stages[stage - 1]
            chi2 = getattr(stage_result, "redchi", np.nan)
            chi2_values.append(chi2)

            label = f"Stage {stage}"
            if hasattr(self, "stages_summary"):
                stage_col = f"Stage_{stage}"
                if (stage_col, "vary") in self.stages_summary.columns:
                    varied_params = self.stages_summary.loc[
                        self.stages_summary[(stage_col, "vary")] == True
                    ].index.tolist()
                    varied_params = [p for p in varied_params if p != "redchi"]
                    if varied_params:
                        label = ", ".join(varied_params[:2]) + (
                            f" +{len(varied_params)-2}" if len(varied_params) > 2 else ""
                        )
            stage_labels.append(label)

        fig, ax = plt.subplots(figsize=(6, 3.5))

        ax.plot(stages, chi2_values, marker="o", lw=2, color="seagreen")

        # Annotate each point
        for stage, chi2 in zip(stages, chi2_values):
            if not np.isnan(chi2):
                ax.annotate(f"{chi2:.3f}", (stage, chi2),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8)

        ax.set_xlabel("Refinement Stage")
        ax.set_ylabel("Reduced χ²")
        ax.set_title("Rietveld χ² Progression")

        # Stage names at bottom
        ax.set_xticks(stages)
        ax.set_xticklabels(stage_labels, rotation=30, ha="right", fontsize=8)

        plt.tight_layout()
        return ax

    def get_stages_summary_table(self):
        """
        Get the stages summary table showing parameter progression through refinement stages.
        
        Returns
        -------
        pandas.DataFrame
            Multi-index DataFrame with parameters as rows and stages as columns.
            Each stage has columns for 'value', 'stderr', 'vary', and 'redchi'.
        """
        if not hasattr(self, "stages_summary"):
            raise ValueError("No stages summary available. Run fit with method='rietveld' first.")
        
        return self.stages_summary

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

    def save(self, filename: str):
        """
        Save the model to a JSON file.

        Parameters
        ----------
        filename : str
            Path to the JSON file where the model will be saved.

        Notes
        -----
        The model is saved as JSON, which is portable and human-readable.
        The saved file can be loaded using the `TransmissionModel.load()` class method.

        Examples
        --------
        >>> model = TransmissionModel(cross_section)
        >>> model.save("my_model.json")
        >>> loaded_model = TransmissionModel.load("my_model.json")
        """
        import json

        # Serialize parameters
        params_dict = {}
        for name, param in self.params.items():
            params_dict[name] = {
                'value': float(param.value),
                'vary': bool(param.vary),
                'min': float(param.min) if param.min is not None else None,
                'max': float(param.max) if param.max is not None else None,
                'expr': param.expr,
            }

        # Serialize cross-section
        xs_dict = {
            'name': self.cross_section.name,
            'materials': self.cross_section.materials,
            'L': float(self.cross_section.L),
            'tstep': float(self.cross_section.tstep),
            'tbins': int(self.cross_section.tbins),
            'first_tbin': int(self.cross_section.first_tbin),
        }

        # Serialize response parameters
        response_dict = None
        if self.response is not None:
            response_dict = {
                'params': {name: {
                    'value': float(p.value),
                    'vary': bool(p.vary),
                    'min': float(p.min) if p.min is not None else None,
                    'max': float(p.max) if p.max is not None else None,
                } for name, p in self.response.params.items()},
                'tstep': float(self.response.tstep),
                'eps': float(self.response.eps),
            }

        # Serialize background parameters
        background_dict = None
        if self.background is not None:
            background_dict = {
                'params': {name: {
                    'value': float(p.value),
                    'vary': bool(p.vary),
                    'min': float(p.min) if p.min is not None else None,
                    'max': float(p.max) if p.max is not None else None,
                } for name, p in self.background.params.items()},
            }

        # Create the model data dictionary
        model_data = {
            'version': '1.0',
            'type': 'TransmissionModel',
            'cross_section': xs_dict,
            'response': response_dict,
            'background': background_dict,
            'params': params_dict,
            'n': float(self.n),
        }

        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load(cls, filename: str) -> 'TransmissionModel':
        """
        Load a model from a JSON file.

        Parameters
        ----------
        filename : str
            Path to the JSON file containing the saved model.

        Returns
        -------
        TransmissionModel
            The loaded model instance.

        Examples
        --------
        >>> model = TransmissionModel.load("my_model.json")
        >>> result = model.fit(data)
        """
        import json

        with open(filename, 'r') as f:
            model_data = json.load(f)

        # Reconstruct cross-section
        xs_data = model_data['cross_section']
        xs = CrossSection()

        # Restore cross-section materials
        for mat_name, mat_info in xs_data['materials'].items():
            xs.add_material(
                mat_name,
                mat_info,
                splitby=mat_info.get('splitby', 'elements'),
                total_weight=mat_info.get('total_weight', 1.0)
            )

        xs.name = xs_data['name']
        xs.L = xs_data['L']
        xs.tstep = xs_data['tstep']
        xs.tbins = xs_data['tbins']
        xs.first_tbin = xs_data['first_tbin']

        # Determine response and background types from saved params
        response_kind = None
        background_kind = None

        if model_data['response'] is not None:
            # Infer response type from parameters
            response_kind = "expo_gauss"  # Default, can be enhanced later

        if model_data['background'] is not None:
            # Infer background type from number of parameters
            n_bg_params = len(model_data['background']['params'])
            if n_bg_params == 3:
                background_kind = "polynomial3"
            elif n_bg_params == 5:
                background_kind = "polynomial5"

        # Create model instance
        model = cls(
            cross_section=xs,
            response=response_kind,
            background=background_kind,
        )

        # Restore all parameter values
        for name, param_data in model_data['params'].items():
            if name in model.params:
                model.params[name].set(
                    value=param_data['value'],
                    vary=param_data['vary'],
                    min=param_data['min'],
                    max=param_data['max'],
                    expr=param_data['expr']
                )

        # Restore response parameters
        if model_data['response'] is not None and model.response is not None:
            for name, param_data in model_data['response']['params'].items():
                if name in model.response.params:
                    model.response.params[name].set(
                        value=param_data['value'],
                        vary=param_data['vary'],
                        min=param_data['min'],
                        max=param_data['max']
                    )

        # Restore background parameters
        if model_data['background'] is not None and model.background is not None:
            for name, param_data in model_data['background']['params'].items():
                if name in model.background.params:
                    model.background.params[name].set(
                        value=param_data['value'],
                        vary=param_data['vary'],
                        min=param_data['min'],
                        max=param_data['max']
                    )

        model.n = model_data['n']

        return model

    def _save_result(self, result, filename: str, include_model: bool = True):
        """
        Save a fit result to a JSON file.

        Parameters
        ----------
        result : lmfit.model.ModelResult
            The fit result to save.
        filename : str
            Path to the JSON file where the result will be saved.
        include_model : bool, optional
            If True, saves the full model with the result. If False, saves
            only a compressed result with fit parameters. Default is True.
        """
        import json
        import numpy as np

        # Serialize fit parameters
        params_dict = {}
        for name, param in result.params.items():
            params_dict[name] = {
                'value': float(param.value),
                'stderr': float(param.stderr) if param.stderr is not None else None,
                'vary': bool(param.vary),
                'min': float(param.min) if param.min is not None else None,
                'max': float(param.max) if param.max is not None else None,
                'expr': param.expr,
            }

        # Serialize fit statistics
        result_dict = {
            'version': '1.0',
            'type': 'FitResult',
            'params': params_dict,
            'success': bool(result.success),
            'chisqr': float(result.chisqr),
            'redchi': float(result.redchi),
            'aic': float(result.aic) if hasattr(result, 'aic') else None,
            'bic': float(result.bic) if hasattr(result, 'bic') else None,
            'nvarys': int(result.nvarys),
            'ndata': int(result.ndata),
            'nfev': int(result.nfev) if hasattr(result, 'nfev') else None,
            'message': result.message if hasattr(result, 'message') else None,
        }

        # Optionally include the model
        if include_model:
            # Temporarily save model to get its JSON representation
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_f:
                temp_filename = temp_f.name

            try:
                self.save(temp_filename)
                with open(temp_filename, 'r') as f:
                    model_dict = json.load(f)
                result_dict['model'] = model_dict
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2)

    @classmethod
    def load_result(cls, filename: str, model: 'TransmissionModel' = None):
        """
        Load a fit result from a JSON file.

        Parameters
        ----------
        filename : str
            Path to the JSON file containing the saved result.
        model : TransmissionModel, optional
            Model to use for the result. If None and the file contains a model,
            it will be loaded from the file. If the file doesn't contain a model,
            this parameter is required.

        Returns
        -------
        tuple
            A tuple containing (model, params_dict) where model is the TransmissionModel
            instance and params_dict contains the fit parameters and statistics.

        Examples
        --------
        >>> model, result_data = TransmissionModel.load_result("my_result.json")
        >>> print(result_data['redchi'])

        >>> # Or with compressed result
        >>> model, result_data = TransmissionModel.load_result("result.json", model=my_model)
        """
        import json
        import tempfile
        import os

        with open(filename, 'r') as f:
            result_data = json.load(f)

        # Load or use provided model
        if 'model' in result_data:
            # Full result with embedded model
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_f:
                json.dump(result_data['model'], temp_f)
                temp_filename = temp_f.name

            try:
                model = cls.load(temp_filename)
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
        elif model is None:
            raise ValueError("Model not found in file and no model provided. "
                           "Either save with include_model=True or provide a model parameter.")

        # Update model parameters with fit results
        for name, param_data in result_data['params'].items():
            if name in model.params:
                model.params[name].set(
                    value=param_data['value'],
                    vary=param_data.get('vary', True),
                    min=param_data.get('min'),
                    max=param_data.get('max'),
                    expr=param_data.get('expr')
                )

        # Return model and result dictionary
        return model, result_data

    def save(self, filename: str):
        """
        Save the model to a JSON file.

        Parameters
        ----------
        filename : str
            Path to the JSON file where the model will be saved.

        Notes
        -----
        The model is saved as JSON, which is portable and human-readable.
        The saved file can be loaded using the `TransmissionModel.load()` class method.

        Examples
        --------
        >>> model = TransmissionModel(cross_section)
        >>> model.save("my_model.json")
        >>> loaded_model = TransmissionModel.load("my_model.json")
        """
        import json

        # Serialize parameters
        params_dict = {}
        for name, param in self.params.items():
            params_dict[name] = {
                'value': float(param.value),
                'vary': bool(param.vary),
                'min': float(param.min) if param.min is not None else None,
                'max': float(param.max) if param.max is not None else None,
                'expr': param.expr,
            }

        # Serialize cross-section
        xs_dict = {
            'name': self.cross_section.name,
            'materials': self.cross_section.materials,
            'L': float(self.cross_section.L),
            'tstep': float(self.cross_section.tstep),
            'tbins': int(self.cross_section.tbins),
            'first_tbin': int(self.cross_section.first_tbin),
        }

        # Serialize response parameters
        response_dict = None
        if self.response is not None:
            response_dict = {
                'params': {name: {
                    'value': float(p.value),
                    'vary': bool(p.vary),
                    'min': float(p.min) if p.min is not None else None,
                    'max': float(p.max) if p.max is not None else None,
                } for name, p in self.response.params.items()},
                'tstep': float(self.response.tstep),
                'eps': float(self.response.eps),
            }

        # Serialize background parameters
        background_dict = None
        if self.background is not None:
            background_dict = {
                'params': {name: {
                    'value': float(p.value),
                    'vary': bool(p.vary),
                    'min': float(p.min) if p.min is not None else None,
                    'max': float(p.max) if p.max is not None else None,
                } for name, p in self.background.params.items()},
            }

        # Create the model data dictionary
        model_data = {
            'version': '1.0',
            'type': 'TransmissionModel',
            'cross_section': xs_dict,
            'response': response_dict,
            'background': background_dict,
            'params': params_dict,
            'n': float(self.n),
        }

        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load(cls, filename: str) -> 'TransmissionModel':
        """
        Load a model from a JSON file.

        Parameters
        ----------
        filename : str
            Path to the JSON file containing the saved model.

        Returns
        -------
        TransmissionModel
            The loaded model instance.

        Examples
        --------
        >>> model = TransmissionModel.load("my_model.json")
        >>> result = model.fit(data)
        """
        import json

        with open(filename, 'r') as f:
            model_data = json.load(f)

        # Reconstruct cross-section
        xs_data = model_data['cross_section']
        xs = CrossSection()

        # Restore cross-section materials
        for mat_name, mat_info in xs_data['materials'].items():
            xs.add_material(
                mat_name,
                mat_info,
                splitby=mat_info.get('splitby', 'elements'),
                total_weight=mat_info.get('total_weight', 1.0)
            )

        xs.name = xs_data['name']
        xs.L = xs_data['L']
        xs.tstep = xs_data['tstep']
        xs.tbins = xs_data['tbins']
        xs.first_tbin = xs_data['first_tbin']

        # Determine response and background types from saved params
        response_kind = None
        background_kind = None

        if model_data['response'] is not None:
            # Infer response type from parameters
            response_kind = "expo_gauss"  # Default, can be enhanced later

        if model_data['background'] is not None:
            # Infer background type from number of parameters
            n_bg_params = len(model_data['background']['params'])
            if n_bg_params == 3:
                background_kind = "polynomial3"
            elif n_bg_params == 5:
                background_kind = "polynomial5"

        # Create model instance
        model = cls(
            cross_section=xs,
            response=response_kind,
            background=background_kind,
        )

        # Restore all parameter values
        for name, param_data in model_data['params'].items():
            if name in model.params:
                model.params[name].set(
                    value=param_data['value'],
                    vary=param_data['vary'],
                    min=param_data['min'],
                    max=param_data['max'],
                    expr=param_data['expr']
                )

        # Restore response parameters
        if model_data['response'] is not None and model.response is not None:
            for name, param_data in model_data['response']['params'].items():
                if name in model.response.params:
                    model.response.params[name].set(
                        value=param_data['value'],
                        vary=param_data['vary'],
                        min=param_data['min'],
                        max=param_data['max']
                    )

        # Restore background parameters
        if model_data['background'] is not None and model.background is not None:
            for name, param_data in model_data['background']['params'].items():
                if name in model.background.params:
                    model.background.params[name].set(
                        value=param_data['value'],
                        vary=param_data['vary'],
                        min=param_data['min'],
                        max=param_data['max']
                    )

        model.n = model_data['n']

        return model

    def _save_result(self, result, filename: str, include_model: bool = True):
        """
        Save a fit result to a JSON file.

        Parameters
        ----------
        result : lmfit.model.ModelResult
            The fit result to save.
        filename : str
            Path to the JSON file where the result will be saved.
        include_model : bool, optional
            If True, saves the full model with the result. If False, saves
            only a compressed result with fit parameters. Default is True.
        """
        import json
        import numpy as np

        # Serialize fit parameters
        params_dict = {}
        for name, param in result.params.items():
            params_dict[name] = {
                'value': float(param.value),
                'stderr': float(param.stderr) if param.stderr is not None else None,
                'vary': bool(param.vary),
                'min': float(param.min) if param.min is not None else None,
                'max': float(param.max) if param.max is not None else None,
                'expr': param.expr,
            }

        # Serialize fit statistics
        result_dict = {
            'version': '1.0',
            'type': 'FitResult',
            'params': params_dict,
            'success': bool(result.success),
            'chisqr': float(result.chisqr),
            'redchi': float(result.redchi),
            'aic': float(result.aic) if hasattr(result, 'aic') else None,
            'bic': float(result.bic) if hasattr(result, 'bic') else None,
            'nvarys': int(result.nvarys),
            'ndata': int(result.ndata),
            'nfev': int(result.nfev) if hasattr(result, 'nfev') else None,
            'message': result.message if hasattr(result, 'message') else None,
        }

        # Optionally include the model
        if include_model:
            # Temporarily save model to get its JSON representation
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_f:
                temp_filename = temp_f.name

            try:
                self.save(temp_filename)
                with open(temp_filename, 'r') as f:
                    model_dict = json.load(f)
                result_dict['model'] = model_dict
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2)

    @classmethod
    def load_result(cls, filename: str, model: 'TransmissionModel' = None):
        """
        Load a fit result from a JSON file.

        Parameters
        ----------
        filename : str
            Path to the JSON file containing the saved result.
        model : TransmissionModel, optional
            Model to use for the result. If None and the file contains a model,
            it will be loaded from the file. If the file doesn't contain a model,
            this parameter is required.

        Returns
        -------
        tuple
            A tuple containing (model, params_dict) where model is the TransmissionModel
            instance and params_dict contains the fit parameters and statistics.

        Examples
        --------
        >>> model, result_data = TransmissionModel.load_result("my_result.json")
        >>> print(result_data['redchi'])

        >>> # Or with compressed result
        >>> model, result_data = TransmissionModel.load_result("result.json", model=my_model)
        """
        import json
        import tempfile
        import os

        with open(filename, 'r') as f:
            result_data = json.load(f)

        # Load or use provided model
        if 'model' in result_data:
            # Full result with embedded model
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_f:
                json.dump(result_data['model'], temp_f)
                temp_filename = temp_f.name

            try:
                model = cls.load(temp_filename)
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
        elif model is None:
            raise ValueError("Model not found in file and no model provided. "
                           "Either save with include_model=True or provide a model parameter.")

        # Update model parameters with fit results
        for name, param_data in result_data['params'].items():
            if name in model.params:
                model.params[name].set(
                    value=param_data['value'],
                    vary=param_data.get('vary', True),
                    min=param_data.get('min'),
                    max=param_data.get('max'),
                    expr=param_data.get('expr')
                )

        # Return model and result dictionary
        return model, result_data
