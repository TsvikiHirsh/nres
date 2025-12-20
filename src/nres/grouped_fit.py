"""
Grouped fit functionality for spatially-resolved neutron transmission data.

This module contains the GroupedFitResult class for storing and analyzing
fit results from grouped/spatially-resolved data, along with helper functions
for parallel fitting.
"""

import lmfit
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, TYPE_CHECKING
import json
import ast

if TYPE_CHECKING:
    from nres.models import TransmissionModel


def _fit_single_group_worker(args):
    """
    Worker function for parallel fitting of a single group.

    This function is defined at module level so it can be pickled for multiprocessing.
    It reconstructs the model from a serialized dict, fits the data, and returns
    only pickleable results.

    Parameters
    ----------
    args : tuple
        (idx, model_dict, table_dict, L, tstep, fit_kwargs)

    Returns
    -------
    tuple
        (idx, result_dict) where result_dict contains pickleable fit results,
        or (idx, error_string) if fitting failed.
    """
    idx, model_dict, table_dict, L, tstep, fit_kwargs = args

    try:
        # Reconstruct model from dict
        from nres.models import TransmissionModel
        from nres.data import Data
        import pandas as pd

        model = TransmissionModel._from_dict(model_dict)

        # Reconstruct data
        group_data = Data()
        group_data.table = pd.DataFrame(table_dict)
        group_data.L = L
        group_data.tstep = tstep

        # Fit
        result = model.fit(group_data, **fit_kwargs)

        # Extract only pickleable attributes
        result_dict = _extract_pickleable_result(result)

        return idx, result_dict

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return idx, {'error': error_msg}


def _extract_pickleable_result(fit_result):
    """
    Extract only pickleable attributes from a fit result.

    Returns a dictionary with all the important fit result data
    that can be safely passed between processes.
    """
    result_dict = {}

    # Core fit statistics
    for attr in ['success', 'chisqr', 'redchi', 'aic', 'bic',
                 'nvarys', 'ndata', 'nfev', 'message', 'method']:
        if hasattr(fit_result, attr):
            result_dict[attr] = getattr(fit_result, attr)

    # Parameters - serialize to JSON string
    if hasattr(fit_result, 'params'):
        result_dict['params_json'] = fit_result.params.dumps()

    # Best fit values
    if hasattr(fit_result, 'best_values'):
        result_dict['best_values'] = dict(fit_result.best_values)

    # Init values
    if hasattr(fit_result, 'init_values'):
        result_dict['init_values'] = dict(fit_result.init_values)

    # Residual and best_fit arrays
    if hasattr(fit_result, 'residual') and fit_result.residual is not None:
        result_dict['residual'] = fit_result.residual.tolist()
    if hasattr(fit_result, 'best_fit') and fit_result.best_fit is not None:
        result_dict['best_fit'] = fit_result.best_fit.tolist()

    # Covariance matrix
    if hasattr(fit_result, 'covar') and fit_result.covar is not None:
        result_dict['covar'] = fit_result.covar.tolist()

    # Variable names and init_vals
    if hasattr(fit_result, 'var_names'):
        result_dict['var_names'] = list(fit_result.var_names)
    if hasattr(fit_result, 'init_vals'):
        result_dict['init_vals'] = list(fit_result.init_vals)

    # User keywords (needed for plotting)
    if hasattr(fit_result, 'userkws') and fit_result.userkws:
        result_dict['userkws'] = dict(fit_result.userkws)

    # Data array (needed for plotting)
    if hasattr(fit_result, 'data') and fit_result.data is not None:
        result_dict['data'] = fit_result.data.tolist()

    # Weights array (needed for plotting)
    if hasattr(fit_result, 'weights') and fit_result.weights is not None:
        result_dict['weights'] = fit_result.weights.tolist()

    # Stage results if present (for rietveld/staged fits)
    if hasattr(fit_result, 'stages_summary') and fit_result.stages_summary is not None:
        import pandas as pd
        if isinstance(fit_result.stages_summary, pd.DataFrame):
            result_dict['stages_summary'] = fit_result.stages_summary.to_json(orient='split')

    return result_dict


def _reconstruct_result_from_dict(result_dict, model=None):
    """
    Reconstruct a minimal ModelResult-like object from a pickleable dict.

    This creates an object with the same interface as lmfit.ModelResult
    but from serialized data.
    """
    import lmfit
    import numpy as np

    # Create a minimal result object using SimpleNamespace
    from types import SimpleNamespace
    result = SimpleNamespace()

    # Restore basic attributes
    for attr in ['success', 'chisqr', 'redchi', 'aic', 'bic',
                 'nvarys', 'ndata', 'nfev', 'message', 'method']:
        if attr in result_dict:
            setattr(result, attr, result_dict[attr])

    # Restore parameters
    if 'params_json' in result_dict:
        result.params = lmfit.Parameters()
        result.params.loads(result_dict['params_json'])

    # Restore best values and init values
    if 'best_values' in result_dict:
        result.best_values = result_dict['best_values']
    if 'init_values' in result_dict:
        result.init_values = result_dict['init_values']

    # Restore arrays
    if 'residual' in result_dict:
        result.residual = np.array(result_dict['residual'])
    if 'best_fit' in result_dict:
        result.best_fit = np.array(result_dict['best_fit'])
    if 'covar' in result_dict:
        result.covar = np.array(result_dict['covar'])

    # Restore variable info
    if 'var_names' in result_dict:
        result.var_names = result_dict['var_names']
    if 'init_vals' in result_dict:
        result.init_vals = result_dict['init_vals']

    # Restore userkws (needed for plotting)
    if 'userkws' in result_dict:
        result.userkws = result_dict['userkws']

    # Restore data (needed for plotting)
    if 'data' in result_dict:
        result.data = np.array(result_dict['data'])

    # Restore weights (needed for plotting)
    if 'weights' in result_dict:
        result.weights = np.array(result_dict['weights'])

    # Restore stages_summary if available
    if 'stages_summary' in result_dict and result_dict['stages_summary'] is not None:
        import pandas as pd
        from io import StringIO
        result.stages_summary = pd.read_json(StringIO(result_dict['stages_summary']), orient='split')

    # Add model reference if provided
    result.model = model

    return result


class GroupedFitResult:
    """
    Container for fit results from grouped data.

    Stores multiple ModelResult objects indexed by their group identifiers
    (integers, tuples, or strings depending on the data structure).

    Attributes:
    -----------
    results : dict
        Dictionary mapping group indices to lmfit.ModelResult objects.
    indices : list
        List of group indices in order.
    group_shape : tuple or None
        Shape of the grouped data ((ny, nx) for 2D, (n,) for 1D, None for named).

    Examples:
    ---------
    >>> # Access individual results
    >>> grouped_result = model.fit(grouped_data)
    >>> result_0_0 = grouped_result[(0, 0)]
    >>> result_0_0.plot()

    >>> # Plot parameter map
    >>> grouped_result.plot_parameter_map("thickness")

    >>> # Print summary
    >>> grouped_result.summary()
    """

    def __init__(self, group_shape=None):
        """
        Initialize an empty GroupedFitResult.

        Parameters:
        -----------
        group_shape : tuple or None
            Shape of the grouped data.
        """
        self.results = {}
        self.indices = []
        self.group_shape = group_shape

    def _normalize_index(self, index):
        """
        Normalize index for consistent lookup.
        Converts tuples to strings without spaces: (10, 20) -> "(10,20)"
        Accepts both "(10,20)" and "(10, 20)" string formats.
        """
        if isinstance(index, tuple):
            return str(index).replace(" ", "")
        elif isinstance(index, str):
            return index.replace(" ", "")
        else:
            return str(index)

    def _parse_string_index(self, string_idx):
        """
        Parse a string index back to its original form.
        "(10,20)" or "(10, 20)" -> (10, 20)
        "5" -> 5
        "center" -> "center"
        """
        try:
            parsed = ast.literal_eval(string_idx)
            return parsed
        except (ValueError, SyntaxError):
            return string_idx

    def add_result(self, index, result):
        """
        Add a fit result for a specific group.

        Parameters:
        -----------
        index : int, tuple, or str
            The group index.
        result : lmfit.ModelResult
            The fit result for this group.
        """
        # Normalize index for consistent storage
        normalized_index = self._normalize_index(index)
        self.results[normalized_index] = result
        if normalized_index not in self.indices:
            self.indices.append(normalized_index)

    def __getitem__(self, index):
        """
        Access a specific group's result.

        Supports flexible index access:
        - Tuples: (0, 0) or "(0,0)" or "(0, 0)"
        - Integers: 5 or "5"
        - Strings: "groupname"
        """
        normalized_index = self._normalize_index(index)

        if normalized_index not in self.results:
            raise KeyError(f"Index {index} not found in results. Available: {self.indices}")
        return self.results[normalized_index]

    def __len__(self):
        """Return number of group results."""
        return len(self.results)

    def __repr__(self):
        """String representation."""
        return f"GroupedFitResult({len(self.results)} groups, shape={self.group_shape})"

    def summary(self):
        """
        Print summary statistics for all group fits.

        Returns a pandas DataFrame with fit statistics and parameter values/errors for each group.

        Returns:
        --------
        pandas.DataFrame
            Summary table with columns: index, success, redchi, parameters, and parameter errors.
        """
        import pandas as pd

        summary_data = []
        for idx in self.indices:
            result = self.results[idx]
            row = {
                'index': str(idx),
                'success': result.success if hasattr(result, 'success') else None,
                'redchi': result.redchi if hasattr(result, 'redchi') else None,
                'chisqr': result.chisqr if hasattr(result, 'chisqr') else None,
                'nfev': result.nfev if hasattr(result, 'nfev') else None,
                'nvarys': result.nvarys if hasattr(result, 'nvarys') else None,
            }

            # Add all parameter values and errors
            if hasattr(result, 'params'):
                for param_name in result.params:
                    param = result.params[param_name]
                    row[param_name] = param.value
                    row[f"{param_name}_err"] = param.stderr if param.stderr is not None else np.nan

            summary_data.append(row)

        df = pd.DataFrame(summary_data)
        print("\nGrouped Fit Results Summary")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        return df

    def stages_summary(self, index):
        """
        Get the stages summary table for a specific group.

        Parameters:
        -----------
        index : int, tuple, or str
            The group index to get stages summary for.

        Returns:
        --------
        pandas.DataFrame or None
            The stages summary table for the specified group, or None if not available.
        """
        normalized_index = self._normalize_index(index)

        if normalized_index not in self.results:
            raise ValueError(f"Index {index} not found. Available indices: {self.indices}")

        result = self.results[normalized_index]

        if hasattr(result, 'stages_summary'):
            return result.stages_summary
        else:
            print(f"Warning: No stages_summary available for index {index}")
            return None

    def plot_parameter_map(self, param_name, query=None, kind=None, **kwargs):
        """
        Plot spatial map of a fitted parameter value, error, or fit statistic.

        Parameters:
        -----------
        param_name : str
            Name of the parameter to visualize. Can be:
            - Parameter name: "thickness", "norm", etc.
            - Parameter error: "thickness_err", "norm_err", etc.
            - Fit statistic: "redchi", "chisqr", "aic", "bic"
        query : str, optional
            Pandas query string to filter results (e.g., "redchi < 2").
        kind : str, optional
            Plot type. If None (default), auto-detected based on group_shape:
            - 2D data: 'pcolormesh'
            - 1D data: 'line'
            - Named groups: 'bar'
        **kwargs : dict, optional
            Additional plotting parameters (cmap, title, vmin, vmax, figsize).

        Returns:
        --------
        matplotlib.Axes
            The plot axes.
        """
        import pandas as pd

        # Auto-detect plot kind based on group_shape if not specified
        if kind is None:
            if self.group_shape and len(self.group_shape) == 2:
                kind = 'pcolormesh'
            elif self.group_shape and len(self.group_shape) == 1:
                kind = 'line'
            else:
                kind = 'bar'

        # Build DataFrame with all parameters, errors, and statistics
        data_for_query = []
        param_values = {}

        for idx in self.indices:
            result = self.results[idx]
            row = {'index': idx}

            try:
                # Add all parameter values and errors
                for pname in result.params:
                    param = result.params[pname]
                    row[pname] = param.value
                    row[f"{pname}_err"] = param.stderr if param.stderr is not None else np.nan

                # Add fit statistics
                row['redchi'] = result.redchi if hasattr(result, 'redchi') else np.nan
                row['chisqr'] = result.chisqr if hasattr(result, 'chisqr') else np.nan
                row['aic'] = result.aic if hasattr(result, 'aic') else np.nan
                row['bic'] = result.bic if hasattr(result, 'bic') else np.nan
                row['nfev'] = result.nfev if hasattr(result, 'nfev') else np.nan

                data_for_query.append(row)

                # Extract the specific parameter value requested
                if param_name.endswith('_err'):
                    # Error requested
                    base_param = param_name[:-4]
                    if base_param in result.params:
                        param_values[idx] = result.params[base_param].stderr
                elif param_name in ['redchi', 'chisqr', 'aic', 'bic', 'nfev']:
                    # Statistic requested
                    param_values[idx] = getattr(result, param_name, np.nan)
                elif param_name in result.params:
                    # Parameter value requested
                    param_values[idx] = result.params[param_name].value
                else:
                    param_values[idx] = np.nan

            except Exception as e:
                param_values[idx] = np.nan

        # Apply query filter if provided
        indices_to_plot = self.indices
        if query:
            df = pd.DataFrame(data_for_query)
            try:
                filtered_df = df.query(query)
                indices_to_plot = [row['index'] for _, row in filtered_df.iterrows()]
                # Mask out filtered indices
                for idx in self.indices:
                    if idx not in indices_to_plot:
                        param_values[idx] = np.nan
            except Exception as e:
                print(f"Warning: Query failed: {e}")
                print("Plotting all data without filtering.")

        # Extract kwargs
        cmap = kwargs.pop("cmap", "viridis")
        title = kwargs.pop("title", None)
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        figsize = kwargs.pop("figsize", None)

        # Create visualization based on group_shape
        if self.group_shape and len(self.group_shape) == 2:
            # 2D pcolormesh
            xs = []
            ys = []
            for idx_str in self.indices:
                idx = self._parse_string_index(idx_str)
                if isinstance(idx, tuple) and len(idx) == 2:
                    xs.append(idx[0])
                    ys.append(idx[1])
            xs = sorted(set(xs))
            ys = sorted(set(ys))

            if len(xs) == 0 or len(ys) == 0:
                raise ValueError("No valid 2D indices found for plotting")

            # Calculate grid spacing
            x_spacing = xs[1] - xs[0] if len(xs) > 1 else 1
            y_spacing = ys[1] - ys[0] if len(ys) > 1 else 1

            # Create coordinate arrays including edges for pcolormesh
            x_edges = np.array(xs) - x_spacing / 2
            x_edges = np.append(x_edges, xs[-1] + x_spacing / 2)
            y_edges = np.array(ys) - y_spacing / 2
            y_edges = np.append(y_edges, ys[-1] + y_spacing / 2)

            # Create 2D array for values
            param_array = np.full((len(ys), len(xs)), np.nan)

            # Map indices to array positions
            x_map = {x: i for i, x in enumerate(xs)}
            y_map = {y: i for i, y in enumerate(ys)}

            for idx_str in self.indices:
                idx = self._parse_string_index(idx_str)
                if isinstance(idx, tuple) and len(idx) == 2:
                    x, y = idx
                    if x in x_map and y in y_map:
                        param_array[y_map[y], x_map[x]] = param_values[idx_str]

            fig, ax = plt.subplots(figsize=figsize)
            im = ax.pcolormesh(x_edges, y_edges, param_array, cmap=cmap, vmin=vmin, vmax=vmax,
                              shading='flat', **kwargs)
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            ax.set_aspect('equal')
            if title is None:
                title = f"{param_name} Map"
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label=param_name)
            return ax

        elif self.group_shape and len(self.group_shape) == 1:
            # 1D plot
            indices_array = np.array([self._parse_string_index(idx) for idx in self.indices])
            values = np.array([param_values[idx] if param_values[idx] is not None else np.nan for idx in self.indices])

            # Get errors if available for errorbar plot
            errors = None
            if kind == 'errorbar' and not param_name.endswith('_err'):
                errors = []
                for idx in self.indices:
                    result = self.results[idx]
                    if param_name in result.params:
                        stderr = result.params[param_name].stderr
                        errors.append(stderr if stderr is not None else 0)
                    else:
                        errors.append(0)
                errors = np.array(errors)

            fig, ax = plt.subplots(figsize=figsize)

            if kind == 'line':
                ax.plot(indices_array, values, 'o-', **kwargs)
            elif kind == 'bar':
                ax.bar(indices_array, values, **kwargs)
            elif kind == 'errorbar':
                if errors is not None:
                    ax.errorbar(indices_array, values, yerr=errors, fmt='o-', capsize=5, **kwargs)
                else:
                    ax.plot(indices_array, values, 'o-', **kwargs)
            else:
                raise ValueError(f"Unknown kind '{kind}'. Must be 'line', 'bar', or 'errorbar'.")

            ax.set_xlabel("Index")
            ax.set_ylabel(param_name)
            if title is None:
                title = f"{param_name} vs Index"
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            return ax

        else:
            # Named indices - bar or line plot
            fig, ax = plt.subplots(figsize=figsize)
            positions = np.arange(len(self.indices))
            values = [param_values[idx] if param_values[idx] is not None else np.nan for idx in self.indices]

            # Get errors if available for errorbar plot
            errors = None
            if kind == 'errorbar' and not param_name.endswith('_err'):
                errors = []
                for idx in self.indices:
                    result = self.results[idx]
                    if param_name in result.params:
                        stderr = result.params[param_name].stderr
                        errors.append(stderr if stderr is not None else 0)
                    else:
                        errors.append(0)

            if kind == 'line':
                ax.plot(positions, values, 'o-', **kwargs)
            elif kind == 'bar':
                ax.bar(positions, values, **kwargs)
            elif kind == 'errorbar':
                if errors is not None:
                    ax.errorbar(positions, values, yerr=errors, fmt='o', capsize=5, **kwargs)
                else:
                    ax.plot(positions, values, 'o', **kwargs)
            else:
                raise ValueError(f"Unknown kind '{kind}'. Must be 'line', 'bar', or 'errorbar'.")

            ax.set_xticks(positions)
            ax.set_xticklabels(self.indices, rotation=45, ha='right')
            ax.set_ylabel(param_name)
            if title is None:
                title = f"{param_name} by Group"
            ax.set_title(title)
            plt.tight_layout()
            return ax
