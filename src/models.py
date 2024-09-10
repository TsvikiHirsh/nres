import lmfit
import numpy as np



class TransmissionModelResult(lmfit.model.ModelResult):
    def __init__(self, model, params, **kwargs):
        """
        Custom result class for TransmissionModel.
        """
        super().__init__(model, params, **kwargs)

    def plot(self, **kwargs):
        """
        Custom plot method that includes a legend with isotopes.
        
        Parameters:
        - kwargs: keyword arguments for the plot function.
        
        Returns:
        - matplotlib figure and axis.
        """
        fig, ax = plt.subplots()
        # Plot the data and the best-fit model
        self.plot_fit(ax=ax, **kwargs)
        
        # Create custom legend based on isotopes
        legend_text = "Isotopes: "
        ax.legend([legend_text], loc='best')

        return fig, ax

class TransmissionModel(lmfit.Model):
    def __init__(self, cross_section, vary_weights=False, vary_background=False, **kwargs):
        """
        Initialize the TransmissionModel, a subclass of lmfit.Model.

        Parameters:
        - cross_section: callable
            A function that takes energy (E) as input and returns the cross section.
        - vary_weights: bool, optional (default=False)
            If True, allows the isotope weights to vary during fitting.
        - vary_background: bool, optional (default=False)
            If True, allows the background parameters (b0, b1, b2) to vary during fitting.
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
        bg_args = {"b0": kwargs.get("b0", 1e-5), "b1": kwargs.get("b1", 1e-5), "b2": kwargs.get("b2", 1e-5)}
        for b in bg_args:
            self.params.add(b, value=bg_args[b], vary=vary_background)

        # set the n parameter as fixed
        self.params.add("n", value=0.01, vary=False)

    def transmission(self, E, thickness=1, n=0.01, norm=1., b0=0., b1=0., b2=0.):
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
        # Background polynomial
        bg = b0 + b1 * np.sqrt(E) + b2 * np.sqrt(E)
        
        # Transmission function
        T = norm * np.exp(-self.cross_section(E) * thickness * n) * (1 - bg) + bg
        return T

    # def fit(self, data, params=None, **kwargs):
    #     """
    #     Fit the model to the data.

    #     Parameters:
    #     - data: array-like
    #         The data to fit the model to.
    #     - params: Parameters object, optional
    #         The initial parameter values for the fit.
    #     - kwargs: dict
    #         Additional keyword arguments passed to the lmfit.Model.fit method.

    #     Returns:
    #     - TransmissionModelResult
    #         The result of the fit.
    #     """
    #     # Perform the fit using the parent class's fit method
    #     fit_result = super().fit(data, params=params or self.params, **kwargs)
    #     return TransmissionModelResult(fit_result, params or self.params)
    

