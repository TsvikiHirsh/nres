import pandas
import lmfit

class Transmission_model:

    def __init__(self,isotopes,
                      thickness=1,
                      n=1,
                      norm=1,
                      bg0=0.1,
                      bg1=0,
                      bg2=0,
                      vary_weights=False):
        
        self.isotopes = isotopes

    def model(x):
        xs = sum([weight*cross_section for weight,isotope]
        trans = exp(-self.params["n"]*self.params["thickness"])