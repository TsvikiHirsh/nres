
from nres import utils
import pandas as pd
import numpy as np

class Data:

    def __init__(self,**kwargs):
        self.table = None
        
    @classmethod
    def _read_counts(cls,filename="run2_graphite_00000/graphite.csv"):
        df = pd.read_csv(filename,names=["tof","counts","err"],header=None,skiprows=1)
        if all(df["err"].isnull()):
            df["err"] = np.sqrt(df["counts"])
        
        df.attrs["label"] = filename.split("/")[-1].rstrip(".csv")
        return df
    
    @classmethod
    def from_counts(cls, signal:str ,openbeam:str ,
                    empty_signal:str="",empty_openbeam:str="",
                    tstep:float=1.56255e-9,L:float=10.59):
        signal = cls._read_counts(signal)
        openbeam = cls._read_counts(openbeam)

        signal["energy"] = utils.time2energy(signal["tof"]*tstep,L)

        transmission = signal["counts"]/openbeam["counts"]
        err = transmission*np.sqrt((signal["err"]/signal["counts"])**2 + (openbeam["err"]/openbeam["counts"])**2)


        if empty_signal and empty_openbeam:
            empty_signal = cls._read_counts(empty_signal)
            empty_openbeam = cls._read_counts(empty_openbeam)


            transmission*=empty_openbeam["counts"]/empty_signal["counts"]
            err = transmission*np.sqrt((signal["err"]/signal["counts"])**2 + (openbeam["err"]/openbeam["counts"])**2 +\
                                (empty_signal["err"]/empty_signal["counts"])**2 + (empty_openbeam["err"]/empty_openbeam["counts"])**2)
        
        df = pd.DataFrame({"energy":signal["energy"],"trans":transmission,"err":err,})
        df.attrs["label"]  = signal.attrs["label"]
        self_data = cls()
        self_data.table = df
        self_data.tgrid = signal["tof"]

        return self_data
    
    @classmethod
    def from_transmission(cls, filename:str):
        df = pd.read_csv(filename,names=["energy","trans","err"],header=None,skiprows=0,delim_whitespace=True)
        self_data = cls()
        self_data.table = df
        return self_data
    
    def plot(self,**kwargs):
        xlim = kwargs.pop("xlim",(0.5e6,1e7))
        ylim = kwargs.pop("ylim",(0.,1.))
        ecolor = kwargs.pop("ecolor","0.8")
        xlabel = kwargs.pop("xlabel","Energy [eV]")
        ylabel = kwargs.pop("ylabel","Transmission")
        logx = kwargs.pop("logx",True)
        self.table.dropna().plot(x="energy",y="trans",yerr="err",
                                 xlim=xlim,ylim=ylim,logx=logx,ecolor=ecolor,
                                 xlabel=xlabel,ylabel=ylabel,**kwargs)