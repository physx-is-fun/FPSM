from libraries import *
from variables import *

# Defining a class for the simulation parameters
# # Class for holding info about the simulation params
class SIM_config:
    def __init__(self,N,time_window,frequency0,wavelength0):
        self.wavelength0 = wavelength0 
        self.number_of_points=N
        dt = time_window / N
        self.time_step = dt
        t = np.linspace(0,time_window,N)                                                                                  
        t = t - np.mean(t)
        self.t = t 
        f = fftshift(fftfreq(N,d=dt))
        self.f=f                                         
        self.frequency0 = frequency0
        f_rel = f + frequency0
        self.f_rel = f_rel
        wavelength = speed_of_light / f
        self.wavelength = wavelength                                                           
        wavelength_rel = wavelength + wavelength0
        self.wavelength_rel = wavelength_rel

# Class for holding info about the fiber
class Fiber_config:
    def __init__(self,nsteps,L,gamma,beta2,alpha_dB_per_km):
        self.nsteps=nsteps
        self.ntraces=nsteps+1                                           
        self.dz=L/nsteps
        self.zlocs_array=np.linspace(0,L,nsteps)
        #dz = 1e-14
        #self.dz = dz
        #self.zlocs_array=np.linspace(0,nsteps*dz,nsteps)                       
        self.gamma=gamma
        self.beta2=beta2
        self.alpha_dB_per_km=alpha_dB_per_km