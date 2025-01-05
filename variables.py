from libraries import *

# constants
pi = np.pi
speed_of_light=300 # Speed of light [nm/fs]

# Defining parameters for the simulation
# Initialize Gaussian pulse parameters (OCTAVIUS-85M-HP from THORLABS) https://www.thorlabs.com/thorproduct.cfm?partnumber=OCTAVIUS-85M-HP
wavelength0=800                                             # Pulse central wavelengt [nm]
frequency0=speed_of_light/wavelength0                       # Pulse central frequency [Hz] 0.375*1e15 Hz = 0.375 PHz which equals to 800 nm
duration=8                                                  # Pulse duration in FWHM [s]
repetition_frequency=85*1e6                                 # Pulse repetition frequency [Hz]
average_power=600*1e-3                                      # Pulse average power [W]
pulse_energy=average_power/repetition_frequency             # Pulse energy [J]
peak_power=pulse_energy/duration                            # Pulse peak power [W]
amplitude = np.sqrt(peak_power)                             # Electrical field strength amplitude in units of sqrt(W)
N= 2**10 #2**10                                             # Number of points                                                    
#dt = 0.01                                                  # Time resolution [fs]
Time_window=100                                             # Time window [fs]

# Defining the parameters of the fiber
Length=1e-6 #2e-10                                                                # Fiber length [km]
nsteps=2**15 #2**15                                                                # Number of steps we divide the fiber into
effective_mode_diameter=5e-6                                                       # Effective mode diameter [m] from https://www.thorlabs.com/thorproduct.cfm?partnumber=780HP
effective_mode_area=(pi/4)*effective_mode_diameter**2                              # Effective mode area [m^2]
nonlinear_refractive_index=2.7*1e-20                                               # Nonlinear refractive index [m^2/W] of fused silica @ 800 nm from https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-26-37940&id=424534
gammaconstant=(2*pi*nonlinear_refractive_index)/(wavelength0*effective_mode_area)  # Nonlinear parameter [1/(W*m)]
gammaconstant *= 1e3                                                               # Nonlinear parameter [1/(W*km)]
beta2=36.16                                                                        # GVD in fs^2/mm (units typically used when referring to beta2) of fused silica @ 800nm from https://www.newport.com/n/the-effect-of-dispersion-on-ultrashort-pulses
beta2 *= 1e6                                                                       # GVD in fs^2/km 
alpha_dB_per_km=0.2                                                                # Power attenuation coeff in deciBel per mm. Usual value @ 1550 nm is 0.2 dB/km
# NOTE: beta2>0 is normal dispersion with red light pulling ahead, causing a negative leading chirp
# NOTE: beta2<0 is anomalous dispersion with blue light pulling ahead, causing a positive leading chirp

# Some useful parameters
nonlinear_length=1/(gammaconstant*peak_power)
dispersion_length=(duration**2)/(np.abs(beta2))