from variables import *
from libraries import *
from classes import *

sim_config = SIM_config(N,Time_window,frequency0,wavelength0)
fiber=Fiber_config(nsteps,Length,gammaconstant,beta2,alpha_dB_per_km)

# Defining fuctions to calculating the phase, chirp and wavelet
def getPhase(pulse):
    phi=np.unwrap(np.angle(pulse)) #Get phase starting from 1st entry
    phi=phi-phi[int(len(phi)/2)]   #Center phase on middle entry
    return phi    

def getChirp(time,pulse):
    phi=getPhase(pulse)
    dphi=np.diff(phi ,prepend = phi[0] - (phi[1]  - phi[0]  ),axis=0) #Change in phase. Prepend to ensure consistent array size 
    dt  =np.diff(time,prepend = time[0]- (time[1] - time[0] ),axis=0) #Change in time.  Prepend to ensure consistent array size
    return -1/(2*pi)*dphi/dt

def wavelet(t,duration_s,frequency_Hz):
    wl = np.exp(-1j*2*pi*frequency_Hz*t)*np.sqrt(np.exp(-0.5*(t/duration_s)**2 )/np.sqrt(2*pi)/duration_s)
    return wl

# Defining Functions to simulate a Gaussian pulse
# # Function returns pulse power or spectrum PSD
def getPower(amplitude):
    return np.abs(amplitude) ** 2

# Function gets the energy of a pulse or spectrum by integrating the power
def getEnergy(time_or_frequency,amplitude):
    return np.trapz(getPower(amplitude),time_or_frequency)

def GaussianPulseTime(time,amplitude,duration):
    return amplitude*np.exp(-2*np.log(2)*((time)/(duration))**2)*(1+0j)
    #return amplitude*np.exp(-2*np.log(2)*((time)/(duration))**2)*np.exp(1j*2*pi*time*frequency0)

def GaussianPulseFrequency(frequency,frequency0,amplitude,duration):
    return 2*amplitude*duration*np.sqrt(pi/(8*np.log(2)))*np.exp(-((duration**2)/(8*np.log(2)))*(2*pi*frequency - 2*pi*frequency0)**2)*(1+0j)

# Getting the spectrum based on a given pulse
def getSpectrumFromPulse(time,frequency,pulse_amplitude):
    #pulseEenergy=getEnergy(time,pulse_amplitude) # Get pulse energy
    dt=time[1]-time[0]
    spectrum_amplitude=fftshift(fft(pulse_amplitude))*dt # Take FFT and do shift
    #spectrumEnergy=getEnergy(frequency,spectrum_amplitude) # Get spectrum energy
    #err=np.abs((pulseEenergy/spectrumEnergy-1))
    #assert( err<1e-7 ), f'ERROR = {err}: Energy changed when going from Pulse to Spectrum!!!'
    return spectrum_amplitude

def getPulseFromSpectrum(time,frequency,spectrum_aplitude):
    #spectrumEnergy=getEnergy(frequency,spectrum_aplitude)
    dt=time[1]-time[0]
    pulse=ifft(ifftshift(spectrum_aplitude))/dt
    #pulseEnergy=getEnergy(time,pulse)
    #err=np.abs((pulseEnergy/spectrumEnergy-1))
    #assert( err<1e-7 ), f'ERROR = {err}: Energy changed when going from Spectrum to Pulse!!!'
    return pulse

# Equivalent function for generating a Gaussian spectrum
def GaussianSpectrum(time,frequency,amplitude,bandwidth):
    return getSpectrumFromPulse(time,frequency,GaussianPulseTime(time,amplitude,1/bandwidth))

# Getting FWHM based on a given pulse
# Find the FWHM of the frequency/time domain of the signal
def FWHM(X, Y):
    deltax = X[1] - X[0]
    half_max = max(Y) / 2.
    l = np.where(Y > half_max, 1, 0)
    return np.sum(l) * deltax

def Linear_term():
    return  - fiber.alpha_dB_per_km / 2 + (1j * fiber.beta2 / 2) * (1j * 2 * pi * sim_config.f) ** 2

def Nonlinear_term():
    return - 1j * fiber.gamma 

def RightHandSide(t,y):
    LinearSpectrumVector = getSpectrumFromPulse(sim_config.t,sim_config.f,y)
    NonlinearSpectrumVector = getSpectrumFromPulse(sim_config.t,sim_config.f,getPower(y)*y)
    toInverseFourierTransform = Linear_term() * LinearSpectrumVector + Nonlinear_term() * NonlinearSpectrumVector
    dydz = getPulseFromSpectrum(sim_config.t,sim_config.f,toInverseFourierTransform)
    return dydz

# Defining the Fourier Pseudo Spectral Method function
def FPSM(fiber:Fiber_config,sim:SIM_config,pulse):
    pulseMatrix = []
    spectrumMatrix = []
    solver = complex_ode(RightHandSide)
    z0 = fiber.zlocs_array[0]
    z1 = fiber.zlocs_array[1]
    z_last = fiber.zlocs_array[-1]
    dz = np.abs(z1 - z0)
    solver.set_initial_value(pulse,z0)
    i = 0
    while solver.successful() and solver.t < z_last:
        solver.integrate(solver.t+dz)
        pulseMatrix.append(solver.y)
        spectrumVector = getSpectrumFromPulse(sim.t,sim.f,solver.y)
        spectrumMatrix.append(spectrumVector)
        i += 1
        delta = int(round(i*100/fiber.nsteps)) - int(round((i-1)*100/fiber.nsteps))
        if delta == 1:
            print(str(int(round(i*100/fiber.nsteps))) + " % ready")
    pulseMatrix = np.array(pulseMatrix)
    spectrumMatrix = np.array(spectrumMatrix)
    return pulseMatrix, spectrumMatrix

def savePlot(fileName):
    if not os.path.isdir('results/'):
        os.makedirs('results/')
    plt.savefig('results/%s.png'%(fileName))

def plotFirstAndLastPulse(matrix, sim:SIM_config):
    t=sim.t
    plt.figure()
    plt.title("Initial pulse and final pulse")
    power = getPower(matrix[0,:])
    maximum_power=np.max(power)
    plt.plot(t,getPower(matrix[0,:])/maximum_power,label="Initial Pulse")
    plt.plot(t,getPower(matrix[-1,:])/maximum_power,label="Final Pulse")
    plt.axis([-5*duration,5*duration,0,1])
    plt.xlabel("Time [fs]")
    plt.ylabel("Power [arbitrary unit]")
    plt.legend()
    savePlot('initial and final pulse')
    plt.show()

def plotFirstAndLastSpectrum(matrix,sim:SIM_config,FWHM_frequency_final):
    f=sim.f_rel
    frequency0=sim.frequency0
    plt.figure()
    plt.title("Initial spectrum and final spectrum")
    power = getPower(matrix[0,:])
    maximum_power=np.max(power)
    plt.plot(f,getPower(matrix[0,:])/maximum_power,label="Initial Spectrum")
    plt.plot(f,getPower(matrix[-1,:])/maximum_power,label="Final Spectrum")
    plt.axis([frequency0-5*FWHM_frequency_final,frequency0+5*FWHM_frequency_final,0,1])
    plt.xlabel("Frequency [pHz]")
    plt.ylabel("Power spectral density [arbitrary unit]")
    plt.legend()
    savePlot('initial and final spectrum')
    plt.show()

def plotPulseMatrix2D(matrix,fiber:Fiber_config,sim:SIM_config):
    fig, ax = plt.subplots()
    ax.set_title('Distance-time pulse evolution (a.u.)')
    t=sim.t
    z = fiber.zlocs_array 
    T, Z = np.meshgrid(t, z)
    P=getPower(matrix[:,:])/np.max(getPower(matrix[:,:]))
    P[P<1e-100]=1e-100
    surf=ax.contourf(T, Z, P,levels=40)
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Distance [km]')
    cbar=fig.colorbar(surf, ax=ax)
    ax.set_xlim(left=-5*duration)
    ax.set_xlim(right=5*duration)
    savePlot('distance-time pulse evolution')
    plt.show()

def plotSpectrumMatrix2D(matrix,fiber:Fiber_config,sim:SIM_config,FWHM_frequency_final):
    frequency0=sim.frequency0
    fig, ax = plt.subplots()
    ax.set_title('Distance-spectrum evolution')
    f=sim.f_rel
    z = fiber.zlocs_array
    F, Z = np.meshgrid(f, z)
    Pf=getPower(matrix[:,:])/np.max(getPower(matrix[:,:]))
    Pf[Pf<1e-100]=1e-100
    surf=ax.contourf(F, Z, Pf,levels=40)
    ax.set_xlabel('Frequency [PHz]')
    ax.set_ylabel('Distance [km]')
    ax.set_xlim(left=frequency0 - 5*FWHM_frequency_final)
    ax.set_xlim(right=frequency0 + 5*FWHM_frequency_final)
    cbar=fig.colorbar(surf, ax=ax)
    savePlot('distance-spectrum evolution') 
    plt.show()

def plotPSDwavelength(matrix,sim:SIM_config):
    wavelength=sim.wavelength*1e-1
    wavelength0=sim.wavelength0
    power=getPower(matrix[0,:])*2*pi*speed_of_light/(wavelength**2)
    maximum_power=np.max(power)
    plt.plot(wavelength,(getPower(matrix[0,:])*2*pi*speed_of_light/(wavelength**2))/maximum_power,label="Initial Spectrum")
    plt.plot(wavelength,(getPower(matrix[-1,:])*2*pi*speed_of_light/wavelength**2)/maximum_power,label="Final Spectrum")
    plt.title('Power spectral density as function of the wavelength')
    plt.axis([0,wavelength0*3,0,1])
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Power spectral density [arbitrary unit]")
    plt.legend()
    savePlot('power spectral density in function of wavelength')
    plt.show()

def plotSpectrogram(sim_config:SIM_config, pulse, nrange_pulse, nrange_spectrum, label=None):
    t=sim_config.t
    fc=sim_config.frequency0
    f = sim_config.f_rel
    Nmin_pulse = np.max([int(sim_config.number_of_points / 2 - nrange_pulse), 0])
    Nmax_pulse = np.min([int(sim_config.number_of_points / 2 + nrange_pulse),sim_config.number_of_points - 1,])
    Nmin_spectrum = np.max([int(sim_config.number_of_points / 2 - nrange_spectrum), 0])
    Nmax_spectrum = np.min([int(sim_config.number_of_points / 2 + nrange_spectrum),sim_config.number_of_points - 1,])
    t=t=sim_config.t[Nmin_pulse:Nmax_pulse]
    pulse = pulse[Nmin_pulse:Nmax_pulse]
    f_rel = sim_config.f_rel[Nmin_spectrum:Nmax_spectrum]
    result_matrix = np.zeros((len(f_rel),len(t)))*1j
    for idx, f_Hz in enumerate(f_rel):
        current_wavelet = lambda time: wavelet(time,sim_config.time_step,f_Hz)
        result_matrix[idx,:] = signal.fftconvolve(current_wavelet(t), pulse, mode='same')
    Z = np.abs(result_matrix) ** 2
    Z /= np.max(Z)
    fig, ax = plt.subplots(dpi=300)
    ax.set_title('Wavelet transform (spectrogram) of the %s pulse'%(label))
    T, F = np.meshgrid(t, f[Nmin_spectrum:Nmax_spectrum])
    surf = ax.contourf(T, F, Z , levels=40)
    ax.set_xlabel(f"Time [fs]")
    ax.set_ylabel(f"Frequency [pHz]")
    tkw = dict(size=4, width=1.5)
    #ax.yaxis.label.set_color('b')
    n_ticks = len(ax.get_yticklabels())-2
    norm=plt.Normalize(0,1)
    cbar = fig.colorbar(surf, ax=ax)
    text='spectrogram of the ' + label + ' pulse'
    savePlot(text) 
    plt.show()