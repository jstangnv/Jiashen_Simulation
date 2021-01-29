from nvsys import *

def sinusoid(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def expdecay_sinusoid(x, A, B, C, D, E):
    return A * np.exp(-E*x) * np.sin(B * x + C) + D

def two_sines(x, A1, A2, frq1, p1, C):
    return A1*np.cos(frq1*x + p1) + A2*np.cos(.5*frq1*x + .5*p1) + C

def get_frq(xdata, ydata, par_guess=None, bounds=None):
    if par_guess is None:
        par_guess=[np.amax(ydata) - np.amin(ydata), 1, 0, np.median(ydata), 0]
    if bounds is None:
        bounds=((0, 0, -2*np.pi, np.amin(ydata), -np.inf), ((np.amax(ydata) - np.amin(ydata))*5, np.inf, 2*np.pi, np.amax(ydata), np.inf))
    params, params_covariance = optimize.curve_fit(expdecay_sinusoid, xdata, ydata, p0=par_guess, bounds=bounds, maxfev=100000, loss='soft_l1')
    frq_fit = params[1]/2/np.pi
    return frq_fit, params, np.sqrt(np.diag(params_covariance))

def fit2harmonics(xdata, ydata, par_guess=None, bounds=None):
    maxminusmin = np.amax(ydata) - np.amin(ydata)
    if par_guess is None:
        par_guess=[maxminusmin, 0, 1, 0, np.median(ydata)]
    if bounds is None:
        bounds=((0, 0, 0, -np.pi, np.amin(ydata)), (maxminusmin*5, maxminusmin*5, np.inf, np.pi, np.amax(ydata)))
    params, params_covariance = optimize.curve_fit(two_sines, xdata, ydata, p0=par_guess, bounds=bounds, maxfev=100000)
    return params, np.sqrt(np.diag(params_covariance))


def get_fft(t, data, cutoff_frq=0.1, mdn_width=0):
    power_spect = np.abs(np.fft.fft(data))
    freq = np.fft.fftfreq(data.size, d=t[1]-t[0])
    freqstep = freq[1]- freq[0]
    
    bools_frq = freq > cutoff_frq
    freq = freq[bools_frq]
    if mdn_width:
        power_spect = scs.medfilt(power_spect[bools_frq], mdn_width)
    else:
        power_spect = power_spect[bools_frq]
    return freq, power_spect
