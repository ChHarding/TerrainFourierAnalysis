# Functions and modules needed later
import pandas as pd
import numpy as np
from math import ceil, log, sqrt
from scipy.signal import lfilter, correlate
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt

np.set_printoptions(suppress="True") # no scientific notation BS!

# TODO/Idea: taper each profile and glue together instead of averaging

def avg_amplitude(ar2D):
    '''Average amplitude (and sample distances) of multiple profiles 

    In:2D array with x and y for each profile. 
    
    Example:

    0	        45.8051	0	        44.3079	0	        46.5567	0	        43.3852	0           45.8522
    0.999646378	45.9988	0.999646378	44.3802	0.999646378	46.8987	0.999646381	43.4725	0.999646378	46.086
    1.99929276	46.292	1.99929276	44.3912	1.999292758	47.1472	1.99929276	43.6379	1.999292758	46.3454
    2.998939136	46.5865	2.998939138	44.4444	2.998939135	47.2781	2.998939138	43.8029	2.998939135	46.6047
    3.998585513	46.8623	3.998585513	44.5189	3.998585513	47.5032	3.998585514	43.932	3.998585513	46.8179
    4.998231894	47.0579	4.998231894	44.5927	4.998231894	47.5503	4.998231894	43.9986	4.998231894	46.8762
    5.997878273	47.3064	5.997878273	44.6927	5.997878273	47.5814	5.997878273	44.0973	5.997878273	46.9156
    6.997524651	47.4657	6.997524651	44.7955	6.997524649	47.6474	6.997524651	44.2244	6.997524649	47.0666
    7.99717103	47.6274	7.997171032	44.9234	7.99717103	47.6773	7.997171032	44.3314	7.99717103	47.145
    8.996817409	48.0722	8.996817409	45.0169	8.996817409	47.6545	8.996817409	44.4845	8.996817405	47.22
    9.996463787	48.1629	9.996463787	45.0775	9.996463787	47.9896	9.996463787	44.6547	9.996463787	47.2566
    10.99611017	48.3331	10.99611017	45.1808	10.99611017	48.1158	10.99611017	44.7278	10.99611017	47.2998
    
    Returns avg of all x cols and of all y cols
    '''
    _, num_cols = ar2D.shape

    xidx = range(0, num_cols, 2)
    yidx = range(1, num_cols, 2)

    xcols = ar2D[:, xidx]
    ycols = ar2D[:, yidx]

    x_mn = xcols.mean(axis=1)
    y_mn = ycols.mean(axis=1)

    # subtract mean of y from y (de-meaning aka remove trend order 0)
    avg_y = np.mean(y_mn)
    y_mn = y_mn - avg_y

    return x_mn, y_mn


def get_profiles(ar2D):
    '''Return lists of arrays for x and y of profiles'''
    _, num_cols = ar2D.shape
    xidx = range(0, num_cols, 2)
    yidx = range(1, num_cols, 2)

    xcols = ar2D[:, xidx]
    ycols = ar2D[:, yidx]

    x_lst = []
    y_lst = []

    # .T => transpose so we can iterate over columns
    for x in xcols.T: x_lst.append(x)
    for y in ycols.T: y_lst.append(y)

    return x_lst, y_lst

def plot_trends(x,y, deg_lst, name):
    '''for the single (averaged) profile in x/y, plot all trend orders given in deg_list. 
       x and y are list-like 
       deg_list must have at least one trend order, order must be > 0
       name is the name of the profile (used only for plot title
       plot Also shows variance captured by trend
       '''
    fig1 = plt.figure(figsize=(10,5)) 
    plt.title(name + ": Averaged profile (mean centered) and trends with captured variance")                                                                                          
    ax1 = plt.subplot(label="trends") # to silence depreciation warning
    ax1.plot(x, y, c='Black', label="Avg. profile")  # raw data 

    for deg in deg_lst:
        coefs = np.polynomial.polynomial.polyfit(x, y, deg)
        ffit =  np.polynomial.polynomial.Polynomial(coefs)   
        res = y - ffit(x)  # residual

        # how much of the total variance does the trend capture?
        var_cap = 1 - np.var(res, ddof=1) / np.var(y, ddof=1) 
        
        plt.plot(x, ffit(x), lw=0.5, label=f"Trend order {deg}, {round(var_cap, 4)}%") # trend along x

    ax1.legend()
    plt.show()

    # TODO: return plot instead?

def remove_trend(x_lst, y_lst, fname, deg,  plot=False):
    '''Given a list of profiles, fit a trend of order deg (0,1,2,3, ..) through each and return as list
    of residuals. 
    Input MUST be lists of list-likes
    if plot is True, also plot residuals
    '''
    if plot:
        fig1 = plt.figure(figsize=(10, 5))
        ax1 = plt.subplot(label="residuals") # to silence depreciation warning 
        plt.title(fname + ": Profiles (top) and their de-trended residuals (bottom)")                                                                                          
        colors = ["r", "c", "m", "y", "k", "b", "g"] * 10
    res_lst = [] # list of residuals
    var_cap_lst = [] # captured variance

    i = 0
    for x,y in zip(x_lst, y_lst):
                                                                                           
        if plot: ax1.plot(x, y, c=colors[i], lw=0.5, label=f"profile {i}") 
        coefs = np.polynomial.polynomial.polyfit(x, y, deg)
        ffit =  np.polynomial.polynomial.Polynomial(coefs)   
        
        res = y - ffit(x)  # residual
        if plot: plt.plot(x, res, c=colors[i], lw=0.5)

        # how much of the total variance does the trend capture?
        var_cap = 1 - np.var(res, ddof=1) / np.var(y, ddof=1) 
        #print(i, deg, var_cap)
        i += 1
        var_cap_lst.append(var_cap)
        res_lst.append(res)
        
    if plot: 
        ax1.legend()
        plt.show()
    return res_lst, var_cap_lst

def fft1d_pgram(x, y, smoothwin):
    '''return 1D periodogram (and frequency bins) for wave amplitudes y at sample distances x
    will be tapered and padded to next power of 2
    smoothwin: window size for mean smoothing pgram (only > 1 has an effect) 
    returns:  P: array with mean square Amplitude (m2)
              y: array with frequencies of the bins for P
                    
    '''

    # calculate next power of 2 greater than length(t) to pad with zeros
    twopower=2**(ceil(log(len(x))/log(2))) # next power of 2 given len of x  
    y = np.append(y, np.zeros(twopower-len(y))) # pad with 0s

    dt = np.ptp(x)/(len(x)-1)  # ptp = peak to peak = range
    dfreq = 1/(dt*twopower)
    nyquist = dfreq*(twopower/2)
    
    # TODO: use rfft (real only):https://mark-kramer.github.io/Case-Studies-Python/03.html

    w = np.hanning(twopower) # Hanning window
    tapered = w * y # apply window to get a tapered array
    Wss = sqrt(2) * sum(w)**2 # to correct for the reduction of amplitude by the windowing
    

    # TODO: apply padding AFTER taper! https://www.ocean.washington.edu/courses/ess522/lectures/07_taperingandpractical.pdf

    myfft = np.fft.fft(tapered, twopower)
    Y = np.fft.fftshift(myfft)
    P = Y * np.conj(Y) / Wss    # DFT periodogram - with this normalization, 
                                # the sum over all elements of the spectral
                                # power array equals the variance of the data

    P = np.real(P) # only need real part (amplitude) of the complex array, don't need phase

    start = ceil(twopower / 2)
    P = 2 * P[start:]  # Take only first half of spectrum (2nd half is
                       # redundant). Factor of 2 corrects for this.
                       # Last element corresponds to one frequency bin less than the nyquist frequency,
                       # first element to zero frequency (DC) (=data mean)
                             
    f = np.arange(0, nyquist, dfreq) # frequency bin boundaries

    # Chop off DC component (leftmost bin)
    f=f[1:]
    P=P[1:]

    if smoothwin > 1:
        # smooth spectrum (in log space) with a running mean
        P_filt_log10 = lfilter(np.ones(smoothwin) / smoothwin, 1, np.log10(P))
        P = 10 ** P_filt_log10 # un-log10

    return P, f 

def fft1d_theor_signif(P,f, x,y, siglvl, smoothwin):
    ''' For a peridogram P with freq bins f, made from amplitudes y over x for a significance level (0.0 - 1.0) 
    and window smoothing width (0,1,2,...) , return:
    fft_theor_sc: a line (for freq bins) showing the Theoretical red-noise power spectrum  
    signif: a line (for freq bins) for the significance level. Where P plots above that line, it is significant (as per the given level)
    lag1: the highest auto correlation (not sure if needed outside of this function ...)
    '''

    dt = np.ptp(x)/(len(x)-1)  # ptp = peak to peak = range
    twopower = len(y)
    dfreq = 1/(dt*twopower)
    nyquist = dfreq*(twopower/2)

    # time series variance, after detrending and padding
    variance = np.std(y, ddof=1) ** 2 # Note: default ddof for numpy is 0 but 1 for matlab

    # SIGNIFICANCE LEVELS
    auto_corr = correlate(y, y,  mode='full', method='fft'); # auto correlation
    maxlag = np.max(auto_corr) # max correlation TODO: using max() here instead of an index as matlab does - CHECK
    auto_corr_norm = auto_corr / maxlag # normalize (0.0 - 1.0) with max lag value

    # TODO: different than matlab - check this!
    center_lag = len(auto_corr_norm) // 2 # index for lag with max norm corr (always 1.0?)
    lag1 = auto_corr_norm[center_lag-1] # lag-1 autocorrelation i.e. next highest (non 1.0) normalized correlation

    # Theoretical red-noise power spectrum from Eqn(16) in Torrence & Compo 1998
    freq = dt*f  # normalized frequency
    A = 1 - lag1**2
    B = 1 - 2 * lag1 * np.cos(np.pi * freq / nyquist) + lag1**2
    fft_theor =  A / B 

    # Perron Matlab code comment: Made this correction 5/8/10 after re-reading T&C98
    fft_theor_sc = fft_theor * variance / sum(fft_theor)  # scale with time-series variance. Note that this expression only applies to the DFT periodogram.

    dof = 2*smoothwin; # degrees of freedom, see Torrence & Compo 1998. If the Fourier coefficients are normally distributed, then the power (squared Fourier coefficients) will have 2 DOF.
            # If the spectrum is smoothed by averaging M adjacent frequencies
            # together, each of which has 2 DOF, the DOF for each frequency in
            # the smoothed spectrum is 2*M. Hence the multiplication by
            # smoothwin. In general, the significance level will be more on
            # the mark if the spectrum is smoothed over a larger number of
            # frequencies--that is, the fraction of the spectrum that exceeds,
            # e.g., the 0.05 significance level will be closer to 0.05.
    chisquare = chi2.ppf(0.95, df=dof) / dof # Note: Not using the Torrence & Compo 1998 routines from matlab
    signif = fft_theor_sc * chisquare #Eqn(18) in Torrence & Compo 1998

    return fft_theor_sc, signif 

# tests
if __name__ == "__main__":\

    filename = 'S09_5p.txt'

    # read in tab separated txt file, no header
    df = pd.read_csv(filename, sep="\t", header=None) # tab separated txt file
    print(filename, ":", len(df), "samples,", len(list(df)) // 2, "profiles")
    ar2D = df.to_numpy()

    # put x and y columns into lists 
    x_lst, y_lst = get_profiles(ar2D)

    # plot trends
    x, y = avg_amplitude(ar2D) # averages profiles into a single profile
    trend_list = [1, 2, 3] 
    plot_trends(x, y, trend_list, filename);
    
    # De-trend and store residuals for each profile. 
    res_lst, var_last = remove_trend(x_lst, y_lst, filename, deg=3, plot=True)

