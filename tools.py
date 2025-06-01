import numpy as np
import matplotlib.pylab as plt
import scipy.signal as sps

from .bmx import finding_Index_Time
from .bmx_data import load_data
from .spectrum_wwind import spectrum_wwind


def get_windowed_b(pos, shot, window):
    """Output the windowed b-vector of shot at pos"""
    br, bt, bz, time = load_data(pos, shot)
    start_dex = finding_Index_Time(time * 1e-6, window[0])
    end_dex = finding_Index_Time(time * 1e-6, window[1])

    wbr, wbt, wbz = br[start_dex:end_dex], bt[start_dex:end_dex], bz[start_dex:end_dex]
    time = time[start_dex:end_dex]

    return wbr, wbt, wbz, time


def get_windowed_bmag(pos, shot, window):
    """Output the windowed bmag and time arrays of {shot} at {pos}."""
    bmag, time = get_bmag(pos, shot)

    start_dex = finding_Index_Time(time * 1e-6, window[0])
    end_dex = finding_Index_Time(time * 1e-6, window[1])

    wbmag = bmag[start_dex:end_dex]
    time = time[start_dex:end_dex]

    return wbmag, time


def get_windowed_filtered_bmag(pos, shot, window, filt):
    """Output the windowed filtered bmag and time arrays of {shot} at {pos}."""
    filt_bmag, time = get_filtered_bmag(pos, shot, filt)

    start_dex = finding_Index_Time(time * 1e-6, window[0])
    end_dex = finding_Index_Time(time * 1e-6, window[1])

    wfilt_bmag = filt_bmag[start_dex:end_dex]
    time = time[start_dex:end_dex]

    return wfilt_bmag, time


def get_filtered_bmag(pos, shot, filt):
    """Output the filtered bmag and time arrays of {shot} at {pos}."""
    bmag, time = get_bmag(pos, shot)
    filt_bmag = HPF(bmag, filt)
    return filt_bmag, time


def get_bmag(pos, shot):
    """Output the bmag and time arrays of shot at pos."""
    br, bt, bz, time = load_data(pos, shot)
    bmag = np.sqrt(br**2 + bt**2 + bz**2)
    return bmag, time


def HPF(data: np.ndarray, filter_Freq: float, fs=125e6, N=4) -> np.ndarray:
    """Outputs the filtered data"""
    Wn = 2.0 * filter_Freq / fs
    B, A = sps.butter(N, Wn, btype="highpass")
    output = sps.filtfilt(B, A, data)
    return output


def correlation(sing_1, sing_2, time, normalized=True, mode="same"):
    """Normalization based on the number of point considered
    The normalization parameter normalizes the correlation coefficient by the
    sample size of the overlapping signals minus 1. Whereas, if the normalization
    parameter is set to false, the correlation coefficients are all normalized by
    the sample size of the entire signal.
    """

    corr = np.correlate(sing_2, sing_1, mode=mode)
    dt = time[1] - time[0]
    tau = dt * (np.arange(corr.size) - corr.size / 2)

    if normalized:

        time_Window = time[-1] - time[0]
        norma_factor = (time_Window - np.abs(tau)) / dt
        corr = corr / (norma_factor - 1)
    else:
        corr = corr / (time.size - 1)

    return corr, tau


def correlationv2(sing_1, sing_2, time, normalized=True, mode="same", Tref=6):
    """Different Normalization based on the number of point considered"""

    corr = np.correlate(sing_1, sing_2, mode=mode)
    dt = time[1] - time[0]
    tau = dt * (np.arange(corr.size) - corr.size / 2)
    Nref = Tref / dt
    if normalized:
        normalization = np.zeros(tau.shape)
        time_Window = time[-1] - time[0]
        number = np.abs(tau) / dt
        ind_loc = np.where(number > Nref)[0]
        number[ind_loc] = Nref
        norma_factor = time_Window / dt - number
        normalization = 1 / (norma_factor - 1)
    else:
        normalization = 1

    corr = normalization * corr

    return corr, tau


def plot_components(pos: int, shot: int):
    """Plot the raw magnetic data."""
    x, y, z, time = load_data(pos, shot)
    fig = plt.figure()
    fig.suptitle("Raw Fields")
    gs = fig.add_gridspec(3, 1)
    axx = fig.add_subplot(gs[0, 0])
    axy = fig.add_subplot(gs[1, 0])
    axz = fig.add_subplot(gs[2, 0])

    axx.plot(time, x)
    axy.plot(time, y)
    axz.plot(time, z)

    plt.show()
    plt.close()

    return None


def restore_log(x, y):
    """Transform the spectral index into a log trend.
    Outputs f, slope, y_int
    """
    z = spectral_index(x, y)

    slope = z[0]
    y_int = z[1]

    f = 10 ** (y_int) * x ** (slope)
    return f, slope, y_int


def spectral_index(x, y):
    """Obtain the spectral index of given data."""
    xlog = np.log(x)
    ylog = np.log(y)

    z = np.polyfit(xlog, ylog, 1)

    return z


def gen_separation_array(refPos, max_probe_number=8):
    """Outputs the separation array w.r.t the reference probe"""
    range = gen_max_probe_range(refPos, max_probe_number=max_probe_number)
    separation = np.arange(0, range) * 2.6

    return separation


def gen_max_probe_range(refPos, max_probe_number=8):
    """The probe number range"""
    startProbeNumber = (refPos - 1) / 2
    range = max_probe_number - startProbeNumber
    return int(range)


def get_pos_index(refPos, pos):
    positions = np.arange(refPos, 16, 2)
    indices = np.where(positions == pos)
    posIndex = indices[0][0]
    return posIndex


def get_power_spectrum(data, time, alias="hanning"):
    """Output the power spectrum and frequency"""

    freq, _, _, pwr, pwr_den, _, _, _, _ = spectrum_wwind(
        data, time * 1e-6, window=alias
    )

    return freq, pwr, pwr_den
