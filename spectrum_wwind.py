# compute FFT of given array

import numpy as np
from scipy.signal.windows import blackman, bartlett, hann, hamming, kaiser


def spectrum_wwind(array: np.ndarray, time: np.ndarray, window="hanning") -> np.ndarray:
    """
    Inputs:
        array,
        time,
        window = "hanning"
    Outputs:
        freq,
        freq2,
        comp,
        pwr,
        pwr_den,
        mag,
        phase2,
        cos_phase,
        dt
    """
    # time should be in seconds
    # Size of array
    Nw = array.shape[0]

    # Calculate time step (assumed to be in seconds)
    dt = time[1] - time[0]

    # Calculate array of frequencies, shift
    w = np.fft.fftfreq(Nw, dt)
    w0 = np.fft.fftshift(w)

    # make window
    # blackman window
    if window == "blackman":
        bwin = blackman(Nw)  # pretty good
    if window == "hanning":
        bwin = hann(Nw)  # pretty good
        S1 = np.sum(bwin)
        S2 = np.sum(bwin**2)
    if window == "hamming":
        bwin = hamming(Nw)  # not as good
    if window == "bartlett":
        bwin = bartlett(Nw)  # pretty good
    if window == "kaiser":
        bwin = kaiser(Nw, 6)
    if window == "None":
        bwin = 1.0

    # Calculate FFT
    # aw = prefactor*np.fft.fft(array*bwin)
    aw = np.fft.fft(array * bwin)
    aw0 = np.fft.fftshift(aw)

    # Calcuate Phase
    phase = np.angle(aw)
    phase0 = np.fft.fftshift(phase)

    # Adjust arrays if not div by 2
    if not np.mod(Nw, 2):
        w0 = np.append(w0, -w0[0])
        aw0 = np.append(aw0, -aw0[0])
        phase0 = np.append(phase0, -phase0[0])

    # Cut FFTs in half
    Nwi = Nw // 2
    w2 = w0[Nwi:]
    aw2 = aw0[Nwi:]
    phase2 = phase0[Nwi:]

    comp = aw
    pwr = (
        2 / S1**2 * (np.abs(aw2)) ** 2
    )  # This provides the power at every frequency bin
    pwr_den = (
        2 * dt * (np.abs(aw2)) ** 2 / S2
    )  # This provides the power/frequency at every frequency bin

    mag = np.sqrt(pwr)
    cos_phase = np.cos(phase2)
    freq = w2
    freq2 = w

    return freq, freq2, comp, pwr, pwr_den, mag, phase2, cos_phase, dt
