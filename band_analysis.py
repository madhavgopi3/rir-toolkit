import numpy as np
from scipy.signal import butter, sosfiltfilt

def octave_edges(center_freq, fraction):
    """
    Returns lower and upper frequency limits for a fractional-octave band.

    fraction = 1 is octave
    fraction = 3 is third-octave
    fraction = 6 is sixth-octave
    """
    factor = 2.0 ** (1.0 / (2.0 * fraction)) 

    f_low = center_freq / factor
    f_high = center_freq * factor

    return f_low, f_high

# Used for freq response band values
def freqband_average_db(freqs: np.ndarray, magnitude_db: np.ndarray, center_freq: float, fraction: int):

    f_low, f_high = octave_edges(center_freq, fraction)

    mask = (freqs >= f_low) & (freqs <= f_high)

    linear_power = 10 ** (magnitude_db[mask] / 10.0) # We cannot average in dB, so converting to linear
    return 10 * np.log10(np.mean(linear_power) + 1e-30) # Returning in dB


# Used for descriptors' band analysis
def bandpass_rir(rir, fs, center_freq, fraction, order=6):

    h = np.asarray(rir, dtype=np.float64).squeeze()
    f_low, f_high = octave_edges(center_freq, fraction)

    # Guard the upper edge against Nyquist for the highest bands.
    nyq = fs / 2.0
    f_high = min(f_high, 0.999 * nyq)

    # sos: second-order sections (numerically stable for higher orders)
    sos = butter(order, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, h)  # filter forward and backward, avoids shifting the rir in time
