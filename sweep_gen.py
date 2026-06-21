import numpy as np

def generate_log_sweep(
        fs: int,
        duration: float,
        f_start: int,
        f_end: int,
        amplitude: float = 0.8
) -> np.ndarray:
    
    t = np.linspace(0.0, duration, int(fs * duration), endpoint = False)

    #Farina's exponential sine sweep

    L = duration/np.log(f_end/f_start)
    phase = 2.0 * np.pi * f_start * L * (np.exp(t/L) - 1.0)
    sweep = amplitude * np.sin(phase)

    return sweep.astype(np.float64)

def generate_inverse_filter(
        sweep: np.ndarray,
        fs: int,
        f_start: int,
        f_end: int
) -> np.ndarray:

    n = len(sweep)
    duration = n / fs
    t = np.linspace(0.0, duration, n, endpoint= False)

    #Amplitude correction for exponential sweep filter
    envelope = np.exp(t * np.log(f_end/f_start)/duration)
    inverse = sweep[::-1]/envelope

    return inverse.astype(np.float64)

# Pad signals with pre defined values of pre and post silences.
def pad_signal(
        signal: np.ndarray,
        fs: int,
        pre_silence: float,
        post_silence: float
) -> np.ndarray:
    
    pre = np.zeros(int(pre_silence*fs), dtype = np.float64)
    post = np.zeros(int(post_silence*fs), dtype = np.float64)
    
    return np.concatenate([pre, signal, post])

# To normalize signal amplitude from having too high/low values
def normalise_peak(
        signal: np.ndarray, 
        peak: float = 0.999) -> np.ndarray: # We use .999 instead of 1 to give a safe headroom.
    max_val = np.max(np.abs(signal))
    if max_val < 1e-12: # To avoid dividing by 0. Below 1e-12 = effectively silence.
        return signal.copy()
    return (signal / max_val) * peak
