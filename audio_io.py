from pathlib import Path
from math import gcd
from scipy.signal import resample_poly

import soundfile as sf
import numpy as np

def load_audio(path: str | Path, target_fs: int | None = None, mono: bool = True):
    
    x, fs = sf.read(str(path), dtype = np.float64, always_2d = False)
    x = np.asarray(x, dtype=np.float64)

    if mono and x.ndim > 1:
        x = np.mean(x, axis = 1) # Takes the mean of both channels and mixes it down to mono.

    if target_fs is not None and fs  != target_fs:
        x = resample_audio(x, fs, target_fs)
        fs = target_fs

    return x, fs

def save_audio(path: str | Path, signal: np.ndarray, fs: int):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True) #parents creates all missing parent directories, exist_ok to stop throwing errors if it already exists
    sf.write(str(path), signal, fs)

def resample_audio(x: np.ndarray, fs: int, target_fs: int) -> np.ndarray:
    if fs == target_fs:
        return x.copy()
    
    g = gcd(fs, target_fs) # Greatest common divisor is taken to save computing power
    up = target_fs // g # // signifies floor division
    down = fs // g

    if x.ndim == 1: # if signal is mono
        return resample_poly(x, up, down).astype(np.float64) # We use polyphase resampling because fft resampling introduces artifacts
    
    channels = []
    for ch in range(x.shape[1]): # x.shape[1] loops through each of the channels. So it loops twice if stereo.
        channels.append(resample_poly(x[:, ch], up, down)) # resampling one channel at a time. Output would be channels = [resampled_ch0, resampled_ch1], which is a list
    return np.stack(channels, axis = 1).astype(np.float64) # Converts the list into a matrix. 

def normalise_for_saving(x: np.ndarray, peak: float = 0.999) -> np.ndarray:
    max_val = np.max(np.abs(x))

    if max_val < 1e-12: # If signal is practically 0, we have to avoid dividing by 0. Just return the original signal.
        return x.copy()
        
    return (x/max_val) * peak
    
def check_clipping(x: np.ndarray, thresh: float = 0.999) -> bool:
    return bool(np.any(np.abs(x) >= thresh))      



    


    


