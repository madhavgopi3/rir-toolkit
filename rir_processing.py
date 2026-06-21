import numpy as np
from scipy.signal import hilbert


def normalise_rir(h: np.ndarray, peak: float = 0.999) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64).squeeze()

    max_val = np.max(np.abs(h))
    
    if max_val < 1e-12:
        return h.copy()

    return (h / max_val) * peak


def trim_rir_robust(
        x: np.ndarray,
        fs: int,
        pre_ms: float,
        min_tail_ms: float,
        threshold_over_noise_db: float,
        arrival_smooth_ms: float,
        tail_smooth_ms: float,
        safety_offset_ms: float,
) -> tuple[np.ndarray, int, int, int, np.ndarray]:

    x = np.asarray(x, dtype=np.float64).squeeze()

    # Estimate noise from the end of the RIR. tail_fraction specifies the fraction of signal for consideration.
    
    tail_fraction = 0.1
    n = len(x)
    tail_samples = max(1, int(tail_fraction * n))
    tail = x[-tail_samples:]

    noise_rms = np.sqrt(np.mean(tail ** 2) + 1e-12)
    noise_db = 20 * np.log10(noise_rms + 1e-12)

    threshold_db = noise_db + threshold_over_noise_db

    # Hilbert transform + call smoothing
    # smooth_ms = Smoothing window in milliseconds
    # We make a moving average filter. Light filtering because arrival_smooth_ms = 1ms. Higher value, heavy smoothing.
    hilb_sig = hilbert(x)
    envelope = np.abs(hilb_sig)

    arrival_window_samples = max(1, int(arrival_smooth_ms / 1000 * fs))
    kernel = np.ones(arrival_window_samples, dtype=np.float64) / arrival_window_samples
    envelope_smooth = np.convolve(envelope, kernel, mode="same")

    env_db = 20 * np.log10(envelope_smooth + 1e-12)

    # Find the peak. Backtrack by 20ms. This is search_start
    # Find where the signal first crosses threshold_db from here. This is onset_idx.
    # refine_end is onset_idx + 2 ms. And search within this 2ms window to find the peak with argmax.
    global_peak_idx = int(np.argmax(envelope_smooth))

    backtrack_ms = 20.0
    backtrack_samples = max(1, int((backtrack_ms / 1000.0) * fs)) #Instead of searching the whole signal, we backtrack and search 20 ms from the peak.
    search_start = max(0, global_peak_idx - backtrack_samples)

    candidates = np.where(env_db[search_start:global_peak_idx + 1] > threshold_db)[0]

    if len(candidates) == 0:
        peak_idx = global_peak_idx
    else:
        onset_idx = search_start + int(candidates[0])

        refine_end = min(len(envelope_smooth), onset_idx + max(1, int((2.0 / 1000.0) * fs)))
        local_peak_idx = int(np.argmax(envelope_smooth[onset_idx:refine_end]))

        peak_idx = onset_idx + local_peak_idx

    # Find the end where the envelope falls back near the noise floor.

    # SEARCH AREA: The place from the required rir peak + we add a random 300ms area. peak_idx + min_tail_samples
    # CANDIDATES: Indices in SEARCH AREA where the value falls below noise_db
    # end_idx = Place where we think the noise floor is hit + safety_offset_samples

    # Computing another envelope for tail analysis. This one is much smoother cuz of higher smooth_ms value.
    tail_hilb_sig = hilbert(x)
    tail_envelope = np.abs(tail_hilb_sig)

    tail_window_samples = max(1, int(tail_smooth_ms / 1000 * fs))
    tail_kernel = np.ones(tail_window_samples, dtype=np.float64) / tail_window_samples
    tail_envelope_smooth = np.convolve(tail_envelope, tail_kernel, mode="same")

    tail_envelope_db = 20 * np.log10(tail_envelope_smooth + 1e-12)

    min_tail_samples = max(1, int((min_tail_ms / 1000) * fs))
    search_start_idx = min(len(x) - 1, peak_idx + min_tail_samples) # Peak of rir + we add a buffer from where to start searching for the noise

    end_candidates = np.where(tail_envelope_db[search_start_idx:] <= noise_db)[0] # Filter out the indices where rir falls below noise floor.

    if len(end_candidates) == 0: # If there is no noise tail at the end of rir
        end_idx = len(x) # Returns the last index of the rir itself
    else:
        # A small offset to ensure the tail is not cut aggressively.
        safety_offset_samples = int((safety_offset_ms / 1000) * fs)
        end_idx = min(len(x), search_start_idx + end_candidates[0] + safety_offset_samples)

    pre_samples = int((pre_ms / 1000) * fs)
    start_idx = max(0, peak_idx - pre_samples)

    #Safety check if something goes wrong
    if end_idx <= start_idx:
        end_idx = min(len(x), peak_idx + int((min_tail_ms / 1000.0) * fs)) # keep min_tail_ms of the RIR after peak

    trimmed = x[start_idx:end_idx]

    return trimmed, start_idx, end_idx, peak_idx, tail_envelope_smooth

