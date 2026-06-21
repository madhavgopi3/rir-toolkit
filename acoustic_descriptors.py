import numpy as np
from scipy.signal import hilbert

EPS = 1e-30 # Epsilon


def _as_1d_float(x):
    return np.asarray(x, dtype=np.float64).squeeze()


def _power_to_db(x):
    return 10.0 * np.log10(np.maximum(x, EPS))


def find_direct_sound_index(rir, fs, search_ms=20.0, peak_drop_db=20.0):
    """
    Locate the start of the impulse response (t = 0).

    The onset point is the point before the peak of the direct
    sound, where the impulse response first rises above (peak - 20 dB). We work
    on a lightly smoothed Hilbert envelope so a non-ideal source (which is not a
    perfect Dirac) does not cause a spurious early trigger.
    """
    h = _as_1d_float(rir)

    if len(h) == 0:
        return 0

    envelope = np.abs(hilbert(h))

    smooth_samples = max(1, int(0.001 * fs))  # 1 ms smoothing
    kernel = np.ones(smooth_samples, dtype=np.float64) / smooth_samples
    envelope = np.convolve(envelope, kernel, mode="same")

    peak_index = int(np.argmax(envelope))
    peak_val = float(envelope[peak_index])

    if peak_val <= 0.0:
        return peak_index

    # -20 dB relative to the peak, in amplitude terms (10 ** (-20/20) = 0.1).
    threshold = peak_val * (10.0 ** (-peak_drop_db / 20.0))

    # Search backwards from the peak, but no further than search_ms, so distant
    # background noise before the arrival cannot trigger the onset.
    search_samples = max(1, int((search_ms / 1000.0) * fs))
    start = max(0, peak_index - search_samples)

    candidates = np.where(envelope[start:peak_index + 1] >= threshold)[0]

    if len(candidates) == 0:
        return peak_index

    return start + int(candidates[0])


def lundeby_knee(
    rir,
    fs,
    block_ms,
    tail_fraction,
    margin_db,
    max_iter,
):
    """
    Lundeby style knee estimate: the point where the decay meets the noise
    floor. The late, linear part of the decay is fitted (the direct sound and
    early reflections at the top of the curve are skipped).

    Returns:
        knee_sample: sample where decay meets the estimated noise floor
        noise_power: average noise power after the knee
        noise_db: noise floor in dB
    """
    h = _as_1d_float(rir)
    n = len(h)

    if n == 0:
        return 0, 0.0, -np.inf

    energy = h ** 2

    block_size = max(1, int((block_ms / 1000.0) * fs))
    num_blocks = n // block_size # floor division

    if num_blocks < 3: # if the rir is too short, return the mean energy of rir as the noise.
        noise_power = float(np.mean(energy) + EPS)
        return n, noise_power, _power_to_db(noise_power) # the last sample is considered the knee

    # compute block power
    usable = energy[:num_blocks * block_size]
    block_power = np.mean(usable.reshape(num_blocks, block_size), axis=1) # Each row will represent a block and the average of each row/block is taken.
    block_db = _power_to_db(block_power)

    # initial noise estimate
    tail_blocks = max(1, int(tail_fraction * num_blocks))
    noise_power = float(np.mean(block_power[-tail_blocks:]) + EPS)
    noise_db = _power_to_db(noise_power)

    # Find the loudest block
    peak_block = int(np.argmax(block_db))
    peak_level = float(block_db[peak_block])
    fit_headroom_db = 5.0  # skip the direct sound / early reflections at the top
    knee_block = num_blocks - 1

    for _ in range(max_iter):
        old_knee = knee_block

        # Finds blocks that are at least 5 dB (fit_headroom_db) below the peak.
        below_peak = np.where(
            block_db[peak_block:knee_block + 1] <= peak_level - fit_headroom_db
        )[0]
        if len(below_peak) == 0:
            break
        fit_start = peak_block + int(below_peak[0])

        # keeps only those blocks above noise_db + margin_db
        search = block_db[fit_start:knee_block + 1]
        valid = np.where(search > noise_db + margin_db)[0]

        if len(valid) < 2: # If fewer than two valid points exist, a line cannot be reliably fitted.
            break

        fit_end = fit_start + int(valid[-1]) # fit_end is the last valid block above the noise margin.
        if fit_end <= fit_start:
            break

        x = np.arange(fit_start, fit_end + 1, dtype=np.float64) * (block_size / fs) # time axis in seconds for each block
        y = block_db[fit_start:fit_end + 1] 

        slope, intercept = np.polyfit(x, y, 1)

        if slope >= 0:
            break

        knee_time = (noise_db - intercept) / slope
        knee_block = int(round(knee_time * fs / block_size))
        knee_block = int(np.clip(knee_block, peak_block + 1, num_blocks - 1)) # Forces the knee_block value to stay after the peak and before the end of the RIR.

        safety_blocks = max(1, int(0.050 * fs / block_size))  # 50 ms
        noise_start = min(knee_block + safety_blocks, num_blocks - 1) # Starts estimating noise 50 ms (which is safety_blocks) after the knee.

        if noise_start < num_blocks - 1:
            noise_power = float(np.mean(block_power[noise_start:]) + EPS)
            noise_db = _power_to_db(noise_power)

        if abs(knee_block - old_knee) <= 1: # If the knee changes by one block or less, the iteration is considered stable.
            break

    knee_sample = min(n, knee_block * block_size)

    # avoid adding margin_db and averages from the knee_block till the end while returning.
    if knee_sample < n - 1:
        noise_power = float(np.mean(energy[knee_sample:]) + EPS)
        noise_db = _power_to_db(noise_power)

    return knee_sample, noise_power, noise_db


def impulse_to_noise_ratio_db(rir, noise_power):
    """
    Returns peak energy of the IR over the background-noise energy (both in power), in dB.
    """
    h = _as_1d_float(rir)
    if len(h) == 0 or noise_power <= 0.0:
        return np.inf
    peak_energy = float(np.max(h ** 2))
    return 10.0 * np.log10((peak_energy + EPS) / (noise_power + EPS))


def schroeder_edc_db(
    rir,
    fs,
    noise_compensate=True,
    block_ms=10.0,
    tail_fraction=0.1,
    margin_db=10.0,
    max_iter=10,
):
    """
    Normalised Schroeder Energy Decay Curve in dB (0 dB at t = 0).

    With noise_compensate=True a Lundeby-style knee is found, the curve is
    integrated only up to the knee and the expected residual noise energy is
    subtracted.

    Returns edc_db, knee, noise_power, noise_db
    """
    h = _as_1d_float(rir)
    n = len(h)

    if n == 0:
        return np.array([], dtype=np.float64), 0, 0.0, -np.inf # If the signal is empty, returns zero knee, zero noise power, and negative infinity dB.

    energy = h ** 2

    if not noise_compensate:
        edc = np.cumsum(energy[::-1])[::-1]
        max_edc = np.max(edc)
        if max_edc <= 0:
            return np.zeros_like(h), n, 0.0, -np.inf #zeros_like is used for creating an array with the same dimensions as h, with zeros.
        edc = edc / max_edc
        return _power_to_db(edc), n, 0.0, -np.inf

    knee, noise_power, noise_db = lundeby_knee(
        h,
        fs,
        block_ms=block_ms,
        tail_fraction=tail_fraction,
        margin_db=margin_db,
        max_iter=max_iter,
    )

    knee = int(np.clip(knee, 1, n))
    useful_energy = energy[:knee]

    raw_edc = np.cumsum(useful_energy[::-1])[::-1]

    # Expected accumulated noise energy from each time sample to the knee.
    remaining_samples = np.arange(knee, 0, -1, dtype=np.float64)
    noise_correction = noise_power * remaining_samples

    edc = raw_edc - noise_correction
    edc[edc < EPS] = EPS

    full_edc = np.full(n, EPS, dtype=np.float64)
    full_edc[:knee] = edc

    full_edc = full_edc / np.max(full_edc)
    edc_db = _power_to_db(full_edc)

    return edc_db, knee, noise_power, noise_db


def decay_time_from_edc(
    edc_db,
    fs,
    upper_db,
    lower_db,
    min_points=50,
    min_r2=0.90,
    min_rt=0.03,
    max_rt=10.0,
):
    """
    Reverberation time from a linear regression of the Schroeder decay between
    upper_db and lower_db, extrapolated to a 60 dB decay (T = -60 / slope). upper and lower dbs 
    depend on the parameter calculated. Eg: -5 to -35 for t30.
    Returns NaN if the fit is non-decaying, too non-linear or has physically impossible values.
    """
    edc_db = _as_1d_float(edc_db)

    t = np.arange(len(edc_db), dtype=np.float64) / fs

    idx = np.where((edc_db <= upper_db) & (edc_db >= lower_db))[0]

    if len(idx) < min_points: # Return nan if not enough points exist for fitting. Set to a default of 50.
        return np.nan

    x = t[idx]
    y = edc_db[idx]

    slope, intercept = np.polyfit(x, y, 1)

    if not np.isfinite(slope) or slope >= 0:
        return np.nan

    y_fit = slope * x + intercept

    # Both these values are used to calculate R^2
    ss_res = np.sum((y - y_fit) ** 2) # Sum of squared errors (residuals) between actual decay and fitted line. Large ss_res means poor fit.
    ss_tot = np.sum((y - np.mean(y)) ** 2) # Total variance of the actual decay data.

    if ss_tot <= 0: # If there is no variation in data
        return np.nan

    r2 = 1.0 - ss_res / ss_tot # An R2 close to 1 means the decay is very linear.

    if r2 < min_r2:            # reject poorly-fitting (non-linear) decays
        return np.nan

    rt60 = -60.0 / slope

    if not (min_rt <= rt60 <= max_rt):   # reject physically impossible values
        return np.nan

    return rt60


def clarity_c50(rir, fs, direct_index=0, knee=None):
    h = _as_1d_float(rir)
    energy = h ** 2

    end = len(h) if knee is None else int(np.clip(knee, 1, len(h)))
    start = int(np.clip(direct_index, 0, end))
    split = min(end, start + int(0.050 * fs))

    early = np.sum(energy[start:split])
    late = np.sum(energy[split:end])

    return 10.0 * np.log10((early + EPS) / (late + EPS))


def clarity_c80(rir, fs, direct_index=0, knee=None):
    h = _as_1d_float(rir)
    energy = h ** 2

    end = len(h) if knee is None else int(np.clip(knee, 1, len(h)))
    start = int(np.clip(direct_index, 0, end))
    split = min(end, start + int(0.080 * fs))

    early = np.sum(energy[start:split])
    late = np.sum(energy[split:end])

    return 10.0 * np.log10((early + EPS) / (late + EPS))


def definition_d50(rir, fs, direct_index=0, knee=None):
    h = _as_1d_float(rir)
    energy = h ** 2

    end = len(h) if knee is None else int(np.clip(knee, 1, len(h)))
    start = int(np.clip(direct_index, 0, end))
    split = min(end, start + int(0.050 * fs))

    early = np.sum(energy[start:split])
    total = np.sum(energy[start:end])

    return early / (total + EPS)


def center_time_ts(rir, fs, direct_index=0, knee=None):
    h = _as_1d_float(rir)
    energy = h ** 2

    end = len(h) if knee is None else int(np.clip(knee, 1, len(h)))
    start = int(np.clip(direct_index, 0, end))
    useful_energy = energy[start:end]

    t = np.arange(len(useful_energy), dtype=np.float64) / fs
    ts = np.sum(t * useful_energy) / (np.sum(useful_energy) + EPS)

    return ts * 1000.0


def extract_room_descriptors(
    rir,
    fs,
    noise_compensate,
    direct_sound_search_ms,
    lundeby_block_ms,
    lundeby_tail_fraction,
    lundeby_margin_db,
    lundeby_max_iter,
    direct_index=None,
    onset_drop_db=20.0,
    inr_min_edt_db=15.0,
    inr_min_t20_db=35.0,
    inr_min_t30_db=45.0,
    rt_min_s=0.03,
    rt_max_s=10.0,
    rt_min_r2=0.90,
):
    """
    Main function that calls all the above functions and extracts room-acoustic descriptors from one RIR.

    All energy-based parameters use t = 0 at the direct-sound onset, the decay
    curve is the noise-compensated Schroeder integral, and reverberation times
    are checked by the impulse-to-noise ratio: T20 needs >= 35 dB and
    T30 needs >= 45 dB of usable decay, otherwise NaN is returned.

    Returns edt/rt20/rt30 in s, c50/c80 in dB, d50 as a ratio b/w 0 and 1,
    ts_ms in ms, plus diagnostics (onset, knee, noise floor, INR).
    """
    h = _as_1d_float(rir)
    n_full = len(h)

    if direct_index is None:
        direct_index = find_direct_sound_index(
            h, fs, search_ms=direct_sound_search_ms, peak_drop_db=onset_drop_db,
        )

        # If the detected onset is extremely close to the start, treat the RIR
        # as already trimmed to the direct sound.
        if direct_index < int(0.002 * fs):
            direct_index = 0

    direct_index = int(np.clip(direct_index, 0, max(0, n_full - 1)))

    # Everything below is referenced to t = 0 at the direct sound.
    hs = h[direct_index:]

    edc_db, knee, noise_power, noise_db = schroeder_edc_db(
        hs,
        fs,
        noise_compensate=noise_compensate,
        block_ms=lundeby_block_ms,
        tail_fraction=lundeby_tail_fraction,
        margin_db=lundeby_margin_db,
        max_iter=lundeby_max_iter,
    )

    inr_db = impulse_to_noise_ratio_db(hs, noise_power) if noise_compensate else np.inf

    edt = decay_time_from_edc(
        edc_db, fs, 0.0, -10.0, min_r2=rt_min_r2, min_rt=rt_min_s, max_rt=rt_max_s,
    )
    rt20 = decay_time_from_edc(
        edc_db, fs, -5.0, -25.0, min_r2=rt_min_r2, min_rt=rt_min_s, max_rt=rt_max_s,
    )
    rt30 = decay_time_from_edc(
        edc_db, fs, -5.0, -35.0, min_r2=rt_min_r2, min_rt=rt_min_s, max_rt=rt_max_s,
    )

    # Safety checks
    if np.isfinite(inr_db):
        if inr_db < inr_min_edt_db:
            edt = np.nan
        if inr_db < inr_min_t20_db:
            rt20 = np.nan
        if inr_db < inr_min_t30_db:
            rt30 = np.nan

    # Index up to which the compensated decay is meaningful, for plotting. The
    # curve is cut to roughly the noise floor (~INR below the start),
    # below that the noise subtraction goes vertically down into the EPS floor.
    # Cutting a few dB above the floor keeps the EDC readable without showing
    # that deep drop.
    if noise_compensate and np.isfinite(inr_db):
        display_floor_db = -inr_db + 5.0
        below = np.where(edc_db <= display_floor_db)[0]
        edc_plot_end = int(below[0]) if len(below) else len(edc_db)
    else:
        edc_plot_end = len(edc_db)
    edc_plot_end = int(np.clip(edc_plot_end, 1, len(edc_db)))

    return {
        "edt": edt,
        "rt20": rt20,
        "rt30": rt30,
        "c50": clarity_c50(hs, fs, 0, knee=knee),
        "c80": clarity_c80(hs, fs, 0, knee=knee),
        "d50": definition_d50(hs, fs, 0, knee=knee),
        "ts_ms": center_time_ts(hs, fs, 0, knee=knee),

        "direct_index": direct_index,
        "direct_time_s": direct_index / fs,
        "lundeby_knee_sample": int(direct_index + knee),
        "lundeby_knee_s": (direct_index + knee) / fs,
        "noise_power": noise_power,
        "noise_db": noise_db,

        "edc_db": edc_db,
        "edc_plot_end": edc_plot_end,
        "inr_db": inr_db,
    }