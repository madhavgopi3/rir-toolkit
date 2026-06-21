from dataclasses import dataclass
from pathlib import Path


@dataclass
class MeasurementConfig:

    # Audio / sweep
    fs: int = 48000
    sweep_duration: float = 10.0
    f_start: int = 20
    f_end: int = 20000
    spl_db_offset: float = 2.0

    # External sweep settings
    f_start2: int = 50
    f_end2: int = 22000
    amplitude: float = 0.8

    sweep_pre_silence: float = 0.5
    sweep_post_silence: float = 2.0

    use_external_sweep: bool = False #Make sure to input the sweep settings above when using external sweep

    # RIR trimming
    rir_trim_pre_ms: float = 5.0
    rir_min_tail_ms: float = 300.0
    threshold_over_noise_db: float = 25.0
    arrival_smooth_ms: float = 1.0
    tail_smooth_ms: float = 5.0
    safety_offset_ms: float = 30.0


    # Acoustic descriptors
    descriptor_noise_compensate: bool = True
    direct_sound_search_ms: float = 20.0

    # Setting the level of the point before the IR peak where the level
    # first rises above (peak - 20 dB). This sets that 20 dB trigger.
    direct_onset_drop_db: float = 20.0

    lundeby_block_ms: float = 10.0
    lundeby_tail_fraction: float = 0.1
    lundeby_margin_db: float = 10.0
    lundeby_max_iter: int = 10

    # Minimum impulse-to-noise ratio (INR) for a valid
    # decay-time estimate. If the range of usable decay is less than
    # the values specified below, they are returned as NaN rather than reported.
    inr_min_edt_db: float = 15.0  
    inr_min_t20_db: float = 35.0   
    inr_min_t30_db: float = 45.0   

    # Safety bounds for the fitted decay times (s).
    rt_min_s: float = 0.03
    rt_max_s: float = 10.0
    rt_min_r2: float = 0.90 # reject decays that are not sufficiently linear

    # Band analysis
    n_fft: int = 262144

    # Fractional octave width used for the frequency-response heatmaps only.
    # 1 = octave, 3 = third-octave, 6 = sixth-octave.
    band_fraction: int = 3

    # Octave width for banded EDT / T20 / T30 / C50 / C80 
    rt_band_fraction: int = 1

    # Used for frequency-response heatmaps
    band_centres: tuple[int, ...] = (
        25, 31, 40, 50, 63, 80,
        100, 125, 160, 200, 250, 315,
        400, 500, 630, 800, 1000, 1250,
        1600, 2000, 2500, 3150, 4000, 5000,
        6300, 8000, 10000, 12500, 16000,
    )

    # Used for banded EDT / RT20 / RT30 / C50 / C80.
    # ISO 3382-1 requires 125 Hz .. 4 kHz octave centres as a minimum
    rt_band_centres: tuple[int, ...] = (
        63, 125, 250, 500, 1000, 2000, 4000,
    )

    # Filter order for the banded RT / EDT calculation. A 6th-order Butterworth
    # applied zero-phase (sosfiltfilt) approximates an IEC 61260 class-1
    # octave-band filter.
    band_filter_order: int = 6

    # Paths
    output_dir: Path = Path("output")
    recorded_dir: Path = Path("sweep1")

    recorded_sweep_path: Path = Path("recorded/1.wav")
    recorded_sweep_path2: Path = Path("recorded/2.wav")
    external_sweep_path: Path = Path("sweep_48000_50_22000.wav")

    generated_sweep_name: str = "generated_sweep.wav"
    padded_sweep_name: str = "padded_generated_sweep.wav"
    inverse_sweep_name: str = "inverse_sweep.wav"
    external_inverse_name: str = "external_inverted_sweep.wav"
    rir_name: str = "rir.wav"
    trimmed_rir_name: str = "rir_trimmed.wav"
