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
    rir_trim_pre_ms: float = 0.0
    rir_min_tail_ms: float = 300.0
    threshold_over_noise_db: float = 25.0
    arrival_smooth_ms: float = 1.0
    tail_smooth_ms: float = 5.0
    safety_offset_ms: float = 30.0


    # Acoustic descriptors
    descriptor_noise_compensate: bool = True
    direct_sound_search_ms: float = 20.0

    lundeby_block_ms: float = 10.0
    lundeby_tail_fraction: float = 0.1
    lundeby_margin_db: float = 10.0
    lundeby_max_iter: int = 10

    # Band analysis
    n_fft: int = 262144
    band_fraction: int = 3   # 1 = octave, 3 = third-octave, 6 = sixth-octave
    # band_centres: tuple[int, ...] = (125, 250, 500, 1000, 2000, 4000)
    band_centres: tuple[int, ...] = (
    25, 31, 40, 50, 63, 80,
    100, 125, 160, 200, 250, 315,
    400, 500, 630, 800, 1000, 1250,
    1600, 2000, 2500, 3150, 4000, 5000,
    6300, 8000, 10000, 12500, 16000
    )
    band_filter_order: int = 4  # Filter order for banded RT / EDT calculation

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