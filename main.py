import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from config import MeasurementConfig

from sweep_gen import (
    generate_log_sweep, generate_inverse_filter, normalise_peak, pad_signal
)
from audio_io import (
    load_audio, save_audio, normalise_for_saving, check_clipping
)
from alignment import extract_aligned_segment
from deconvolution import extract_rir
from rir_processing import normalise_rir, trim_rir_robust
from visualisation import (
    plot_rir, plot_spectrogram, plot_waveform, plot_edc,
    compute_fft_rir,
    plot_fft_rir,
    get_rir_at_freq,
    make_grid,
    plot_heatmap,
    show_all
)
from external_sweep import rir_from_external_sweep
from harmonic_separation import extract_ir_sweep
from acoustic_descriptors import extract_room_descriptors
from band_analysis import freqband_average_db, bandpass_rir

def parse_points(path):
    """
    Parse points into rows and columns.
    """
    path = Path(path)
    name = path.stem.upper() # Removes path extension and converts to uppercase.
    row_label = name[0] 
    column_label = int(name[1:])
    return row_label, column_label

def save_figure(fig, output_path, dpi=150, close=True):
    """
    Saves one figure. This function is looped in save_figures
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if close:
        plt.close(fig)


def save_figures(figures, output_dir, point_name, dpi=150):
    """
    Saves all figures from a dictionary.
    """

    output_dir = Path(output_dir)

    for name, item in figures.items():
        save_path = output_dir / name / f"{point_name}_{name}.png"
        save_figure(item, save_path, dpi=dpi, close=True)

def main(cfg=None):
    # Accept a config from a caller (e.g. the GUI); fall back to defaults when
    # run from the command line.
    if cfg is None:
        cfg = MeasurementConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    recorded_dir = cfg.recorded_dir
    output_dir = cfg.output_dir

    output_dirs = {
        "rir_wav": output_dir / "rir_wav",
        "trimmed_rir_wav": output_dir / "trimmed_rir_wav",
        "plots": output_dir / "plots",
        "csv": output_dir / "csv",
        "heatmaps": output_dir / "heatmaps",
    }

    for folder in output_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    recorded_files = sorted(recorded_dir.glob("*.wav"), key=parse_points)

    if len(recorded_files) == 0:
        raise FileNotFoundError(f"No .wav files found in {recorded_dir}")

    results = []

    # GENERATED SWEEP MODE

    if not cfg.use_external_sweep:
        raw_sweep = generate_log_sweep(
            fs=cfg.fs,
            duration=cfg.sweep_duration,
            f_start=cfg.f_start,
            f_end=cfg.f_end,
            amplitude=cfg.amplitude,
        )

        raw_sweep = normalise_peak(raw_sweep, peak=0.999)

        padded_sweep = pad_signal(
            raw_sweep,
            fs=cfg.fs,
            pre_silence=cfg.sweep_pre_silence,
            post_silence=cfg.sweep_post_silence,
        )

        inverse_filter = generate_inverse_filter(
            sweep=raw_sweep,
            fs=cfg.fs,
            f_start=cfg.f_start,
            f_end=cfg.f_end,
        )

        inverse_filter = normalise_peak(inverse_filter, peak=0.999)

        save_audio(output_dir / cfg.generated_sweep_name, raw_sweep, cfg.fs)
        save_audio(output_dir / cfg.padded_sweep_name, padded_sweep, cfg.fs)
        save_audio(output_dir / cfg.inverse_sweep_name, inverse_filter, cfg.fs)

        sweep_for_plot = raw_sweep

        reference_figures = {
            "reference_sweep": plot_waveform(
                sweep_for_plot,
                cfg.fs,
                "Reference Sweep"
            )
        }

        save_figures(
            figures=reference_figures,
            output_dir=output_dirs["plots"],
            point_name="reference",
            dpi=150,
        )

    # EXTERNAL SWEEP MODE  
    
    else:
        sweep_for_plot, _ = load_audio(
            cfg.external_sweep_path,
            target_fs=cfg.fs,
            mono=True
        )

        reference_figures = {
            "reference_sweep": plot_waveform(
                sweep_for_plot,
                cfg.fs,
                "External Reference Sweep"
            )
        }

        save_figures(
            figures=reference_figures,
            output_dir=output_dirs["plots"],
            point_name="reference",
            dpi=150,
        )

        inverse_filter = None

    # ------------------------------------------------------------
    # PROCESS ALL 40 RECORDED FILES
    # ------------------------------------------------------------
    for rec_path in recorded_files:
        point_name = rec_path.stem.upper()
        row_label, column_label = parse_points(point_name)

        print(f"\nProcessing {point_name}")

        if not cfg.use_external_sweep:
            recorded, _ = load_audio(rec_path, target_fs=cfg.fs, mono=True)
            clipped = check_clipping(recorded)

            aligned_recording, lag = extract_aligned_segment(raw_sweep, recorded)

            rir_raw = extract_rir(aligned_recording, inverse_filter)

            ir_lin, ir_nonlin, ir_full = extract_ir_sweep(
                sweep_response=aligned_recording,
                inverse_sweep=inverse_filter,
            )

        else:
            result = rir_from_external_sweep(
                sweep_path=cfg.external_sweep_path,
                recorded_path=rec_path,
                f_start=cfg.f_start2,
                f_end=cfg.f_end2,
                target_fs=cfg.fs,
                mono=True,
            )

            sweep_for_plot = result["sweep"]
            recorded = result["recorded"]
            inverse_filter = result["inverse_filter"]
            lag = result["lag_samples"]
            rir_raw = result["rir_raw"]

            clipped = check_clipping(recorded)

            aligned_recording = result["aligned_recording"]

            ir_lin, ir_nonlin, ir_full = extract_ir_sweep(
            sweep_response=aligned_recording,
            inverse_sweep=inverse_filter,
)
        # COMMON POST-PROCESSING
        
        rir_trimmed, trim_start, trim_end, peak_idx, envelope = trim_rir_robust(
        rir_raw,
        fs=cfg.fs,
        pre_ms=cfg.rir_trim_pre_ms,
        min_tail_ms=cfg.rir_min_tail_ms,
        threshold_over_noise_db=cfg.threshold_over_noise_db,
        arrival_smooth_ms=cfg.arrival_smooth_ms,
        tail_smooth_ms=cfg.tail_smooth_ms,
        safety_offset_ms=cfg.safety_offset_ms,
        )

        descriptors = extract_room_descriptors(
        rir=rir_trimmed,
        fs=cfg.fs,
        noise_compensate=cfg.descriptor_noise_compensate,
        direct_sound_search_ms=cfg.direct_sound_search_ms,
        lundeby_block_ms=cfg.lundeby_block_ms,
        lundeby_tail_fraction=cfg.lundeby_tail_fraction,
        lundeby_margin_db=cfg.lundeby_margin_db,
        lundeby_max_iter=cfg.lundeby_max_iter,
        onset_drop_db=cfg.direct_onset_drop_db,
        inr_min_edt_db=cfg.inr_min_edt_db,
        inr_min_t20_db=cfg.inr_min_t20_db,
        inr_min_t30_db=cfg.inr_min_t30_db,
        rt_min_s=cfg.rt_min_s,
        rt_max_s=cfg.rt_max_s,
        rt_min_r2=cfg.rt_min_r2,
)

        freqs, magnitude_db = compute_fft_rir(
        h=rir_trimmed,
        fs=cfg.fs,
        n_fft=cfg.n_fft,
        )

        magnitude_spl = magnitude_db + cfg.spl_db_offset

        band_response_values = {}
        band_descriptor_values = {}

        # Frequency-response band values
        for centre in cfg.band_centres:
            band_response_values[f"{centre}Hz_fr_db"] = freqband_average_db(
                freqs,
                magnitude_spl,
                centre,
                cfg.band_fraction,
            )


        # Banded acoustic descriptors
        for centre in cfg.rt_band_centres:
            filtered_rir = bandpass_rir(
                rir_trimmed,
                cfg.fs,
                centre,
                cfg.rt_band_fraction,
                order=cfg.band_filter_order,
            )

            band_desc = extract_room_descriptors(
                rir=filtered_rir,
                fs=cfg.fs,
                noise_compensate=cfg.descriptor_noise_compensate,
                direct_sound_search_ms=cfg.direct_sound_search_ms,
                lundeby_block_ms=cfg.lundeby_block_ms,
                lundeby_tail_fraction=cfg.lundeby_tail_fraction,
                lundeby_margin_db=cfg.lundeby_margin_db,
                lundeby_max_iter=cfg.lundeby_max_iter,
                direct_index=descriptors["direct_index"],
                onset_drop_db=cfg.direct_onset_drop_db,
                inr_min_edt_db=cfg.inr_min_edt_db,
                inr_min_t20_db=cfg.inr_min_t20_db,
                inr_min_t30_db=cfg.inr_min_t30_db,
                rt_min_s=cfg.rt_min_s,
                rt_max_s=cfg.rt_max_s,
                rt_min_r2=cfg.rt_min_r2,
            )

            band_descriptor_values[f"{centre}Hz_edt"] = band_desc["edt"]
            band_descriptor_values[f"{centre}Hz_rt20"] = band_desc["rt20"]
            band_descriptor_values[f"{centre}Hz_rt30"] = band_desc["rt30"]
            band_descriptor_values[f"{centre}Hz_c50"] = band_desc["c50"]
            band_descriptor_values[f"{centre}Hz_c80"] = band_desc["c80"]
            band_descriptor_values[f"{centre}Hz_d50"] = band_desc["d50"]
            band_descriptor_values[f"{centre}Hz_ts_ms"] = band_desc["ts_ms"]
            band_descriptor_values[f"{centre}Hz_inr_db"] = band_desc["inr_db"]
        

        rir_trimmed_norm = normalise_rir(rir_trimmed)

        if point_name in ["C3", "C4"]:
                freq_1k, magn_1k = get_rir_at_freq(freqs, magnitude_db, 1000)
                print(f"Freq at {freq_1k:2f} is {magn_1k:2f}")

        # SAVE AUDIO

        save_audio(
            output_dirs["rir_wav"] / f"{point_name}_rir_raw.wav",
            normalise_for_saving(rir_raw),
            cfg.fs,
        )

        save_audio(
            output_dirs["trimmed_rir_wav"] / f"{point_name}_rir_trimmed.wav",
            normalise_for_saving(rir_trimmed_norm),
            cfg.fs,
        )

        # CREATE PLOTS

        # Plot the noise-compensated EDC only down to the noise floor; past
        # that point the compensated curve dives into the floor.
        edc_db_plot = descriptors["edc_db"][:descriptors["edc_plot_end"]]

        figures = {
            "recorded_signal": plot_waveform(
                recorded,
                cfg.fs,
                f"Recorded Signal - {point_name}"
            ),
            "recorded_spectrogram": plot_spectrogram(
                recorded,
                cfg.fs,
                f"Recorded Signal Spectrogram - {point_name}"
            ),
            "rir_raw": plot_rir(
                rir_raw,
                cfg.fs,
                f"Raw Extracted RIR - {point_name}"
            ),
            "rir_trimmed": plot_rir(
                rir_trimmed_norm,
                cfg.fs,
                f"Trimmed + Normalised RIR - {point_name}"
            ),
            "frequency_response": plot_fft_rir(
                freqs,
                magnitude_spl,
                f"Frequency Response from RIR - {point_name} [dB SPL]"
            ),
            "edc": plot_edc(
                edc_db_plot,
                cfg.fs,
                f"Energy Decay Curve (ISO, noise-compensated) - {point_name}",
                in_db=True,
            ),
        }

        save_figures(
            figures=figures,
            output_dir=output_dirs["plots"],
            point_name=point_name,
            dpi=150,
        )

        # SAVE RESULTS FOR CSV
        
        results.append({
            "point": point_name,
            "row_label": row_label,
            "column_label": column_label,
            "recording_file": rec_path.name,
            "lag_samples": lag,
            "lag_seconds": lag / cfg.fs,
            "clipped": clipped,
            "direct_peak_sample": peak_idx,
            "direct_peak_seconds": peak_idx / cfg.fs,
            "trim_start_sample": trim_start,
            "trim_end_sample": trim_end,
            "trimmed_length_samples": len(rir_trimmed_norm),
            "trimmed_length_seconds": len(rir_trimmed_norm) / cfg.fs,

            "broadband_edt": descriptors["edt"],
            "broadband_rt20": descriptors["rt20"],
            "broadband_rt30": descriptors["rt30"],
            "broadband_c50": descriptors["c50"],
            "broadband_c80": descriptors["c80"],
            "broadband_d50": descriptors["d50"],
            "broadband_ts_ms": descriptors["ts_ms"],
            "direct_index": descriptors["direct_index"],
            "lundeby_knee_s": descriptors["lundeby_knee_s"],
            "noise_db": descriptors["noise_db"],
            "broadband_inr_db": descriptors["inr_db"],
            **band_response_values,
            **band_descriptor_values,
        })

        print(f"Finished {point_name}")
        print(f"Lag: {lag} samples ({lag / cfg.fs:.4f} s)")
        print(f"Clipped: {clipped}")

    # CREATE OCTAVE BAND & DESCRIPTOR HEATMAPS

    #Banded Frequency Response Heatmaps

    frequency_heatmap_values = []

    for centre in cfg.band_centres:
        key = f"{centre}Hz_fr_db"

        for item in results:
            frequency_heatmap_values.append(item[key])

    frequency_heatmap_values = np.asarray(frequency_heatmap_values, dtype=np.float64)

    freq_vmin = np.nanpercentile(frequency_heatmap_values, 5)
    freq_vmax = np.nanpercentile(frequency_heatmap_values, 95)

    print(f"Frequency heatmap colour scale: {freq_vmin:.2f} to {freq_vmax:.2f} dB")

    for centre in cfg.band_centres:
        key = f"{centre}Hz_fr_db"

        grid = make_grid(results, key)

        fig = plot_heatmap(
            grid,
            title=f"{centre} Hz Frequency Response Across Measurement Grid",
            cbar_label="Magnitude [dB SPL]",
            vmin=freq_vmin,
            vmax=freq_vmax,
        )

        save_figure(
            fig,
            output_dirs["heatmaps"] / f"{key}_heatmap.png",
            dpi=150,
            close=True,
        )
        
    # Banded RT30 heatmaps

    rt_heatmap_values = []

    for centre in cfg.rt_band_centres:
        key = f"{centre}Hz_rt30"

        for item in results:
            rt_heatmap_values.append(item[key])

    rt_heatmap_values = np.asarray(rt_heatmap_values, dtype=np.float64)

    rt_vmin = np.nanpercentile(rt_heatmap_values, 5)
    rt_vmax = np.nanpercentile(rt_heatmap_values, 95)

    print(f"RT30 heatmap colour scale: {rt_vmin:.2f} to {rt_vmax:.2f} s")

    # Then plot each RT30 heatmap using the same colour scale
    for centre in cfg.rt_band_centres:
        key = f"{centre}Hz_rt30"

        grid = make_grid(results, key)

        fig = plot_heatmap(
            grid,
            title=f"{centre} Hz RT30 Across Measurement Grid",
            cbar_label="RT30 [s]",
            vmin=rt_vmin,
            vmax=rt_vmax,
        )

        save_figure(
            fig,
            output_dirs["heatmaps"] / f"{key}_heatmap.png",
            dpi=150,
            close=True,
        )

    # Broadband summary heatmaps
    broadband_heatmaps = {
        "broadband_rt30": "Broadband RT30 [s]",
        "broadband_edt": "Broadband EDT [s]",
        "broadband_c50": "Broadband C50 [dB]",
    }

    for key, label in broadband_heatmaps.items():
        grid = make_grid(results, key)

        fig = plot_heatmap(
            grid,
            title=f"{label} Across Measurement Grid",
            cbar_label=label,
        )

        save_figure(
            fig,
            output_dirs["heatmaps"] / f"{key}_heatmap.png",
            dpi=150,
            close=True,
        )
        
    # SAVE CSV
    
    csv_path = output_dirs["csv"] / f"batch_results_fraction_{cfg.band_fraction}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nDone.")
    print(f"Processed {len(recorded_files)} files.")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()