import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.interpolate import griddata


def plot_waveform(signal: np.ndarray, fs: int, title: str):
    t = np.arange(len(signal)) / fs
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, signal, linewidth=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

"""
def plot_together(signal: np.ndarray, signal2: np.ndarray, fs: int, title: str):
    t = np.arange(len(signal)) / fs
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, signal, linestyle='--', linewidth = 0.8)
    ax.plot(t, signal2 + 0.1, linestyle='-', linewidth = 0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha = 0.3)
    
    fig.tight_layout()
    return fig


"""

#Now same as plot_waveform. Make changes if needed in the future.
def plot_rir(signal: np.ndarray, fs: int, title: str):
    return plot_waveform(signal, fs, title)

def plot_spectrogram(signal: np.ndarray, fs: int, title: str):
    f, t, mag = spectrogram(signal, fs=fs, nperseg=2048, noverlap=1024, mode="magnitude") # Each FFT window uses 2048 samples. Adjacent windows overlap by 1024 samples.

    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(t, f, (20*np.log10(mag + 1e-12)), shading="gouraud") #colour will show the magnitude in dB.
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    ax.grid(True, alpha = 0.3)

    fig.colorbar(mesh, ax=ax, label="Magnitude [dB]")
    fig.tight_layout()
    return fig

def plot_edc(edc: np.ndarray, fs: int, title: str = "Energy Decay Curve"):
    t = np.arange(len(edc))/fs
    edc_db = 10 * np.log10(edc + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, edc_db, linewidth = 0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Level [dB]")
    ax.set_title(title)
    ax.grid(True, alpha = 0.3)

    fig.tight_layout()
    return fig



# MATLAB Adaptation Part

"""
    Plot the full centered deconvolved response.
    First half usually contains nonlinear components,
    second half contains the linear impulse response.
"""

def plot_deconvolution_result(
    ir_full: np.ndarray,
    fs: int,
    title: str = "Full Deconvolved Response",
):
    ir_full = np.asarray(ir_full, dtype=np.float64).squeeze()
    t = np.arange(len(ir_full)) / fs
    floor_db = -120.0

    epsilon = 10 ** (floor_db / 20)
    ir_full_db = 20 * np.log10(np.maximum(np.abs(ir_full), epsilon))

    mid_time = (len(ir_full) // 2) / fs

    fig_wave, ax_wave = plt.subplots(figsize=(10, 4))
    ax_wave.plot(t, ir_full)
    ax_wave.axvline(mid_time, linestyle="--")
    ax_wave.set_xlabel("Time [s]")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title(title)
    ax_wave.grid(True)
    fig_wave.tight_layout()

    fig_db, ax_db = plt.subplots(figsize=(10, 4))
    ax_db.plot(t, ir_full_db)
    ax_db.axvline(mid_time, linestyle="--")
    ax_db.set_xlabel("Time [s]")
    ax_db.set_ylabel("Level [dB]")
    ax_db.set_title(title + " [dB]")
    ax_db.grid(True)
    fig_db.tight_layout()

    return {
        "deconvolution_full": fig_wave,
        "deconvolution_full_db": fig_db,
    }



def plot_linear_and_nonlinear_ir(
    ir_lin: np.ndarray,
    ir_nonlin: np.ndarray,
    fs: int,
):
    """
    Plot linear and nonlinear components separately.
    """
    ir_lin = np.asarray(ir_lin, dtype=np.float64).squeeze()
    ir_nonlin = np.asarray(ir_nonlin, dtype=np.float64).squeeze()

    t_lin = np.arange(len(ir_lin)) / fs
    t_nonlin = np.arange(len(ir_nonlin)) / fs

    fig_nonlin, ax_nonlin = plt.subplots(figsize=(10, 4))
    ax_nonlin.plot(t_nonlin, ir_nonlin)
    ax_nonlin.set_xlabel("Time [s]")
    ax_nonlin.set_ylabel("Amplitude")
    ax_nonlin.set_title("Nonlinear Components")
    ax_nonlin.grid(True)
    fig_nonlin.tight_layout()

    fig_lin, ax_lin = plt.subplots(figsize=(10, 4))
    ax_lin.plot(t_lin, ir_lin)
    ax_lin.set_xlabel("Time [s]")
    ax_lin.set_ylabel("Amplitude")
    ax_lin.set_title("Linear Impulse Response")
    ax_lin.grid(True)
    fig_lin.tight_layout()

    return {
        "nonlinear_ir": fig_nonlin,
        "linear_ir": fig_lin,
    }


def plot_linear_and_nonlinear_db(
    ir_lin: np.ndarray,
    ir_nonlin: np.ndarray,
    fs: int,
    floor_db: float = -120.0,
):
    """
    Plot absolute magnitude in dB for linear and nonlinear components.
    Useful because raw waveform plots can hide low-level structure.
    """
    ir_lin = np.asarray(ir_lin, dtype=np.float64).squeeze()
    ir_nonlin = np.asarray(ir_nonlin, dtype=np.float64).squeeze()

    epsilon = 10 ** (floor_db / 20.0) #Setting a floor to avoid log 0.

    lin_db = 20 * np.log10(np.maximum(np.abs(ir_lin), epsilon))
    nonlin_db = 20 * np.log10(np.maximum(np.abs(ir_nonlin), epsilon))

    t_lin = np.arange(len(ir_lin)) / fs
    t_nonlin = np.arange(len(ir_nonlin)) / fs

    fig_nonlin, ax_nonlin = plt.subplots(figsize=(10, 4))
    ax_nonlin.plot(t_nonlin, nonlin_db)
    ax_nonlin.set_xlabel("Time [s]")
    ax_nonlin.set_ylabel("Level [dB]")
    ax_nonlin.set_title("Nonlinear Components (dB)")
    ax_nonlin.grid(True)
    fig_nonlin.tight_layout()

    fig_lin, ax_lin = plt.subplots(figsize=(10, 4))
    ax_lin.plot(t_lin, lin_db)
    ax_lin.set_xlabel("Time [s]")
    ax_lin.set_ylabel("Level [dB]")
    ax_lin.set_title("Linear Impulse Response (dB)")
    ax_lin.grid(True)
    fig_lin.tight_layout()

    return {
        "non_linear_db": fig_nonlin,
        "linear_ir": fig_lin
    }

def compute_fft_rir(h: np.ndarray, fs:int, n_fft: int): # 65536 because 2^16. freq_resolution = fs/nfft. n_fft is best if it's the next power of 2 greater than len(rir)

    h = np.asarray(h, dtype=np.float64).squeeze()

    H = np.fft.rfft(h, n = n_fft) # H is a complex. Use angle(H) for phase and abs (H) for magnitude.
    freqs = np.fft.rfftfreq(n_fft, d = 1/fs) #freqs is frequency bins. Eg: freqs = [20, 500, 980, 1005, 1500]
    magnitude_db = 20 * np.log10(np.abs(H) + 1e-12)

    return freqs, magnitude_db

def plot_fft_rir(freqs:np.ndarray, magnitude_db:np.ndarray, title: str):
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(freqs, magnitude_db)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB SPL]")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.set_xlim(20, freqs[-1])

    fig.tight_layout()
    return fig

def get_rir_at_freq(freqs: np.ndarray, magnitude_db:np.ndarray, freq: int):

    db_at_freq = np.argmin(np.abs(freqs - freq))

    return freqs[db_at_freq], magnitude_db[db_at_freq]

def show_all():
    plt.show()

def make_grid(results, value_key):
    """
    Return the 8 * 5 grid with the 40 values of a particular acoustic descriptor. Eg: 40 values of RT30.

    Layout:
        columns left to right: E, D, C, B, A
        rows top to bottom:   1, 2, 3, ..., 8

    results: the dictionary for each point with all the descriptors (it is written to the csv)
    value_key: the descriptor I'm plotting
    """

    grid = np.full((8, 5), np.nan)

    x_labels = ["E", "D", "C", "B", "A"]

    for item in results: # Loops through each point and its descriptors. Eg: item = {"point": "C4", "RT30": 0.81, ......}
        point = item["point"].upper()

        letter = point[0]
        number = int(point[1:])

        row = number - 1
        col = x_labels.index(letter) # Converts the point alphabet to number using the x_labels indices.

        grid[row, col] = item[value_key]

    return grid


def interpolate_grid(values_2d):
    """
    Interpolates a 2D measurement grid only for smoother colour-map display.
    """
    values_2d = np.asarray(values_2d, dtype=np.float64)

    n_rows, n_cols = values_2d.shape

    x = np.arange(n_cols)
    y = np.arange(n_rows)
    X, Y = np.meshgrid(x, y)

    interpolation_factor = 10

    xi = np.linspace(0, n_cols - 1, n_cols * interpolation_factor)
    yi = np.linspace(0, n_rows - 1, n_rows * interpolation_factor)
    XI, YI = np.meshgrid(xi, yi)

    zi_smooth = griddata(
        points=(X.ravel(), Y.ravel()),
        values=values_2d.ravel(),
        xi=(XI, YI),
        method="cubic"
    )

    if np.isnan(zi_smooth).any():
        zi_nearest = griddata(
            points=(X.ravel(), Y.ravel()),
            values=values_2d.ravel(),
            xi=(XI, YI),
            method="nearest"
        )
        zi_smooth = np.where(np.isnan(zi_smooth), zi_nearest, zi_smooth)

    return zi_smooth

def plot_heatmap(values_2d, title, cbar_label, output_path=None):
    values_2d = np.asarray(values_2d, dtype=np.float64)

    zi_smooth = interpolate_grid(values_2d)

    fig, ax = plt.subplots(figsize=(9, 5))

    im = ax.imshow(
        zi_smooth,
        origin="upper",
        extent=[
            0.5,
            values_2d.shape[1] + 0.5,
            values_2d.shape[0] + 0.5,
            0.5,
        ],
        aspect="auto",
        cmap="viridis"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel("Measurement position")
    ax.set_ylabel("Measurement row")

    # Your grid is 8 rows x 5 columns:
    # rows = 1 to 8
    # columns = E, D, C, B, A
    ax.set_xticks(np.arange(1, values_2d.shape[1] + 1))
    ax.set_xticklabels(["E", "D", "C", "B", "A"][:values_2d.shape[1]])

    ax.set_yticks(np.arange(1, values_2d.shape[0] + 1))
    ax.set_yticklabels([str(i) for i in range(1, values_2d.shape[0] + 1)])

    x_points = np.arange(1, values_2d.shape[1] + 1)
    y_points = np.arange(1, values_2d.shape[0] + 1)
    Xp, Yp = np.meshgrid(x_points, y_points)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig

