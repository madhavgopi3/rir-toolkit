import sys
import queue
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")  # non-interactive backend - plots are saved, not shown

from config import MeasurementConfig
from main import main


class LogRedirect:
    """Redirect stdout/stderr to a thread-safe queue.

    The pipeline runs on a worker thread, so it must never touch Tk widgets
    directly. Text is pushed onto a queue and drained on the main thread.
    """

    def __init__(self, log_queue: "queue.Queue"):
        self._queue = log_queue

    def write(self, text: str):
        if text:
            self._queue.put(("log", text))

    def flush(self):
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Acoustic Measurement Processor")
        self.resizable(True, True)
        self.minsize(640, 520)

        # All worker -> UI communication goes through this queue.
        self._msg_queue: "queue.Queue" = queue.Queue()

        self._apply_theme()
        self._build_ui()

        # Start the main-thread poller that drains worker messages.
        self.after(50, self._poll_queue)

    def _apply_theme(self):
        # "aqua" only exists on macOS; fall back gracefully elsewhere.
        style = ttk.Style()
        available = style.theme_names()
        for theme in ("aqua", "vista", "clam", "default"):
            if theme in available:
                style.theme_use(theme)
                break

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}

        # -- Paths --
        paths_frame = ttk.LabelFrame(self, text="Paths")
        paths_frame.pack(fill="x", **pad)

        self._recordings_var = tk.StringVar(value="sweep1")
        self._output_var = tk.StringVar(value="output")
        self._ext_sweep_var = tk.StringVar(value="")

        self._add_path_row(paths_frame, "Recordings dir", self._recordings_var, mode="dir", row=0)
        self._add_path_row(paths_frame, "Output dir",     self._output_var,     mode="dir", row=1)

        # -- Sweep mode --
        mode_frame = ttk.LabelFrame(self, text="Sweep Mode")
        mode_frame.pack(fill="x", **pad)

        self._mode_var = tk.StringVar(value="generated")
        ttk.Radiobutton(mode_frame, text="Generated sweep",
                        variable=self._mode_var, value="generated",
                        command=self._on_mode_change).grid(row=0, column=0, sticky="w", padx=12, pady=4)
        ttk.Radiobutton(mode_frame, text="External sweep",
                        variable=self._mode_var, value="external",
                        command=self._on_mode_change).grid(row=0, column=1, sticky="w", padx=12, pady=4)

        # -- Generated sweep settings --
        self._gen_frame = ttk.LabelFrame(self, text="Generated Sweep Settings")
        self._gen_frame.pack(fill="x", **pad)

        self._fs_var       = tk.StringVar(value="48000")
        self._dur_var      = tk.StringVar(value="10.0")
        self._fstart_var   = tk.StringVar(value="20")
        self._fend_var     = tk.StringVar(value="20000")

        self._add_field(self._gen_frame, "Sample rate (Hz)",  self._fs_var,     row=0, col=0)
        self._add_field(self._gen_frame, "Duration (s)",      self._dur_var,    row=0, col=2)
        self._add_field(self._gen_frame, "F start (Hz)",      self._fstart_var, row=1, col=0)
        self._add_field(self._gen_frame, "F end (Hz)",        self._fend_var,   row=1, col=2)

        # -- External sweep settings --
        self._ext_frame = ttk.LabelFrame(self, text="External Sweep Settings")

        self._fstart2_var  = tk.StringVar(value="50")
        self._fend2_var    = tk.StringVar(value="22000")

        self._add_path_row(self._ext_frame, "Sweep file", self._ext_sweep_var, mode="file", row=0)
        self._add_field(self._ext_frame, "F start (Hz)", self._fstart2_var, row=1, col=0)
        self._add_field(self._ext_frame, "F end (Hz)",   self._fend2_var,   row=1, col=2)

        self._on_mode_change()

        # -- Run button + status --
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill="x", padx=10, pady=(6, 2))

        self._run_btn = tk.Button(ctrl_frame, text="Run", width=14, command=self._run)
        self._run_btn.pack(side="left")

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(ctrl_frame, textvariable=self._status_var,
                  foreground="gray").pack(side="left", padx=12)

        # -- Log --
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)

        self._log = scrolledtext.ScrolledText(
            log_frame, state="disabled", wrap="word",
            font="TkFixedFont", background="#1e1e1e", foreground="#d4d4d4",
            insertbackground="white",
        )
        self._log.pack(fill="both", expand=True, padx=4, pady=4)

    def _add_path_row(self, parent, label, var, mode, row):
        ttk.Label(parent, text=label + ":").grid(row=row, column=0, sticky="w", padx=8, pady=3)
        ttk.Entry(parent, textvariable=var, width=42).grid(row=row, column=1, padx=4, pady=3)
        cmd = (lambda v=var: self._browse_dir(v)) if mode == "dir" else (lambda v=var: self._browse_file(v))
        tk.Button(parent, text="Browse", command=cmd).grid(row=row, column=2, padx=4)

    def _add_field(self, parent, label, var, row, col):
        ttk.Label(parent, text=label + ":").grid(row=row, column=col,   sticky="w", padx=8, pady=3)
        ttk.Entry(parent, textvariable=var, width=10).grid(row=row, column=col+1, padx=4, pady=3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _browse_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory(initialdir=var.get() or ".")
        if path:
            var.set(path)

    def _browse_file(self, var: tk.StringVar):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if path:
            var.set(path)

    def _on_mode_change(self):
        if self._mode_var.get() == "generated":
            self._gen_frame.pack(fill="x", padx=10, pady=4,
                                 before=self._ext_frame if self._ext_frame.winfo_ismapped() else None)
            self._ext_frame.pack_forget()
        else:
            self._gen_frame.pack_forget()
            self._ext_frame.pack(fill="x", padx=10, pady=4)

    def _log_write(self, text: str):
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")

    # ------------------------------------------------------------------
    # Main-thread queue poller — the ONLY place that updates widgets while
    # a run is in progress.
    # ------------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._msg_queue.get_nowait()
                if kind == "log":
                    self._log_write(payload)
                elif kind == "status":
                    self._status_var.set(payload)
                elif kind == "done":
                    self._run_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self):
        self._run_btn.configure(state="disabled")
        self._log_write_clear()
        self._status_var.set("Running…")

        try:
            cfg = self._build_config()
        except ValueError as e:
            self._log_write(f"Config error: {e}\n")
            self._run_btn.configure(state="normal")
            self._status_var.set("Error")
            return

        thread = threading.Thread(target=self._worker, args=(cfg,), daemon=True)
        thread.start()

    def _log_write_clear(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _build_config(self) -> MeasurementConfig:
        use_external = self._mode_var.get() == "external"

        try:
            fs    = int(self._fs_var.get())
            dur   = float(self._dur_var.get())
            fst   = int(self._fstart_var.get())
            fend  = int(self._fend_var.get())
            fst2  = int(self._fstart2_var.get())
            fend2 = int(self._fend2_var.get())
        except ValueError as e:
            raise ValueError(f"Invalid number: {e}") from e

        recorded_dir = Path(self._recordings_var.get().strip())
        output_dir   = Path(self._output_var.get().strip())

        if not recorded_dir.exists():
            raise ValueError(f"Recordings dir not found: {recorded_dir}")

        cfg = MeasurementConfig(
            fs=fs,
            sweep_duration=dur,
            f_start=fst,
            f_end=fend,
            f_start2=fst2,
            f_end2=fend2,
            use_external_sweep=use_external,
            recorded_dir=recorded_dir,
            output_dir=output_dir,
        )

        if use_external:
            ext_path = Path(self._ext_sweep_var.get().strip())
            if not ext_path.exists():
                raise ValueError(f"External sweep file not found: {ext_path}")
            cfg = replace(cfg, external_sweep_path=ext_path)

        return cfg

    def _worker(self, cfg: MeasurementConfig):
        # Runs on a background thread. It must only talk to the queue, never
        # to Tk widgets.
        redirect = LogRedirect(self._msg_queue)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = redirect
        sys.stderr = redirect

        try:
            main(cfg)
            self._msg_queue.put(("status", "Done"))
        except Exception:
            tb = traceback.format_exc()
            self._msg_queue.put(("log", "\n" + tb))
            self._msg_queue.put(("status", "Failed"))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self._msg_queue.put(("done", None))


if __name__ == "__main__":
    app = App()
    app.mainloop()
