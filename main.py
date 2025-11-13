from __future__ import annotations

import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import mlab
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional dependency
    sd = None  # type: ignore

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional dependency
    sf = None  # type: ignore

try:
    from moviepy.editor import AudioFileClip
except Exception:  # pragma: no cover - optional dependency
    AudioFileClip = None  # type: ignore


WAVE_COLORS: Dict[str, str] = {
    "Delta": "#00bcd4",
    "Theta": "#4caf50",
    "Alpha": "#ffc107",
    "Beta": "#ff5722",
    "Gamma": "#9c27b0",
}


@dataclass
class ClassificationResult:
    label: str
    scores: Dict[str, float]
    dominant_frequency: float
    signal_energy: float


class SignalProcessor:
    """Responsible for cleaning signals, computing spectra, spectrograms and classifications."""

    def __init__(self) -> None:
        self.band_definitions: List[Tuple[str, Tuple[float, float]]] = [
            ("Delta", (0.5, 4.0)),
            ("Theta", (4.0, 8.0)),
            ("Alpha", (8.0, 13.0)),
            ("Beta", (13.0, 30.0)),
            ("Gamma", (30.0, 100.0)),
        ]

    @property
    def band_names(self) -> List[str]:
        return [name for name, _ in self.band_definitions]

    def classify(self, raw_signal: Sequence[float], sample_rate: int) -> Optional[ClassificationResult]:
        if sample_rate <= 0:
            return None

        signal = self._prepare_signal(raw_signal)
        if signal.size < 32:
            return None

        freqs, spectrum = self._power_spectrum(signal, sample_rate)
        if freqs.size == 0:
            return None

        band_energy: Dict[str, float] = {}
        for name, (low, high) in self.band_definitions:
            mask = (freqs >= low) & (freqs < high)
            band_energy[name] = float(np.mean(spectrum[mask])) if np.any(mask) else 0.0

        total = sum(band_energy.values()) or 1.0
        normalized = {name: energy / total for name, energy in band_energy.items()}
        dominant = max(normalized, key=normalized.get)
        dominant_freq = float(freqs[np.argmax(spectrum)])
        signal_energy = float(np.sqrt(np.mean(signal**2)))

        return ClassificationResult(
            label=dominant,
            scores=normalized,
            dominant_frequency=dominant_freq,
            signal_energy=signal_energy,
        )

    def top_frequencies(
        self, raw_signal: Sequence[float], sample_rate: int, count: int = 5
    ) -> List[Tuple[float, float]]:
        """Return the most energetic frequency bins for display purposes."""
        signal = self._prepare_signal(raw_signal)
        freqs, spectrum = self._power_spectrum(signal, sample_rate)
        if freqs.size == 0:
            return []

        indices = np.argsort(spectrum)[::-1]
        peaks: List[Tuple[float, float]] = []
        for idx in indices:
            freq = float(freqs[idx])
            amp = float(spectrum[idx])
            if freq <= 0:
                continue
            peaks.append((freq, amp))
            if len(peaks) == count:
                break
        return peaks

    def compute_spectrogram(
        self,
        raw_signal: Sequence[float],
        sample_rate: int,
        nfft: int = 512,
        overlap: int = 384,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate a spectrogram suitable for visualization."""
        if sample_rate <= 0:
            return None

        signal = np.asarray(raw_signal, dtype=np.float64)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        if signal.size < nfft:
            return None
        spectrum, freqs, times = mlab.specgram(
            signal,
            NFFT=nfft,
            Fs=sample_rate,
            noverlap=overlap,
            window=mlab.window_hanning,
        )
        return freqs, times, spectrum

    @staticmethod
    def _prepare_signal(raw_signal: Sequence[float]) -> np.ndarray:
        data = np.asarray(raw_signal, dtype=np.float64)
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        data = data - np.mean(data)
        std = np.std(data)
        if std > 0:
            data = data / std
        window = np.hanning(data.size)
        return data * window

    @staticmethod
    def _power_spectrum(signal: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        spectrum = np.abs(np.fft.rfft(signal)) ** 2
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
        return freqs, spectrum

class WavePlotCanvas(FigureCanvas):
    """Matplotlib canvas that renders optional waveform, spectrogram, timeline e distribuição de bandas."""

    def __init__(
        self,
        bands: Sequence[str],
        *,
        show_signal: bool = True,
        include_spectrogram: bool = False,
        include_timeline: bool = False,
    ) -> None:
        self.show_signal = show_signal
        self.include_spectrogram = include_spectrogram
        self.include_timeline = include_timeline
        self.figure = Figure(figsize=(7, 5), tight_layout=True)
        super().__init__(self.figure)

        height_ratios: List[float] = []
        if self.show_signal:
            height_ratios.append(1.0)
        if self.include_spectrogram:
            height_ratios.append(1.4)
        if self.include_timeline:
            height_ratios.append(0.35)
        height_ratios.append(0.9)  # bands

        grid = self.figure.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios)
        slot = 0
        self.ax_signal = self.figure.add_subplot(grid[slot]) if self.show_signal else None
        if self.show_signal:
            slot += 1
        self.ax_spectrogram = self.figure.add_subplot(grid[slot]) if self.include_spectrogram else None
        if self.include_spectrogram:
            slot += 1
        self.ax_classes = self.figure.add_subplot(grid[slot]) if self.include_timeline else None
        if self.include_timeline:
            slot += 1
        self.ax_bands = self.figure.add_subplot(grid[slot])

        self.bands = list(bands)
        self.spec_mesh = None
        self.spec_colorbar = None
        self._configure_axes()

    def _configure_axes(self) -> None:
        if self.ax_signal is not None:
            self.ax_signal.set_title("Sinal")
            self.ax_signal.set_xlabel("Tempo (s)")
            self.ax_signal.set_ylabel("Amplitude normalizada")
            self.ax_signal.grid(True, alpha=0.3)

        if self.ax_spectrogram is not None:
            self.ax_spectrogram.set_title("Espectrograma")
            self.ax_spectrogram.set_ylabel("Frequência (Hz)")
            self.ax_spectrogram.set_xlabel("Tempo (s)")

        if self.ax_classes is not None:
            self.ax_classes.set_title("Classificação ao longo do tempo")
            self.ax_classes.set_yticks([])
            self.ax_classes.set_ylabel("")

        self.ax_bands.set_title("Distribuição de bandas")
        self.ax_bands.set_ylabel("% Energia")
        self.ax_bands.set_ylim(0, 100)

    def reset_visual_state(self) -> None:
        """Clear cached colorbar when um novo espectrograma é carregado."""
        if self.spec_colorbar is not None:
            try:
                self.spec_colorbar.remove()
            except Exception:
                pass
            self.spec_colorbar = None
        self.spec_mesh = None

    def update_plot(
        self,
        display_signal: Optional[np.ndarray],
        sample_rate: int,
        classification: Optional[ClassificationResult],
        spectrogram: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        playback_marker: Optional[float] = None,
        total_duration: Optional[float] = None,
        highlight_window: Optional[Tuple[float, float]] = None,
        timeline_segments: Optional[List[Tuple[float, float, str]]] = None,
        freq_limit: Optional[float] = None,
    ) -> None:
        if self.ax_signal is not None:
            self.ax_signal.cla()
        self.ax_bands.cla()
        if self.ax_spectrogram is not None:
            self.ax_spectrogram.cla()
        if self.ax_classes is not None:
            self.ax_classes.cla()
        self._configure_axes()

        plotted_duration = None
        if self.ax_signal is not None and display_signal is not None and display_signal.size > 0:
            plotted_duration = display_signal.size / float(sample_rate)
            x_axis = np.linspace(0, plotted_duration, display_signal.size, endpoint=False)
            self.ax_signal.plot(x_axis, display_signal, color="#26c6da", linewidth=1.0)
            self.ax_signal.set_xlim(0, total_duration or plotted_duration or 1.0)

        if highlight_window and total_duration and self.ax_signal is not None:
            start, end = highlight_window
            self.ax_signal.axvspan(max(start, 0.0), min(end, total_duration), color="#ffeb3b", alpha=0.25)

        if self.ax_spectrogram is not None and spectrogram:
            freqs, times, spec = spectrogram
            spec_db = 10 * np.log10(spec + 1e-9)
            # realça detalhes removendo a mediana por frequência (parecido com Audacity)
            med = np.median(spec_db, axis=1, keepdims=True)
            spec_db = spec_db - med
            vmin = float(np.percentile(spec_db, 20))
            vmax = float(np.percentile(spec_db, 99))
            if vmin >= vmax:
                vmax = vmin + 1.0
            self.spec_mesh = self.ax_spectrogram.pcolormesh(
                times,
                freqs,
                spec_db,
                shading="auto",
                cmap="turbo",
                vmin=vmin,
                vmax=vmax,
            )
            if self.spec_colorbar is None:
                self.spec_colorbar = self.figure.colorbar(self.spec_mesh, ax=self.ax_spectrogram, pad=0.01, label="dB")
            else:
                self.spec_colorbar.update_normal(self.spec_mesh)
            if timeline_segments:
                for start, end, label in timeline_segments:
                    color = WAVE_COLORS.get(label, "#455a64")
                    self.ax_spectrogram.axvspan(start, end, color=color, alpha=0.18)
            if playback_marker is not None:
                self.ax_spectrogram.axvline(
                    playback_marker, color="#ffffff", linestyle="--", linewidth=1.2, alpha=0.8
                )
            if total_duration:
                self.ax_spectrogram.set_xlim(0, total_duration)
            if highlight_window:
                start, end = highlight_window
                self.ax_spectrogram.axvspan(start, end, color="#ffffff", alpha=0.12)
            if freq_limit:
                self.ax_spectrogram.set_ylim(0, freq_limit)

        if self.ax_classes is not None:
            duration = total_duration or plotted_duration or 1.0
            self.ax_classes.set_xlim(0, duration)
            self.ax_classes.set_ylim(0, 1)
            if timeline_segments:
                for start, end, label in timeline_segments:
                    color = WAVE_COLORS.get(label, "#455a64")
                    self.ax_classes.axvspan(start, end, color=color, alpha=0.4)
            if playback_marker is not None:
                self.ax_classes.axvline(playback_marker, color="#ffffff", linestyle="--", linewidth=1.0, alpha=0.8)

        if classification:
            percentages = [classification.scores.get(band, 0.0) * 100 for band in self.bands]
            colors = [WAVE_COLORS.get(band, "#607d8b") for band in self.bands]
            self.ax_bands.bar(self.bands, percentages, color=colors)
            self.ax_bands.set_ylim(0, 100)

        if playback_marker is not None and (total_duration or plotted_duration) and self.ax_signal is not None:
            limit = total_duration or plotted_duration
            self.ax_signal.axvline(
                min(playback_marker, limit),
                color="#ffeb3b",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
            )

        self.draw_idle()


class LiveStreamWorker(QtCore.QObject):
    """Background worker that pulls audio chunks from sounddevice for live monitoring."""

    chunk_ready = QtCore.pyqtSignal(object, int)
    stream_error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, sample_rate: int, chunk_duration: float, device_index: Optional[int]) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_duration = max(0.5, float(chunk_duration))
        self.device_index = device_index
        self._running = False
        self._stream: Optional["sd.InputStream"] = None  # type: ignore[name-defined]
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()

    @QtCore.pyqtSlot()
    def start_stream(self) -> None:
        if sd is None:
            self.stream_error.emit("Instale o pacote 'sounddevice' para habilitar a captura ao vivo.")
            self.finished.emit()
            return

        blocksize = int(self.sample_rate * self.chunk_duration)
        blocksize = max(blocksize, 512)

        def callback(indata, frames, time_info, status):  # type: ignore[override]
            if status:
                self.stream_error.emit(str(status))
            if self._running:
                self._queue.put(indata.copy().reshape(-1))

        try:
            self._running = True
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=blocksize,
                callback=callback,
                device=self.device_index,
            )
            self._stream.start()
            while self._running:
                try:
                    chunk = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if chunk is None:
                    break
                self.chunk_ready.emit(chunk, self.sample_rate)
        except Exception as exc:  # pragma: no cover - device specific paths
            self.stream_error.emit(str(exc))
        finally:
            self._running = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                finally:
                    self._stream = None
            self.finished.emit()

    @QtCore.pyqtSlot()
    def stop_stream(self) -> None:
        self._running = False
        self._queue.put(None)

class FilePlaybackWorker(QtCore.QObject):
    """Streams a loaded recording chunk-by-chunk, optionally tocando áudio via sounddevice."""

    chunk_ready = QtCore.pyqtSignal(object, int, float, float)
    progress_changed = QtCore.pyqtSignal(float, float)
    playback_error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(
        self,
        data: np.ndarray,
        sample_rate: int,
        chunk_duration: float,
        enable_audio: bool = True,
        start_second: float = 0.0,
    ) -> None:
        super().__init__()
        self.data = data.astype(np.float32)
        self.sample_rate = sample_rate
        self.chunk_duration = max(0.25, chunk_duration)
        self.enable_audio = enable_audio and sd is not None
        self._chunk_size = max(512, int(self.sample_rate * self.chunk_duration))
        self._running = False
        self._paused = False
        self._stream: Optional["sd.OutputStream"] = None  # type: ignore[name-defined]
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._total_samples = self.data.size
        self._total_duration = self._total_samples / float(self.sample_rate)
        start_second = float(np.clip(start_second, 0.0, self._total_duration))
        self._start_sample = int(start_second * self.sample_rate)
        self._cursor = self._start_sample

    @QtCore.pyqtSlot()
    def start(self) -> None:
        if self.enable_audio:
            try:
                self._stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    blocksize=self._chunk_size,
                    callback=self._audio_callback,
                )
                self._stream.start()
            except Exception as exc:  # pragma: no cover - audio specific
                self.enable_audio = False
                self.playback_error.emit(f"Áudio desabilitado: {exc}")

        self._running = True
        while self._running and self._cursor < self._total_samples:
            if self._paused:
                time.sleep(0.05)
                continue
            chunk = self.data[self._cursor : self._cursor + self._chunk_size]
            if chunk.size == 0:
                break
            start_time = (self._cursor) / self.sample_rate
            self._cursor += chunk.size
            end_time = self._cursor / self.sample_rate
            if self.enable_audio and self._stream is not None:
                if chunk.size < self._chunk_size:
                    pad = np.zeros(self._chunk_size, dtype=np.float32)
                    pad[: chunk.size] = chunk
                    self._queue.put(pad)
                else:
                    self._queue.put(chunk)
            self.chunk_ready.emit(
                chunk.copy(),
                self.sample_rate,
                max(start_time, 0.0),
                min(end_time, self._total_duration),
            )
            self.progress_changed.emit(min(end_time, self._total_duration), self._total_duration)
            time.sleep(self._chunk_size / self.sample_rate)
        self._cleanup()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._running = False
        self._paused = False

    def _audio_callback(self, outdata, frames, time_info, status):  # type: ignore[override]
        if status:  # pragma: no cover - device specific
            self.playback_error.emit(str(status))
        try:
            chunk = self._queue.get_nowait()
        except queue.Empty:
            outdata[:] = 0
            return
        out = chunk[:frames].reshape(-1, 1)
        outdata[: out.shape[0]] = out
        if out.shape[0] < frames:
            outdata[out.shape[0] :] = 0

    def _cleanup(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self.finished.emit()

class NeuroWavesWindow(QtWidgets.QMainWindow):
    """Main application window with dual panels for offline review and live monitoring."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NeuroWaves · Classificador de Ondas")
        self.resize(1400, 820)
        QtWidgets.QApplication.setStyle("Fusion")

        self.processor = SignalProcessor()
        self.offline_canvas = WavePlotCanvas(
            self.processor.band_names,
            show_signal=False,
            include_spectrogram=True,
            include_timeline=True,
        )
        self.live_canvas = WavePlotCanvas(self.processor.band_names, show_signal=True, include_spectrogram=False)

        self.history_limit = 120
        self.current_signal: Optional[np.ndarray] = None
        self.file_signal: Optional[np.ndarray] = None
        self.file_sample_rate: Optional[int] = None
        self.file_spectrogram: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.file_duration: float = 0.0
        self.precomputed_segments: List[Tuple[float, float, str]] = []
        self.current_playback_time = 0.0
        self._scrub_active = False
        self.last_offline_result: Optional[ClassificationResult] = None
        self.spec_freq_limit = 60.0

        self.live_worker: Optional[LiveStreamWorker] = None
        self.live_thread: Optional[QtCore.QThread] = None
        self.live_running = False

        self.playback_worker: Optional[FilePlaybackWorker] = None
        self.playback_thread: Optional[QtCore.QThread] = None
        self.playing = False

        self._build_ui()
        self.chunk_spin.valueChanged.connect(self._handle_window_change)
        self._populate_devices()
        self._update_live_buttons()
        self.statusBar().showMessage("Pronto.")
        self._update_offline_transport_state()

    # --- UI blocks -----------------------------------------------------------------
    def _build_ui(self) -> None:
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_offline_tab(), "Análise Offline")
        self.tabs.addTab(self._build_live_tab(), "Monitor Ao Vivo")
        self.setCentralWidget(self.tabs)

    def _build_offline_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(20)

        control_panel = QtWidgets.QFrame()
        control_panel.setMaximumWidth(420)
        control_panel.setStyleSheet(
            """
            QFrame {
                background-color: #0f1b2f;
                border-radius: 18px;
                color: #e0f2ff;
            }
            QPushButton {
                background-color: #3949ab;
                border-radius: 8px;
                padding: 10px;
                color: white;
                font-weight: 600;
            }
            QPushButton:disabled {
                background-color: #37474f;
                color: #9ba7b1;
            }
            QListWidget {
                background-color: #0b1523;
                border: none;
                color: #e3f2fd;
            }
        """
        )
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setSpacing(16)

        title = QtWidgets.QLabel("Biblioteca & Playback")
        title.setFont(QtGui.QFont("Segoe UI", 16, QtGui.QFont.Bold))
        control_layout.addWidget(title)

        self.load_button = QtWidgets.QPushButton("Carregar gravação EEG / áudio")
        self.load_button.clicked.connect(self.load_recording)
        control_layout.addWidget(self.load_button)

        self.file_label = QtWidgets.QLabel("Nenhum arquivo selecionado.")
        self.file_label.setWordWrap(True)
        control_layout.addWidget(self.file_label)

        playback_box = QtWidgets.QGridLayout()
        playback_box.setHorizontalSpacing(8)

        self.offline_play_button = QtWidgets.QPushButton("▶ Reproduzir")
        self.offline_play_button.clicked.connect(self._handle_play_button)
        playback_box.addWidget(self.offline_play_button, 0, 0)

        self.offline_pause_button = QtWidgets.QPushButton("⏸ Pausar")
        self.offline_pause_button.clicked.connect(self.toggle_pause)
        playback_box.addWidget(self.offline_pause_button, 0, 1)

        self.offline_stop_button = QtWidgets.QPushButton("⏹ Parar")
        self.offline_stop_button.clicked.connect(self.stop_file_playback)
        playback_box.addWidget(self.offline_stop_button, 0, 2)

        self.offline_rewind_button = QtWidgets.QPushButton("⏪ -5s")
        self.offline_rewind_button.clicked.connect(lambda: self.rewind_playback(seconds=5.0))
        playback_box.addWidget(self.offline_rewind_button, 1, 0)

        self.offline_restart_button = QtWidgets.QPushButton("⏮ Reiniciar")
        self.offline_restart_button.clicked.connect(self.restart_playback)
        playback_box.addWidget(self.offline_restart_button, 1, 1)

        control_layout.addLayout(playback_box)

        self.scrub_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrub_slider.setRange(0, 1000)
        self.scrub_slider.setValue(0)
        self.scrub_slider.sliderPressed.connect(self._on_scrub_start)
        self.scrub_slider.sliderReleased.connect(self._on_scrub_end)
        self.scrub_slider.valueChanged.connect(self._on_scrub_move)
        self.scrub_slider.setEnabled(False)
        control_layout.addWidget(self.scrub_slider)

        self.playback_time_label = QtWidgets.QLabel("00:00 / 00:00")
        control_layout.addWidget(self.playback_time_label)

        self.spec_limit_spin = QtWidgets.QSpinBox()
        self.spec_limit_spin.setRange(20, 200)
        self.spec_limit_spin.setSingleStep(5)
        self.spec_limit_spin.setValue(int(self.spec_freq_limit))
        self.spec_limit_spin.setSuffix(" Hz")
        self.spec_limit_spin.valueChanged.connect(self._handle_spec_limit_change)
        control_layout.addWidget(self._labeled_widget("Zoom espectral (limite superior)", self.spec_limit_spin))

        freq_title = QtWidgets.QLabel("Principais frequências")
        freq_title.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
        control_layout.addWidget(freq_title)

        self.freq_list = QtWidgets.QListWidget()
        control_layout.addWidget(self.freq_list)

        history_title = QtWidgets.QLabel("Histórico de diagnósticos")
        history_title.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
        control_layout.addWidget(history_title)

        self.history_list = QtWidgets.QListWidget()
        control_layout.addWidget(self.history_list, 1)

        layout.addWidget(control_panel, 0)

        visual_panel = QtWidgets.QFrame()
        visual_panel.setStyleSheet(
            """
            QFrame {
                background-color: #08121f;
                border-radius: 18px;
            }
        """
        )
        visual_layout = QtWidgets.QVBoxLayout(visual_panel)
        visual_layout.setContentsMargins(20, 20, 20, 20)
        visual_layout.setSpacing(16)

        self.offline_wave_label = QtWidgets.QLabel("--")
        self.offline_wave_label.setFont(QtGui.QFont("Segoe UI", 32, QtGui.QFont.Bold))
        self.offline_frequency_label = QtWidgets.QLabel("Frequência dominante: -- Hz")
        self.offline_energy_label = QtWidgets.QLabel("Energia: --")
        self.offline_now_badge = QtWidgets.QLabel("Aguardando sinal")
        self.offline_now_badge.setAlignment(QtCore.Qt.AlignCenter)
        self.offline_now_badge.setStyleSheet(
            "QLabel { border-radius: 12px; padding: 6px 10px; background-color: #1e2b3d; color: #e0f2ff; }"
        )
        summary_box = QtWidgets.QVBoxLayout()
        summary_box.addWidget(self.offline_wave_label)
        summary_box.addWidget(self.offline_now_badge)
        summary_box.addWidget(self.offline_frequency_label)
        summary_box.addWidget(self.offline_energy_label)
        visual_layout.addLayout(summary_box)

        visual_layout.addWidget(self.offline_canvas, 1)

        layout.addWidget(visual_panel, 1)
        return tab
    def _build_live_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setSpacing(12)

        self.device_combo = QtWidgets.QComboBox()
        device_wrapper = self._labeled_widget("Dispositivo de captura", self.device_combo)
        top_bar.addWidget(device_wrapper, 1)

        self.refresh_devices_button = QtWidgets.QPushButton("Atualizar")
        self.refresh_devices_button.clicked.connect(self._populate_devices)
        top_bar.addWidget(self.refresh_devices_button)

        self.sample_rate_spin = QtWidgets.QSpinBox()
        self.sample_rate_spin.setRange(500, 96000)
        self.sample_rate_spin.setValue(2000)
        self.sample_rate_spin.setSuffix(" Hz")
        top_bar.addWidget(self._labeled_widget("Taxa de amostragem", self.sample_rate_spin))

        self.chunk_spin = QtWidgets.QDoubleSpinBox()
        self.chunk_spin.setRange(0.5, 5.0)
        self.chunk_spin.setSingleStep(0.5)
        self.chunk_spin.setValue(2.0)
        self.chunk_spin.setSuffix(" s")
        top_bar.addWidget(self._labeled_widget("Janela de análise", self.chunk_spin))

        self.start_button = QtWidgets.QPushButton("Iniciar Monitoramento")
        self.start_button.clicked.connect(self.start_live_stream)
        self.stop_button = QtWidgets.QPushButton("Parar")
        self.stop_button.clicked.connect(self.stop_live_stream)
        top_bar.addWidget(self.start_button)
        top_bar.addWidget(self.stop_button)

        layout.addLayout(top_bar)

        live_summary = QtWidgets.QFrame()
        live_summary.setStyleSheet(
            """
            QFrame {
                background-color: #0f1d33;
                border-radius: 16px;
                color: #e8f0ff;
            }
        """
        )
        summary_layout = QtWidgets.QHBoxLayout(live_summary)
        summary_layout.setContentsMargins(20, 20, 20, 20)
        summary_layout.setSpacing(30)

        self.live_wave_label = QtWidgets.QLabel("--")
        self.live_wave_label.setFont(QtGui.QFont("Segoe UI", 34, QtGui.QFont.Bold))
        summary_layout.addWidget(self.live_wave_label)

        live_meta = QtWidgets.QVBoxLayout()
        self.live_frequency_label = QtWidgets.QLabel("Frequência dominante: -- Hz")
        self.live_energy_label = QtWidgets.QLabel("Energia: --")
        live_meta.addWidget(self.live_frequency_label)
        live_meta.addWidget(self.live_energy_label)
        summary_layout.addLayout(live_meta)

        layout.addWidget(live_summary)
        layout.addWidget(self.live_canvas, 1)
        return tab

    def _labeled_widget(self, label: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        caption = QtWidgets.QLabel(label)
        caption.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
        layout.addWidget(caption)
        layout.addWidget(widget)
        return wrapper

    # --- Device handling -----------------------------------------------------------
    def _populate_devices(self) -> None:
        self.device_combo.clear()
        if sd is None:
            self.device_combo.addItem("sounddevice não instalado")
            self.start_button.setEnabled(False)
            self.statusBar().showMessage("Instale 'sounddevice' para captura ao vivo.")
            return

        try:
            devices = sd.query_devices()
        except Exception as exc:  # pragma: no cover - host specific
            self.device_combo.addItem("Erro ao consultar dispositivos")
            self.start_button.setEnabled(False)
            self.statusBar().showMessage(str(exc))
            return

        has_device = False
        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                label = f"{index}: {device.get('name', 'Entrada')} ({device['max_input_channels']}ch)"
                self.device_combo.addItem(label, index)
                has_device = True

        if not has_device:
            self.device_combo.addItem("Nenhum dispositivo de entrada encontrado")
            self.start_button.setEnabled(False)
            self.statusBar().showMessage("Conecte um EEG/EMG via P2 ou interface de áudio.")
        else:
            self.start_button.setEnabled(True)
            self.device_combo.setCurrentIndex(0)

    # --- Live stream ---------------------------------------------------------------
    def start_live_stream(self) -> None:
        if self.live_running or sd is None:
            return

        device = self.device_combo.currentData()
        sample_rate = int(self.sample_rate_spin.value())
        chunk_duration = float(self.chunk_spin.value())

        self.live_worker = LiveStreamWorker(sample_rate, chunk_duration, device)
        self.live_thread = QtCore.QThread()
        self.live_worker.moveToThread(self.live_thread)
        self.live_thread.started.connect(self.live_worker.start_stream)
        self.live_worker.chunk_ready.connect(self._handle_live_chunk)
        self.live_worker.stream_error.connect(self.statusBar().showMessage)
        self.live_worker.finished.connect(self._handle_stream_finished)
        self.live_thread.start()
        self.live_running = True
        self.statusBar().showMessage("Monitoramento ao vivo iniciado.")
        self._update_live_buttons()

    def stop_live_stream(self) -> None:
        if not self.live_running:
            return
        if self.live_worker:
            self.live_worker.stop_stream()
        if self.live_thread:
            self.live_thread.quit()
            self.live_thread.wait()
            self.live_thread = None
        self.live_worker = None
        self.live_running = False
        self.statusBar().showMessage("Monitoramento ao vivo encerrado.")
        self._update_live_buttons()

    def _handle_live_chunk(self, chunk: np.ndarray, sample_rate: int) -> None:
        self.current_signal = chunk
        result = self.processor.classify(chunk, sample_rate)
        self.live_canvas.update_plot(chunk, sample_rate, result)
        if result:
            self._update_live_summary(result)
            self._append_history("Ao vivo", result)

    def _handle_stream_finished(self) -> None:
        if self.live_thread:
            self.live_thread.quit()
            self.live_thread.wait()
            self.live_thread = None
        self.live_worker = None
        self.live_running = False
        self._update_live_buttons()

    def _update_live_buttons(self) -> None:
        self.start_button.setEnabled(not self.live_running and self.device_combo.count() > 0)
        self.stop_button.setEnabled(self.live_running)

    def _update_live_summary(self, result: ClassificationResult) -> None:
        self.live_wave_label.setText(result.label)
        self.live_wave_label.setStyleSheet(f"color: {WAVE_COLORS.get(result.label, '#ffffff')};")
        self.live_frequency_label.setText(f"Frequência dominante: {result.dominant_frequency:0.1f} Hz")
        self.live_energy_label.setText(f"Energia: {result.signal_energy:0.2f}")
    # --- Offline analysis ----------------------------------------------------------
    def load_recording(self) -> None:
        if sf is None:
            self.statusBar().showMessage("Instale o pacote 'soundfile' para abrir gravações.")
            return

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Selecionar gravação EEG/áudio",
            str(Path.home()),
            "Áudio (*.wav *.flac *.ogg *.mp3);;Todos os arquivos (*.*)",
        )
        if not file_path:
            return

        raw_data = None
        sample_rate = None
        try:
            if sf is not None:
                raw_data, sample_rate = sf.read(file_path, dtype="float32")
        except Exception:
            raw_data = None
            sample_rate = None

        if raw_data is None or sample_rate is None:
            suffix = Path(file_path).suffix.lower()
            if suffix == ".mkv":
                if AudioFileClip is None:
                    self.statusBar().showMessage("Instale 'moviepy' + 'ffmpeg' para abrir arquivos MKV.")
                    return
                try:
                    clip = AudioFileClip(file_path)
                    sample_rate = int(clip.fps)
                    raw_data = clip.to_soundarray(fps=sample_rate).astype("float32")
                except Exception as exc:  # pragma: no cover - depends on codecs
                    self.statusBar().showMessage(f"Falha ao extrair áudio do MKV: {exc}")
                    return
                finally:
                    try:
                        clip.close()
                    except Exception:
                        pass
            else:
                self.statusBar().showMessage("Falha ao carregar arquivo. Converta para WAV/FLAC ou instale suporte MKV.")
                return

        data = raw_data
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        self.file_signal = data
        self.file_sample_rate = int(sample_rate)
        self.file_spectrogram = self.processor.compute_spectrogram(data, self.file_sample_rate)
        self.file_duration = len(data) / float(self.file_sample_rate)
        self.file_label.setText(f"Arquivo: {Path(file_path).name}")
        self.stop_file_playback()
        self.offline_canvas.reset_visual_state()
        window_sec = float(self.chunk_spin.value())
        self._prepare_offline_analysis(data, self.file_sample_rate, window_sec)
        self.statusBar().showMessage("Gravação carregada. Pronto para análise e playback.")

        result = self.processor.classify(data, self.file_sample_rate)
        self.last_offline_result = result
        self._update_offline_summary(result)
        self._update_frequency_list(data, self.file_sample_rate)
        self._render_offline_overview()
        self.scrub_slider.setValue(0)
        self.playback_time_label.setText(f"00:00 / {self._format_time(self.file_duration)}")
        self._update_offline_transport_state()

    def _handle_play_button(self) -> None:
        if self.playback_worker and self.playing:
            return
        if self.playback_worker and not self.playing:
            self.playback_worker.resume()
            self.playing = True
            self.statusBar().showMessage("Reprodução retomada.")
            self._update_offline_transport_state()
            return
        start_point = self.current_playback_time if self.current_playback_time > 0 else 0.0
        self.start_file_playback(start_second=start_point)

    def start_file_playback(self, start_second: Optional[float] = None) -> None:
        if self.file_signal is None or self.file_sample_rate is None:
            self.statusBar().showMessage("Carregue um arquivo antes de reproduzir.")
            return

        target = start_second if start_second is not None else 0.0
        target = max(0.0, min(target, self.file_duration or 0.0))
        self.stop_file_playback(reset_position=False)
        self.current_playback_time = target

        self.playback_worker = FilePlaybackWorker(
            self.file_signal,
            self.file_sample_rate,
            chunk_duration=float(self.chunk_spin.value()),
            enable_audio=True,
            start_second=target,
        )
        self.playback_thread = QtCore.QThread()
        self.playback_worker.moveToThread(self.playback_thread)
        self.playback_thread.started.connect(self.playback_worker.start)
        self.playback_worker.chunk_ready.connect(self._handle_playback_chunk)
        self.playback_worker.progress_changed.connect(self._update_playback_progress)
        self.playback_worker.playback_error.connect(self.statusBar().showMessage)
        self.playback_worker.finished.connect(self._handle_playback_finished)
        self.playback_thread.start()
        self.playing = True
        if self.file_duration:
            slider_value = int((target / self.file_duration) * 1000)
            self.scrub_slider.setValue(slider_value)
        self.statusBar().showMessage(f"Reprodução iniciada em {self._format_time(target)}.")
        self._update_offline_transport_state()

    def toggle_pause(self) -> None:
        if not self.playback_worker:
            return
        if self.playing:
            self.playback_worker.pause()
            self.playing = False
            self.statusBar().showMessage("Reprodução pausada.")
        else:
            self.playback_worker.resume()
            self.playing = True
            self.statusBar().showMessage("Reprodução retomada.")
        self._update_offline_transport_state()

    def stop_file_playback(self, reset_position: bool = True) -> None:
        if self.playback_worker:
            self.playback_worker.stop()
        if self.playback_thread:
            self.playback_thread.quit()
            self.playback_thread.wait()
            self.playback_thread = None
        self.playback_worker = None
        self.playing = False
        if reset_position:
            self.current_playback_time = 0.0
            self.scrub_slider.setValue(0)
            self.playback_time_label.setText("00:00 / 00:00")
        self._scrub_active = False
        self._update_offline_transport_state()

    def _handle_playback_chunk(self, chunk: np.ndarray, sample_rate: int, start_time: float, end_time: float) -> None:
        self.current_signal = chunk
        result = self.processor.classify(chunk, sample_rate)
        total_duration = self.file_duration or (
            len(self.file_signal) / self.file_sample_rate if self.file_signal is not None else None
        )
        self.current_playback_time = end_time
        self.offline_canvas.update_plot(
            None,
            self.file_sample_rate or sample_rate,
            result,
            self.file_spectrogram,
            playback_marker=end_time,
            total_duration=total_duration,
            highlight_window=(start_time, end_time),
            timeline_segments=self.precomputed_segments,
            freq_limit=self.spec_freq_limit,
        )
        if result:
            self._update_offline_summary(result)

    def _handle_playback_finished(self) -> None:
        if self.playback_thread:
            self.playback_thread.quit()
            self.playback_thread.wait()
            self.playback_thread = None
        self.playback_worker = None
        self.playing = False
        self.statusBar().showMessage("Reprodução concluída.")
        self._update_offline_transport_state()

    def _on_scrub_start(self) -> None:
        if self.file_duration <= 0:
            return
        self._scrub_active = True

    def _on_scrub_end(self) -> None:
        if self.file_duration <= 0:
            self._scrub_active = False
            return
        self._scrub_active = False
        target = (self.scrub_slider.value() / 1000) * self.file_duration
        self.seek_file_playback(target)

    def _on_scrub_move(self, value: int) -> None:
        if not self._scrub_active or self.file_duration <= 0:
            return
        preview = (value / 1000) * self.file_duration
        self.playback_time_label.setText(f"{self._format_time(preview)} / {self._format_time(self.file_duration)}")

    def seek_file_playback(self, position: float) -> None:
        if self.file_signal is None or self.file_sample_rate is None or self.file_duration <= 0:
            return
        position = max(0.0, min(position, self.file_duration))
        was_playing = self.playing
        self.stop_file_playback(reset_position=False)
        self.current_playback_time = position
        slider_value = int((position / self.file_duration) * 1000)
        self.scrub_slider.setValue(slider_value)
        self.playback_time_label.setText(f"{self._format_time(position)} / {self._format_time(self.file_duration)}")
        highlight = (position, min(position + float(self.chunk_spin.value()), self.file_duration))
        self._render_offline_overview(highlight=highlight)
        if was_playing:
            self.start_file_playback(start_second=position)

    def rewind_playback(self, seconds: float = 5.0) -> None:
        self.seek_file_playback(max(0.0, self.current_playback_time - seconds))

    def restart_playback(self) -> None:
        self.seek_file_playback(0.0)

    def _update_playback_progress(self, current: float, total: float) -> None:
        if total <= 0:
            return
        percent = int((current / total) * 1000)
        if not self._scrub_active:
            self.scrub_slider.setValue(min(percent, 1000))
        self.playback_time_label.setText(f"{self._format_time(current)} / {self._format_time(total)}")

    def _update_offline_summary(self, result: Optional[ClassificationResult]) -> None:
        if result:
            self.offline_wave_label.setText(result.label)
            self.offline_wave_label.setStyleSheet(f"color: {WAVE_COLORS.get(result.label, '#ffffff')};")
            self.offline_frequency_label.setText(f"Frequência dominante: {result.dominant_frequency:0.1f} Hz")
            self.offline_energy_label.setText(f"Energia: {result.signal_energy:0.2f}")
            self.offline_now_badge.setText(f"Ao vivo · {result.label}")
            badge_color = WAVE_COLORS.get(result.label, "#1e2b3d")
            text_color = "#050d1b" if badge_color != "#1e2b3d" else "#e0f2ff"
            self.offline_now_badge.setStyleSheet(
                f"QLabel {{ border-radius: 12px; padding: 6px 10px; background-color: {badge_color}; color: {text_color}; }}"
            )
            self.last_offline_result = result
        else:
            self.offline_wave_label.setText("--")
            self.offline_frequency_label.setText("Frequência dominante: -- Hz")
            self.offline_energy_label.setText("Energia: --")
            self.offline_now_badge.setText("Aguardando sinal")
            self.offline_now_badge.setStyleSheet(
                "QLabel { border-radius: 12px; padding: 6px 10px; background-color: #1e2b3d; color: #e0f2ff; }"
            )

    def _update_frequency_list(self, data: np.ndarray, sample_rate: int) -> None:
        self.freq_list.clear()
        peaks = self.processor.top_frequencies(data, sample_rate)
        if not peaks:
            self.freq_list.addItem("Sem dados suficientes para espectro.")
            return
        max_amp = max(amp for _, amp in peaks) or 1.0
        for freq, amp in peaks:
            percent = amp / max_amp * 100
            item = QtWidgets.QListWidgetItem(f"{freq:6.1f} Hz  ·  {percent:4.1f}% energia relativa")
            self.freq_list.addItem(item)

    def _update_offline_transport_state(self) -> None:
        has_file = self.file_signal is not None and self.file_duration > 0
        has_worker = self.playback_worker is not None
        self.offline_play_button.setEnabled(has_file)
        self.offline_pause_button.setEnabled(has_worker)
        self.offline_stop_button.setEnabled(has_worker)
        self.offline_rewind_button.setEnabled(has_file)
        self.offline_restart_button.setEnabled(has_file)
        self.scrub_slider.setEnabled(has_file)

    def _render_offline_overview(self, highlight: Optional[Tuple[float, float]] = None) -> None:
        if self.file_signal is None or self.file_sample_rate is None:
            return
        self.offline_canvas.update_plot(
            None,
            self.file_sample_rate,
            self.last_offline_result,
            self.file_spectrogram,
            total_duration=self.file_duration,
            timeline_segments=self.precomputed_segments,
            highlight_window=highlight,
            freq_limit=self.spec_freq_limit,
        )

    def _append_history(self, source: str, result: ClassificationResult) -> None:
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {source} · {result.label} · {result.dominant_frequency:0.1f} Hz"
        self.history_list.insertItem(0, entry)
        if self.history_list.count() > self.history_limit:
            self.history_list.takeItem(self.history_list.count() - 1)

    def _prepare_offline_analysis(self, data: np.ndarray, sample_rate: int, window_sec: float) -> None:
        self.precomputed_segments = []
        if sample_rate <= 0 or window_sec <= 0 or data.size == 0:
            return
        win_samples = max(int(window_sec * sample_rate), int(sample_rate * 0.5))
        win_samples = min(win_samples, data.size)
        if win_samples <= 0:
            return
        hop = max(win_samples // 2, sample_rate // 5, 1)
        limit = 6000
        idx = 0
        while idx + win_samples <= data.size and len(self.precomputed_segments) < limit:
            chunk = data[idx : idx + win_samples]
            result = self.processor.classify(chunk, sample_rate)
            if result:
                start = idx / sample_rate
                end = (idx + win_samples) / sample_rate
                self.precomputed_segments.append((start, end, result.label))
            idx += hop
        if not self.precomputed_segments:
            result = self.processor.classify(data, sample_rate)
            if result:
                self.precomputed_segments.append((0.0, self.file_duration, result.label))

    def _handle_window_change(self, value: int) -> None:
        if self.file_signal is None or self.file_sample_rate is None:
            return
        new_window = max(float(value), 0.25)
        if self.playback_worker:
            self.stop_file_playback()
        self.statusBar().showMessage("Atualizando análise offline...")
        self.offline_canvas.reset_visual_state()
        self._prepare_offline_analysis(self.file_signal, self.file_sample_rate, new_window)
        self._render_offline_overview()
        self.statusBar().showMessage("Janela de análise atualizada.")

    def _handle_spec_limit_change(self, value: int) -> None:
        self.spec_freq_limit = float(value)
        highlight = None
        if self.current_playback_time > 0 and self.file_duration > 0:
            highlight = (
                self.current_playback_time,
                min(self.current_playback_time + float(self.chunk_spin.value()), self.file_duration),
            )
        self._render_offline_overview(highlight=highlight)

    @staticmethod
    def _format_time(value: float) -> str:
        minutes, seconds = divmod(int(value), 60)
        return f"{minutes:02d}:{seconds:02d}"

    # --- Qt lifecycle --------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - UI hook
        self.stop_live_stream()
        self.stop_file_playback()
        super().closeEvent(event)

def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = NeuroWavesWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
