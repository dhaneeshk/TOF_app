"""Processing configuration panel."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from pytof_new.config.models import ProcessingConfig
from pytof_new.gui.interaction import set_compact_widths


class ProcessingPanel(QtWidgets.QGroupBox):
    """Collect processing settings."""

    shot_analysis_requested = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__("Processing")
        self.baseline_start = QtWidgets.QSpinBox()
        self.baseline_start.setRange(0, 1_000_000)
        self.baseline_start.setValue(0)
        self.baseline_stop = QtWidgets.QSpinBox()
        self.baseline_stop.setRange(1, 1_000_000)
        self.baseline_stop.setValue(32)
        self.subtract_baseline = QtWidgets.QCheckBox("Subtract baseline")
        self.subtract_baseline.setChecked(True)
        self.high_pass = QtWidgets.QCheckBox("High-pass filter")
        self.high_pass_cutoff_mhz = QtWidgets.QDoubleSpinBox()
        self.high_pass_cutoff_mhz.setRange(0.000001, 1_000_000.0)
        self.high_pass_cutoff_mhz.setDecimals(6)
        self.high_pass_cutoff_mhz.setValue(0.1)
        self.low_pass = QtWidgets.QCheckBox("Low-pass filter")
        self.low_pass_cutoff_mhz = QtWidgets.QDoubleSpinBox()
        self.low_pass_cutoff_mhz.setRange(0.000001, 1_000_000.0)
        self.low_pass_cutoff_mhz.setDecimals(6)
        self.low_pass_cutoff_mhz.setValue(50.0)
        self.filter_order = QtWidgets.QSpinBox()
        self.filter_order.setRange(1, 10)
        self.filter_order.setValue(4)
        self.reference_subtraction = QtWidgets.QCheckBox("Reference subtraction")
        self.reference_path = QtWidgets.QLineEdit("")
        self.reference_status = QtWidgets.QLabel("No reference loaded")
        self.reference_status.setWordWrap(True)
        self.absolute_signal = QtWidgets.QCheckBox("Absolute signal")
        self.smoothing = QtWidgets.QCheckBox("Smoothing")
        self.smoothing_window = QtWidgets.QSpinBox()
        self.smoothing_window.setRange(3, 501)
        self.smoothing_window.setSingleStep(2)
        self.smoothing_window.setValue(11)
        self.smoothing_interval = QtWidgets.QLabel("")
        self.time_zero_us = QtWidgets.QDoubleSpinBox()
        self.time_zero_us.setRange(-1_000_000, 1_000_000)
        self.time_zero_us.setDecimals(3)
        self.shot_analysis_records = QtWidgets.QSpinBox()
        self.shot_analysis_records.setRange(2, 300)
        self.shot_analysis_records.setValue(100)
        self.shot_analysis_delay_ms = QtWidgets.QDoubleSpinBox()
        self.shot_analysis_delay_ms.setRange(10.0, 10_000.0)
        self.shot_analysis_delay_ms.setDecimals(3)
        self.shot_analysis_delay_ms.setValue(10.0)
        self.shot_analysis_max_lag_ns = QtWidgets.QDoubleSpinBox()
        self.shot_analysis_max_lag_ns.setRange(0.001, 1_000_000.0)
        self.shot_analysis_max_lag_ns.setDecimals(3)
        self.shot_analysis_max_lag_ns.setValue(100.0)
        self.shot_analysis_min_mz = QtWidgets.QDoubleSpinBox()
        self.shot_analysis_min_mz.setRange(0.0, 1_000_000.0)
        self.shot_analysis_min_mz.setDecimals(3)
        self.shot_analysis_min_mz.setValue(50.0)
        self.shot_analysis_button = QtWidgets.QPushButton("Shot analysis")
        self.shot_analysis_result = QtWidgets.QLabel("Shot analysis: not run")
        self.shot_analysis_result.setWordWrap(True)
        self.baseline_start.setToolTip("Start sample for the baseline region.")
        self.baseline_stop.setToolTip("Stop sample for the baseline region; should usually stay within pretrigger samples.")
        self.subtract_baseline.setToolTip("Subtract a per-record baseline before averaging or plotting.")
        self.high_pass.setToolTip("Apply a zero-phase high-pass Butterworth filter to each record before averaging.")
        self.high_pass_cutoff_mhz.setToolTip("High-pass cutoff frequency in MHz. Must be below Nyquist.")
        self.low_pass.setToolTip("Apply a zero-phase low-pass Butterworth filter to each record before averaging.")
        self.low_pass_cutoff_mhz.setToolTip("Low-pass cutoff frequency in MHz. Must be below Nyquist.")
        self.filter_order.setToolTip("Butterworth filter order used for enabled high-pass and low-pass filters.")
        self.reference_subtraction.setToolTip("Subtract a saved blank/reference trace from each record before averaging.")
        self.reference_path.setToolTip("HDF5 reference file saved from a blank cumulative spectrum.")
        self.reference_status.setToolTip("Shows whether a reference path has been configured for subtraction.")
        self.absolute_signal.setToolTip("Convert processed records to absolute values before averaging and display.")
        self.smoothing.setToolTip("Apply Savitzky-Golay smoothing to the averaged trace only.")
        self.smoothing_window.setToolTip("Odd Savitzky-Golay smoothing window length in samples.")
        self.smoothing_interval.setToolTip("Approximate time span covered by the smoothing window at the selected sample rate.")
        self.time_zero_us.setToolTip("Time-zero offset in microseconds. Positive values shift the TOF axis earlier by this amount.")
        self.shot_analysis_records.setToolTip("Number of one-record spectra to acquire for cross-correlation jitter analysis.")
        self.shot_analysis_delay_ms.setToolTip("Delay between records to avoid acquisition/readout bottlenecks.")
        self.shot_analysis_max_lag_ns.setToolTip("Maximum lag searched by cross-correlation, in nanoseconds.")
        self.shot_analysis_min_mz.setToolTip("Only use signal at or above this m/z for cross-correlation jitter estimation.")
        self.shot_analysis_button.setToolTip("Acquire delayed single-record spectra and estimate timing jitter by cross-correlation.")
        self.shot_analysis_result.setToolTip("Latest cross-correlation timing-jitter result.")

        form = QtWidgets.QFormLayout(self)
        form.addRow("Baseline start", self.baseline_start)
        form.addRow("Baseline stop", self.baseline_stop)
        form.addRow(self.subtract_baseline)
        form.addRow(self.high_pass)
        form.addRow("High-pass cutoff (MHz)", self.high_pass_cutoff_mhz)
        form.addRow(self.low_pass)
        form.addRow("Low-pass cutoff (MHz)", self.low_pass_cutoff_mhz)
        form.addRow("Filter order", self.filter_order)
        form.addRow(self.reference_subtraction)
        form.addRow("Reference file", self.reference_path)
        form.addRow("Reference status", self.reference_status)
        form.addRow(self.absolute_signal)
        form.addRow(self.smoothing)
        form.addRow("Smoothing window", self.smoothing_window)
        form.addRow("Smoothing interval", self.smoothing_interval)
        form.addRow("Time zero offset (us)", self.time_zero_us)
        form.addRow("Shot records", self.shot_analysis_records)
        form.addRow("Shot delay (ms)", self.shot_analysis_delay_ms)
        form.addRow("Shot max lag (ns)", self.shot_analysis_max_lag_ns)
        form.addRow("Shot min m/z", self.shot_analysis_min_mz)
        form.addRow(self.shot_analysis_button)
        form.addRow("Shot result", self.shot_analysis_result)
        set_compact_widths(self, 105)
        self.reference_path.setMaximumWidth(180)
        self._sample_rate_hz = 1.25e9
        self.shot_analysis_button.clicked.connect(self.shot_analysis_requested)
        self.smoothing_window.valueChanged.connect(lambda _value: self._update_smoothing_interval())
        self.reference_path.textChanged.connect(lambda _value: self._update_reference_status())
        self._update_smoothing_interval()
        self._update_reference_status()

    def config(self, detector_polarity: int, mass_calibration: tuple[float, float, float]) -> ProcessingConfig:
        """Return an immutable processing config snapshot."""
        window = self.effective_smoothing_window()
        return ProcessingConfig(
            detector_polarity=detector_polarity,
            baseline_start=int(self.baseline_start.value()),
            baseline_stop=int(self.baseline_stop.value()),
            subtract_baseline=self.subtract_baseline.isChecked(),
            high_pass_enabled=self.high_pass.isChecked(),
            high_pass_cutoff_hz=self.high_pass_cutoff_mhz.value() * 1e6,
            low_pass_enabled=self.low_pass.isChecked(),
            low_pass_cutoff_hz=self.low_pass_cutoff_mhz.value() * 1e6,
            filter_order=int(self.filter_order.value()),
            reference_subtraction_enabled=self.reference_subtraction.isChecked(),
            reference_path=Path(self.reference_path.text()) if self.reference_path.text().strip() else None,
            absolute_signal_enabled=self.absolute_signal.isChecked(),
            smoothing_enabled=self.smoothing.isChecked(),
            smoothing_window=window,
            time_zero_offset_s=self.time_zero_us.value() * 1e-6,
            mass_calibration_enabled=True,
            mass_calibration=mass_calibration,
        )

    def set_sample_rate_hz(self, sample_rate_hz: float) -> None:
        """Update the sample rate used for the smoothing interval display."""
        self._sample_rate_hz = sample_rate_hz
        self._update_smoothing_interval()

    def effective_smoothing_window(self) -> int:
        """Return the odd smoothing window that will be used by processing."""
        window = int(self.smoothing_window.value())
        if window % 2 == 0:
            window += 1
        return window

    def _update_smoothing_interval(self) -> None:
        window = self.effective_smoothing_window()
        interval_us = window / self._sample_rate_hz * 1e6
        self.smoothing_interval.setText(f"~{interval_us:.3f} us ({window} samples)")

    def set_reference_loaded(self, path: str, record_count: int) -> None:
        """Show that a reference was saved or loaded and is ready for subtraction."""
        self.reference_path.setText(path)
        previous_blocked = self.reference_subtraction.blockSignals(True)
        self.reference_subtraction.setChecked(True)
        self.reference_subtraction.blockSignals(previous_blocked)
        self.reference_status.setText(f"Reference ready: {record_count} records")

    def _update_reference_status(self) -> None:
        if self.reference_path.text().strip():
            self.reference_status.setText("Reference path set")
        else:
            self.reference_status.setText("No reference loaded")

    def shot_analysis_settings(self) -> tuple[int, float, float, float]:
        """Return shot-analysis count, delay seconds, max lag seconds, and min m/z."""
        return (
            int(self.shot_analysis_records.value()),
            self.shot_analysis_delay_ms.value() * 1e-3,
            self.shot_analysis_max_lag_ns.value() * 1e-9,
            self.shot_analysis_min_mz.value(),
        )

    def set_shot_analysis_running(self, running: bool) -> None:
        """Update shot-analysis controls for worker state."""
        self.shot_analysis_button.setEnabled(not running)
        if running:
            self.shot_analysis_result.setText("Shot analysis: running")

    def set_shot_analysis_progress(self, current: int, total: int) -> None:
        """Show shot-analysis acquisition progress."""
        self.shot_analysis_result.setText(f"Shot analysis: {current}/{total} records")

    def set_shot_analysis_result(self, result: object) -> None:
        """Show a jitter result object in nanoseconds."""
        self.shot_analysis_result.setText(
            "Shot analysis: "
            f"RMS {result.rms_jitter_s * 1e9:.3g} ns, "
            f"mean {result.mean_shift_s * 1e9:.3g} ns, "
            f"uncertainty {result.uncertainty_s * 1e9:.3g} ns, "
            f"N={result.record_count}"
        )
