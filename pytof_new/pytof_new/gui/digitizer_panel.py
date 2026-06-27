"""Digitizer configuration panel with basic/advanced mode split."""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from pytof_new.config.models import AcquisitionPriority, AcquisitionWorkflow, DigitizerConfig
from pytof_new.gui.interaction import set_compact_widths


SEGMENT_ALIGNMENT = 32
MIN_SEGMENT_SAMPLES = 64


class DigitizerPanel(QtWidgets.QGroupBox):
    """Collect digitizer settings with basic/advanced mode split."""

    segment_samples_changed = QtCore.Signal()
    settings_changed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__("Digitizer")
        self._repetition_period_s: float | None = None
        self._calibration: tuple[float, float, float] | None = None

        # --- Basic controls ---
        self.basic_group = QtWidgets.QGroupBox("Basic")
        basic_layout = QtWidgets.QFormLayout(self.basic_group)

        self.acquisition_workflow = QtWidgets.QComboBox()
        self.acquisition_workflow.addItem("Live Averaged Spectrum", AcquisitionWorkflow.LIVE_AVERAGED)
        self.acquisition_workflow.addItem("Live Raw-Shot Spectrum", AcquisitionWorkflow.LIVE_RAW)
        self.acquisition_workflow.addItem("Finite Shot Analysis", AcquisitionWorkflow.FINITE_SHOT_ANALYSIS)
        self.acquisition_workflow.setToolTip("What the user is trying to acquire; hardware batch details are planned separately.")

        self.tof_window_us = QtWidgets.QDoubleSpinBox()
        self.tof_window_us.setRange(0.001, 1_000_000.0)
        self.tof_window_us.setDecimals(3)
        self.tof_window_us.setValue(50.0)
        self.tof_window_us.setToolTip("Post-trigger TOF acquisition window in microseconds.")

        self.mz_readout = QtWidgets.QLabel("—")
        self.mz_readout.setToolTip("Corresponding m/z at the end of the TOF window (from current calibration).")

        tof_row = QtWidgets.QHBoxLayout()
        tof_row.addWidget(self.tof_window_us)
        tof_row.addWidget(QtWidgets.QLabel("→ m/z:"))
        tof_row.addWidget(self.mz_readout)
        tof_container = QtWidgets.QWidget()
        tof_container.setLayout(tof_row)

        self.priority = QtWidgets.QComboBox()
        self.priority.addItem("Fast updates", AcquisitionPriority.FAST_UPDATES)
        self.priority.addItem("Balanced", AcquisitionPriority.BALANCED)
        self.priority.addItem("Highest time resolution", AcquisitionPriority.HIGHEST_TIME_RESOLUTION)
        self.priority.addItem("Best signal-to-noise", AcquisitionPriority.BEST_SIGNAL_TO_NOISE)
        self.priority.setCurrentIndex(1)
        self.priority.setToolTip(
            "Fast updates: smaller batches and lower latency. "
            "Balanced: sensible general-purpose default. "
            "Highest time resolution: prefers 1.25 GS/s where valid. "
            "Best signal-to-noise: favors larger averaging and 32-bit accumulation."
        )

        self.basic_input_range = QtWidgets.QComboBox()
        self.basic_input_range.addItem("±200 mV", 0.2)
        self.basic_input_range.addItem("±500 mV", 0.5)
        self.basic_input_range.addItem("±1 V", 1.0)
        self.basic_input_range.addItem("±2.5 V", 2.5)
        self.basic_input_range.setCurrentIndex(1)

        self.target_interval = QtWidgets.QDoubleSpinBox()
        self.target_interval.setRange(0.01, 60.0)
        self.target_interval.setDecimals(2)
        self.target_interval.setValue(0.5)
        self.target_interval.setSuffix(" s")
        self.target_interval.setToolTip("Approximate time between display updates (continuous mode).")

        self.total_shots = QtWidgets.QSpinBox()
        self.total_shots.setRange(1, 10_000_000)
        self.total_shots.setValue(1000)
        self.total_shots.setToolTip("Total physical raw shots in the full finite analysis run.")

        self.plan_preview = QtWidgets.QPlainTextEdit()
        self.plan_preview.setReadOnly(True)
        self.plan_preview.setMinimumHeight(150)
        self.plan_preview.setMaximumHeight(190)
        self.plan_preview.setToolTip("Read-only preview of the planned finite hardware batch and run behavior.")

        basic_layout.addRow("Acquisition mode", self.acquisition_workflow)
        basic_layout.addRow("Acquisition priority", self.priority)
        basic_layout.addRow("TOF window (µs)", tof_container)
        basic_layout.addRow("Input range", self.basic_input_range)
        basic_layout.addRow("Target display interval", self.target_interval)
        basic_layout.addRow("Total shots", self.total_shots)
        basic_layout.addRow("Plan preview", self.plan_preview)

        # --- Advanced mode toggle ---
        self.advanced_checkbox = QtWidgets.QCheckBox("Advanced")
        self.advanced_checkbox.setToolTip("Enable manual override of acquisition parameters.")

        # --- Advanced controls ---
        self.advanced_group = QtWidgets.QGroupBox("Advanced")

        self.sample_rate = QtWidgets.QComboBox()
        self.sample_rate.addItem("1.25 GHz (0.8 ns)", 1.25e9)
        self.sample_rate.addItem("0.625 GHz (1.6 ns)", 0.625e9)
        self.sample_rate.addItem("0.3125 GHz (3.2 ns)", 0.3125e9)
        self.sample_rate.addItem("0.15625 GHz (6.4 ns)", 0.15625e9)

        self.input_range = QtWidgets.QComboBox()
        self.input_range.addItem("+/-200 mV", 0.2)
        self.input_range.addItem("+/-500 mV", 0.5)
        self.input_range.addItem("+/-1 V", 1.0)
        self.input_range.addItem("+/-2.5 V", 2.5)
        self.input_range.setCurrentIndex(1)

        self.accumulator_mode = QtWidgets.QComboBox()
        self.accumulator_mode.addItem("Automatic", "automatic")
        self.accumulator_mode.addItem("32-bit", "32bit")
        self.accumulator_mode.addItem("16-bit", "16bit")

        self.coupling = QtWidgets.QComboBox()
        self.coupling.addItems(["dc", "ac"])

        self.bandwidth = QtWidgets.QCheckBox("Bandwidth limit")

        self.trigger_source = QtWidgets.QComboBox()
        self.trigger_source.addItems(["external0", "software", "channel0"])

        self.trigger_level = QtWidgets.QDoubleSpinBox()
        self.trigger_level.setRange(-5, 5)
        self.trigger_level.setDecimals(3)
        self.trigger_level.setValue(1.5)

        self.trigger_edge = QtWidgets.QComboBox()
        self.trigger_edge.addItems(["rising", "falling"])

        self.termination = QtWidgets.QComboBox()
        self.termination.addItem("50 Ohms", 50)
        self.termination.addItem("1 kOhm", 1000)

        self.pretrigger = QtWidgets.QSpinBox()
        self.pretrigger.setRange(0, 1_000_000)
        self.pretrigger.setValue(32)
        self.pretrigger.setToolTip("Samples recorded before trigger; used for baseline estimation.")

        self.calculated_segment_samples = QtWidgets.QLineEdit()
        self.calculated_segment_samples.setReadOnly(True)
        self.calculated_segment_samples.setToolTip(
            "Total record length including pretrigger samples. "
            "Calculated from TOF window, sample rate, and pretrigger."
        )

        self.override_segment = QtWidgets.QCheckBox("Override calculated segment size")
        self.segment_override = QtWidgets.QSpinBox()
        self.segment_override.setRange(MIN_SEGMENT_SAMPLES, 100_000_000)
        self.segment_override.setSingleStep(SEGMENT_ALIGNMENT)
        self.segment_override.setValue(65536)
        self.segment_override.setEnabled(False)

        self.window_warning = QtWidgets.QLabel("")
        self.window_warning.setWordWrap(True)
        self.window_warning.setMinimumHeight(42)
        self.window_warning.setStyleSheet("color: #d18f00;")

        self.averages_per_record = QtWidgets.QSpinBox()
        self.averages_per_record.setRange(1, 1_000_000)
        self.averages_per_record.setValue(100)
        self.averages_per_record.setToolTip("Number of physical shots summed into one FPGA record (N).")

        self.fpga_sums_per_batch = QtWidgets.QSpinBox()
        self.fpga_sums_per_batch.setRange(1, 1_000_000)
        self.fpga_sums_per_batch.setValue(1)
        self.fpga_sums_per_batch.setToolTip("Number of FPGA-summed records per finite batch (k).")

        self.raw_shots_per_batch = QtWidgets.QSpinBox()
        self.raw_shots_per_batch.setRange(1, 1_000_000)
        self.raw_shots_per_batch.setValue(500)
        self.raw_shots_per_batch.setToolTip("Number of raw segments per batch (continuous RAW_MULTI).")

        self.timeout = QtWidgets.QDoubleSpinBox()
        self.timeout.setRange(0.01, 120)
        self.timeout.setValue(5.0)
        self.timeout.setToolTip("Maximum time to wait for an acquisition operation.")

        # Tooltips for existing controls
        self.sample_rate.setToolTip("Digitizer sample rate. The value in parentheses is the sample interval.")
        self.input_range.setToolTip("Digitizer full-scale input range.")
        self.coupling.setToolTip("Input coupling setting.")
        self.bandwidth.setToolTip("Enable digitizer bandwidth limit when supported.")
        self.trigger_source.setToolTip("Source that defines the digitizer trigger.")
        self.trigger_level.setToolTip("External or channel trigger threshold in volts.")
        self.trigger_edge.setToolTip("Trigger edge polarity.")
        self.termination.setToolTip("Trigger input termination impedance.")

        adv_layout = QtWidgets.QFormLayout(self.advanced_group)
        adv_layout.addRow("Sample rate", self.sample_rate)
        adv_layout.addRow("Input range", self.input_range)
        adv_layout.addRow("Coupling", self.coupling)
        adv_layout.addRow("Filter", self.bandwidth)
        adv_layout.addRow("Trigger source", self.trigger_source)
        adv_layout.addRow("Trigger level (V)", self.trigger_level)
        adv_layout.addRow("Trigger edge", self.trigger_edge)
        adv_layout.addRow("Termination", self.termination)
        adv_layout.addRow("Pretrigger samples", self.pretrigger)
        adv_layout.addRow("Segment samples (incl. pretrigger)", self.calculated_segment_samples)
        adv_layout.addRow(self.override_segment)
        adv_layout.addRow("Segment override", self.segment_override)
        adv_layout.addRow("Accumulator mode", self.accumulator_mode)
        adv_layout.addRow("Shots per FPGA sum", self.averages_per_record)
        adv_layout.addRow("FPGA sums per hardware batch", self.fpga_sums_per_batch)
        adv_layout.addRow("Raw shots per hardware batch", self.raw_shots_per_batch)
        adv_layout.addRow("Timeout (s)", self.timeout)

        # --- Assemble top-level layout ---
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.basic_group)
        content_layout.addWidget(self.advanced_checkbox)
        content_layout.addWidget(self.advanced_group)
        content_layout.addWidget(self.window_warning)
        content_layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.addWidget(scroll)

        set_compact_widths(self, 115)
        self.calculated_segment_samples.setMaximumWidth(115)

        # --- Signals ---
        self.sample_rate.currentIndexChanged.connect(self._update_calculated_segment_samples)
        self.pretrigger.valueChanged.connect(self._update_calculated_segment_samples)
        self.tof_window_us.valueChanged.connect(self._update_calculated_segment_samples)
        self.acquisition_workflow.currentIndexChanged.connect(self._update_workflow_visibility)
        self.advanced_checkbox.toggled.connect(self.advanced_group.setEnabled)
        self.advanced_checkbox.toggled.connect(self._on_advanced_toggled)
        self.override_segment.toggled.connect(self.segment_override.setEnabled)
        for widget in (
            self.acquisition_workflow, self.priority, self.basic_input_range, self.sample_rate,
            self.input_range, self.coupling, self.trigger_source, self.trigger_edge,
            self.termination, self.accumulator_mode,
        ):
            widget.currentIndexChanged.connect(self.settings_changed)
        for widget in (
            self.tof_window_us, self.target_interval, self.total_shots, self.trigger_level,
            self.pretrigger, self.averages_per_record, self.fpga_sums_per_batch,
            self.raw_shots_per_batch, self.timeout, self.segment_override,
        ):
            widget.valueChanged.connect(self.settings_changed)
        self.bandwidth.toggled.connect(self.settings_changed)
        self.advanced_checkbox.toggled.connect(self.settings_changed)
        self.override_segment.toggled.connect(self.settings_changed)

        self._update_calculated_segment_samples()
        self._update_workflow_visibility()
        self.advanced_group.setEnabled(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def config(self) -> DigitizerConfig:
        """Return an immutable digitizer config snapshot."""
        return DigitizerConfig(
            record_mode=self.record_mode(),
            sample_rate_hz=float(self.sample_rate.currentData()),
            input_range_v=float(self.input_range.currentData() if self.advanced_checkbox.isChecked() else self.basic_input_range.currentData()),
            coupling=self.coupling.currentText(),
            bandwidth_limit_enabled=self.bandwidth.isChecked(),
            trigger_source=self.trigger_source.currentText(),
            trigger_level_v=float(self.trigger_level.value()),
            trigger_edge=self.trigger_edge.currentText(),
            trigger_termination_ohm=int(self.termination.currentData()),
            pretrigger_samples=int(self.pretrigger.value()),
            segment_samples=self.segment_samples(),
            number_of_segments=1,
            hardware_averages_per_record=int(self.averages_per_record.value()),
            timeout_s=float(self.timeout.value()),
            # Planner fields
            tof_window_us=float(self.tof_window_us.value()),
            acquisition_workflow=self.acquisition_workflow.currentData(),
            acquisition_priority=self.priority.currentData(),
            target_update_interval_s=float(self.target_interval.value()),
            total_shots=int(self.total_shots.value()),
            advanced_mode=self.advanced_checkbox.isChecked(),
            accumulator_mode=self.accumulator_mode.currentData(),
            fpga_sums_per_batch=int(self.fpga_sums_per_batch.value()),
            raw_shots_per_batch=int(self.raw_shots_per_batch.value()),
            override_segment_samples=self.override_segment.isChecked(),
        )

    def record_mode(self) -> str:
        workflow = self.acquisition_workflow.currentData()
        if workflow == AcquisitionWorkflow.LIVE_AVERAGED:
            return "hardware_average"
        return "raw_segments"

    def manual_segment_samples(self) -> int | None:
        if self.advanced_checkbox.isChecked() and self.override_segment.isChecked():
            return int(self.segment_override.value())
        return None

    def set_plan_preview(self, lines: list[str] | tuple[str, ...], warnings: list[str] | tuple[str, ...] = ()) -> None:
        text_lines = list(lines)
        if warnings and not any(line == "Warnings:" for line in text_lines):
            text_lines.append("Warnings:")
            text_lines.extend(f"WARNING: {warning}" for warning in warnings)
        self.plan_preview.setPlainText("\n".join(text_lines))

    def set_plan_error(self, message: str) -> None:
        self.plan_preview.setPlainText(f"Plan error:\n{message}")

    def set_repetition_period_s(self, repetition_period_s: float) -> None:
        """Set the BME repetition period used for window warnings."""
        self._repetition_period_s = repetition_period_s
        self._update_window_warning()

    def set_calibration(self, coefficients: tuple[float, float, float] | None) -> None:
        """Set the mass calibration coefficients for the m/z readout."""
        self._calibration = coefficients
        self._update_mz_readout()

    def segment_samples(self) -> int:
        """Return the aligned segment size including pretrigger samples."""
        if self.override_segment.isChecked():
            return int(self.segment_override.value())
        return calculate_segment_samples(
            tof_window_s=self.tof_window_us.value() * 1e-6,
            sample_rate_hz=float(self.sample_rate.currentData()),
            pretrigger_samples=int(self.pretrigger.value()),
        )

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _update_calculated_segment_samples(self) -> None:
        seg = self.segment_samples()
        self.calculated_segment_samples.setText(str(seg))
        self._update_window_warning()
        self._update_mz_readout()
        self.segment_samples_changed.emit()

    def _update_window_warning(self) -> None:
        if self._repetition_period_s is None:
            self.window_warning.setText("")
            return
        tof_window_s = self.tof_window_us.value() * 1e-6
        if tof_window_s > self._repetition_period_s:
            self.window_warning.setText(
                "Warning: TOF window exceeds BME repetition period.\n"
                "Later triggers may arrive before this record completes."
            )
        else:
            self.window_warning.setText("")

    def _update_mz_readout(self) -> None:
        if self._calibration is None:
            self.mz_readout.setText("—")
            return
        from pytof_new.hardware.acquisition_planner import mz_at_window_end
        mz = mz_at_window_end(
            tof_window_us=self.tof_window_us.value(),
            pretrigger_samples=int(self.pretrigger.value()),
            sample_rate_hz=float(self.sample_rate.currentData()),
            calibration=self._calibration,
        )
        if mz is not None:
            self.mz_readout.setText(f"{mz:.1f}")
        else:
            self.mz_readout.setText("—")

    def _update_workflow_visibility(self) -> None:
        """Show/hide controls depending on workflow."""
        workflow = self.acquisition_workflow.currentData()
        is_averaged = workflow == AcquisitionWorkflow.LIVE_AVERAGED
        is_finite = workflow == AcquisitionWorkflow.FINITE_SHOT_ANALYSIS
        self.target_interval.setVisible(not is_finite)
        self.total_shots.setVisible(is_finite)
        self.accumulator_mode.setVisible(is_averaged)
        self.averages_per_record.setVisible(is_averaged)
        self.fpga_sums_per_batch.setVisible(is_averaged)
        self.raw_shots_per_batch.setVisible(not is_averaged)
        self.priority.setEnabled(True)

    def _on_advanced_toggled(self, checked: bool) -> None:
        self._update_workflow_visibility()
        if checked:
            self.priority.setEnabled(True)
            self.target_interval.setEnabled(True)


def calculate_segment_samples(tof_window_s: float, sample_rate_hz: float, pretrigger_samples: int) -> int:
    """Calculate aligned segment samples including pretrigger."""
    if tof_window_s <= 0:
        raise ValueError("tof_window_s must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if pretrigger_samples < 0:
        raise ValueError("pretrigger_samples must be non-negative")
    post_trigger_samples = math.ceil(tof_window_s * sample_rate_hz)
    requested = pretrigger_samples + post_trigger_samples
    return max(MIN_SEGMENT_SAMPLES, round_up_to_multiple(requested, SEGMENT_ALIGNMENT))


def round_up_to_multiple(value: int, multiple: int) -> int:
    """Round an integer up to the nearest multiple."""
    return ((value + multiple - 1) // multiple) * multiple
