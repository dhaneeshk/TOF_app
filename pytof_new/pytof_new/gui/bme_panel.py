"""BME delay generator configuration panel."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from pytof_new.config.models import BMEConfig
from pytof_new.gui.interaction import set_compact_widths


CHANNELS = ("A", "B", "C", "D", "E", "F")


class BMEPanel(QtWidgets.QGroupBox):
    """Collect BME timing with basic/advanced mode split."""

    settings_changed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__("BME Delay Generator")
        self._tof_window_us = 50.0

        self.basic_group = QtWidgets.QGroupBox("Basic")
        basic_layout = QtWidgets.QFormLayout(self.basic_group)
        self.extraction_fill_us = _spin(0.001, 1_000_000, 55.0)
        self.extraction_fill_us.setToolTip(
            "Repetition period of the BME trigger pulses = TOF window (us) "
            "defined in digitizer panel + extraction region fill time (us)."
        )
        self.basic_repetition_readout = QtWidgets.QLineEdit()
        self.basic_repetition_readout.setReadOnly(True)
        self.basic_repetition_readout.setToolTip("Calculated BME repetition period in microseconds.")
        basic_layout.addRow("Extraction region fill time (us)", self.extraction_fill_us)
        basic_layout.addRow("Calculated repetition (us)", self.basic_repetition_readout)

        self.advanced_checkbox = QtWidgets.QCheckBox("Advanced")
        self.advanced_checkbox.setToolTip("Enable manual BME timing, channel, polarity, and termination settings.")

        self.advanced_group = QtWidgets.QGroupBox("Advanced")
        adv_layout = QtWidgets.QFormLayout(self.advanced_group)
        self.repetition_us = _spin(0.001, 1_000_000, 105.0)
        self.digitizer_channel = _channel_combo("A")
        self.push_channel = _channel_combo("C")
        self.pull_channel = _channel_combo("F")
        self.digitizer_polarity = _polarity_combo(True)
        self.push_polarity = _polarity_combo(True)
        self.pull_polarity = _polarity_combo(False)
        self.digitizer_width_us = _spin(0.001, 1_000_000, 50.0)
        self.push_width_us = _spin(0.001, 1_000_000, 50.0)
        self.pull_width_us = _spin(0.001, 1_000_000, 50.0)
        self.digitizer_delay_us = _spin(0, 1_000_000, 0.0)
        self.push_delay_us = _spin(0, 1_000_000, 0.0)
        self.pull_delay_us = _spin(0, 1_000_000, 0.0)
        self.trigger_termination = QtWidgets.QComboBox()
        self.trigger_termination.addItem("50 Ohms", 50)
        self.trigger_termination.addItem("1 kOhm / high-Z", 1000)
        self.output_state = QtWidgets.QLabel("Outputs: disabled")

        self.repetition_us.setToolTip("Time between the start of BME trigger pulse sequences.")
        self.digitizer_channel.setToolTip("BME output channel used to trigger the Spectrum digitizer.")
        self.push_channel.setToolTip("BME output channel used for the PUSH extraction trigger.")
        self.pull_channel.setToolTip("BME output channel used for the PULL extraction trigger.")
        self.digitizer_width_us.setToolTip("Digitizer trigger pulse width; defaults to the TOF window and must be below repetition period.")
        self.push_width_us.setToolTip("PUSH trigger pulse width; defaults to the TOF window and must be below repetition period.")
        self.pull_width_us.setToolTip("PULL trigger pulse width; defaults to the TOF window and must be below repetition period.")
        self.digitizer_delay_us.setToolTip("Digitizer trigger delay from BME cycle start; must be below TOF window.")
        self.push_delay_us.setToolTip("PUSH trigger delay from BME cycle start; must be below TOF window.")
        self.pull_delay_us.setToolTip("PULL trigger delay from BME cycle start; must be below TOF window.")
        self.trigger_termination.setToolTip("BME trigger/output termination setting used by the real BME driver.")
        self.output_state.setToolTip("Current output-enable state reported by the acquisition workflow.")

        adv_layout.addRow("Repetition period (us)", self.repetition_us)
        adv_layout.addRow("Digitizer trigger channel", self.digitizer_channel)
        adv_layout.addRow("PUSH trigger channel", self.push_channel)
        adv_layout.addRow("PULL trigger channel", self.pull_channel)
        adv_layout.addRow("Digitizer trigger polarity", self.digitizer_polarity)
        adv_layout.addRow("PUSH trigger polarity", self.push_polarity)
        adv_layout.addRow("PULL trigger polarity", self.pull_polarity)
        adv_layout.addRow("Digitizer width (us)", self.digitizer_width_us)
        adv_layout.addRow("PUSH width (us)", self.push_width_us)
        adv_layout.addRow("PULL width (us)", self.pull_width_us)
        adv_layout.addRow("Digitizer delay (us)", self.digitizer_delay_us)
        adv_layout.addRow("PUSH delay (us)", self.push_delay_us)
        adv_layout.addRow("PULL delay (us)", self.pull_delay_us)
        adv_layout.addRow("Trigger termination", self.trigger_termination)
        adv_layout.addRow(self.output_state)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.basic_group)
        content_layout.addWidget(self.advanced_checkbox)
        content_layout.addWidget(self.advanced_group)
        content_layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(scroll)

        self.extraction_fill_us.valueChanged.connect(self._on_basic_changed)
        self.advanced_checkbox.toggled.connect(self.advanced_group.setEnabled)
        self.advanced_checkbox.toggled.connect(self._on_advanced_toggled)
        for widget in (self.digitizer_channel, self.push_channel, self.pull_channel, self.digitizer_polarity, self.push_polarity, self.pull_polarity, self.trigger_termination):
            widget.currentIndexChanged.connect(self.settings_changed)
        for widget in (
            self.repetition_us,
            self.digitizer_width_us,
            self.push_width_us,
            self.pull_width_us,
            self.digitizer_delay_us,
            self.push_delay_us,
            self.pull_delay_us,
        ):
            widget.valueChanged.connect(self.settings_changed)

        set_compact_widths(self, 115)
        self.basic_repetition_readout.setMaximumWidth(115)
        self._sync_basic_derived_values(emit_signal=False)
        self.advanced_group.setEnabled(False)

    def set_tof_window_us(self, tof_window_us: float) -> None:
        """Set the digitizer TOF window used for basic BME timing defaults."""
        self._tof_window_us = float(tof_window_us)
        self._sync_basic_derived_values(emit_signal=True)

    def config(self) -> BMEConfig:
        """Return an immutable BME config snapshot."""
        return BMEConfig(
            advanced_mode=self.advanced_checkbox.isChecked(),
            tof_window_s=self._tof_window_us * 1e-6,
            extraction_region_fill_time_s=self.extraction_fill_us.value() * 1e-6,
            repetition_period_s=self.repetition_us.value() * 1e-6,
            digitizer_trigger_delay_s=self.digitizer_delay_us.value() * 1e-6,
            push_trigger_delay_s=self.push_delay_us.value() * 1e-6,
            pull_trigger_delay_s=self.pull_delay_us.value() * 1e-6,
            digitizer_trigger_width_s=self.digitizer_width_us.value() * 1e-6,
            push_trigger_width_s=self.push_width_us.value() * 1e-6,
            pull_trigger_width_s=self.pull_width_us.value() * 1e-6,
            digitizer_channel=self.digitizer_channel.currentText(),
            push_channel=self.push_channel.currentText(),
            pull_channel=self.pull_channel.currentText(),
            digitizer_polarity_positive=bool(self.digitizer_polarity.currentData()),
            push_polarity_positive=bool(self.push_polarity.currentData()),
            pull_polarity_positive=bool(self.pull_polarity.currentData()),
            trigger_termination_ohm=int(self.trigger_termination.currentData()),
        )

    def set_outputs_enabled(self, enabled: bool) -> None:
        """Update output status label."""
        self.output_state.setText(f"Outputs: {'enabled' if enabled else 'disabled'}")

    def _on_basic_changed(self) -> None:
        self._sync_basic_derived_values(emit_signal=True)

    def _on_advanced_toggled(self, checked: bool) -> None:
        if not checked:
            self._sync_basic_derived_values(emit_signal=False)
        self.settings_changed.emit()

    def _sync_basic_derived_values(self, *, emit_signal: bool) -> None:
        repetition = self._tof_window_us + self.extraction_fill_us.value()
        self.basic_repetition_readout.setText(f"{repetition:.3f}")
        if not self.advanced_checkbox.isChecked():
            self._set_spin_silently(self.repetition_us, repetition)
            self._set_spin_silently(self.digitizer_width_us, self._tof_window_us)
            self._set_spin_silently(self.push_width_us, self._tof_window_us)
            self._set_spin_silently(self.pull_width_us, self._tof_window_us)
            self._set_spin_silently(self.digitizer_delay_us, 0.0)
            self._set_spin_silently(self.push_delay_us, 0.0)
            self._set_spin_silently(self.pull_delay_us, 0.0)
        if emit_signal:
            self.settings_changed.emit()

    @staticmethod
    def _set_spin_silently(spin: QtWidgets.QDoubleSpinBox, value: float) -> None:
        blocked = spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(blocked)


def _spin(minimum: float, maximum: float, value: float) -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(3)
    spin.setValue(value)
    return spin


def _channel_combo(default: str) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox()
    combo.addItems(CHANNELS)
    combo.setCurrentText(default)
    return combo


def _polarity_combo(default_positive: bool) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox()
    combo.addItem("POS", True)
    combo.addItem("NEG", False)
    combo.setCurrentIndex(0 if default_positive else 1)
    return combo
