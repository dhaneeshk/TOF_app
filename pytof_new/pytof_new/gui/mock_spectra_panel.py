"""Mock spectrum generation settings."""

from __future__ import annotations

import math

from PySide6 import QtWidgets

from pytof_new.config.models import MockSpectraConfig
from pytof_new.gui.interaction import set_compact_widths


class MockSpectraPanel(QtWidgets.QGroupBox):
    """Collect mock-only synthetic spectrum settings."""

    def __init__(self) -> None:
        super().__init__("Mock Spectra")
        self.ion_peaks_enabled = QtWidgets.QCheckBox("Enable ion peaks")
        self.ion_peaks_enabled.setChecked(True)
        self.timing_jitter_ns = QtWidgets.QDoubleSpinBox()
        self.timing_jitter_ns.setRange(0.0, 1_000_000.0)
        self.timing_jitter_ns.setDecimals(3)
        self.timing_jitter_ns.setValue(0.0)
        self.resolving_power = QtWidgets.QDoubleSpinBox()
        self.resolving_power.setRange(1.0, 1_000_000.0)
        self.resolving_power.setDecimals(1)
        self.resolving_power.setValue(1000.0)
        self.noise_rms_mv = QtWidgets.QDoubleSpinBox()
        self.noise_rms_mv.setRange(0.0, 1_000_000.0)
        self.noise_rms_mv.setDecimals(3)
        self.noise_rms_mv.setValue(3.0)
        self.ringing_enabled = QtWidgets.QCheckBox("Enable t=0 ringing")
        self.ringing_amplitude_mv = QtWidgets.QDoubleSpinBox()
        self.ringing_amplitude_mv.setRange(0.0, 1_000_000.0)
        self.ringing_amplitude_mv.setDecimals(3)
        self.ringing_amplitude_mv.setValue(0.0)
        self.ringing_frequency_mhz = QtWidgets.QDoubleSpinBox()
        self.ringing_frequency_mhz.setRange(0.001, 1_000_000.0)
        self.ringing_frequency_mhz.setDecimals(3)
        self.ringing_frequency_mhz.setValue(18.0)
        self.ringing_decay_us = QtWidgets.QDoubleSpinBox()
        self.ringing_decay_us.setRange(0.001, 1_000_000.0)
        self.ringing_decay_us.setDecimals(3)
        self.ringing_decay_us.setValue(2.5)
        self.ringing_phase_deg = QtWidgets.QDoubleSpinBox()
        self.ringing_phase_deg.setRange(-360.0, 360.0)
        self.ringing_phase_deg.setDecimals(1)
        self.ringing_phase_deg.setValue(0.0)
        self.ringing_follows_timing_jitter = QtWidgets.QCheckBox("Ringing follows timing jitter")
        self.ion_peaks_enabled.setToolTip("Turn mock ion peaks on or off while leaving noise, ringing, and timing jitter settings available.")
        self.timing_jitter_ns.setToolTip(
            "RMS timing jitter applied independently to each internal mock spectrum before hardware averaging."
        )
        self.resolving_power.setToolTip(
            "Mock resolving power m/dm. Larger values produce narrower intrinsic peak widths before jitter broadening."
        )
        self.noise_rms_mv.setToolTip("Gaussian random noise RMS amplitude per mock spectrum in millivolts.")
        self.ringing_enabled.setToolTip("Add an optional damped cosine artifact beginning at t=0 in mock traces.")
        self.ringing_amplitude_mv.setToolTip("Initial t=0 ringing amplitude in millivolts before damping.")
        self.ringing_frequency_mhz.setToolTip("Ringing frequency in MHz.")
        self.ringing_decay_us.setToolTip("Exponential ringing decay constant in microseconds.")
        self.ringing_phase_deg.setToolTip("Ringing phase in degrees for A*cos(2*pi*f*t + phase).")
        self.ringing_follows_timing_jitter.setToolTip(
            "When enabled, each internally averaged mock spectrum applies the same timing jitter to ringing and peaks."
        )

        form = QtWidgets.QFormLayout(self)
        form.addRow(self.ion_peaks_enabled)
        form.addRow("Timing jitter RMS (ns)", self.timing_jitter_ns)
        form.addRow("Resolving power m/dm", self.resolving_power)
        form.addRow("Random noise RMS (mV)", self.noise_rms_mv)
        form.addRow(self.ringing_enabled)
        form.addRow("Ringing amplitude (mV)", self.ringing_amplitude_mv)
        form.addRow("Ringing frequency (MHz)", self.ringing_frequency_mhz)
        form.addRow("Ringing decay constant (us)", self.ringing_decay_us)
        form.addRow("Ringing phase (deg)", self.ringing_phase_deg)
        form.addRow(self.ringing_follows_timing_jitter)
        set_compact_widths(self, 105)

    def config(self) -> MockSpectraConfig:
        """Return immutable mock spectrum settings."""
        return MockSpectraConfig(
            ion_peaks_enabled=self.ion_peaks_enabled.isChecked(),
            timing_jitter_s=self.timing_jitter_ns.value() * 1e-9,
            resolving_power=self.resolving_power.value(),
            noise_rms_v=self.noise_rms_mv.value() * 1e-3,
            ringing_enabled=self.ringing_enabled.isChecked(),
            ringing_amplitude_v=self.ringing_amplitude_mv.value() * 1e-3,
            ringing_frequency_hz=self.ringing_frequency_mhz.value() * 1e6,
            ringing_decay_s=self.ringing_decay_us.value() * 1e-6,
            ringing_phase_rad=math.radians(self.ringing_phase_deg.value()),
            ringing_follows_timing_jitter=self.ringing_follows_timing_jitter.isChecked(),
        )
