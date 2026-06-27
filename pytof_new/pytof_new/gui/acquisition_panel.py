"""Acquisition controls and status."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class AcquisitionPanel(QtWidgets.QGroupBox):
    """Controls for starting and stopping acquisition."""

    configure_requested = QtCore.Signal()
    start_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()
    clear_requested = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__("Acquisition")
        self.configure_button = QtWidgets.QPushButton("Arm")
        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.continuous = QtWidgets.QCheckBox("Continuous")
        self.continuous.setChecked(True)
        self.polarity = QtWidgets.QComboBox()
        self.polarity.addItems(["positive", "negative"])
        self.polarity.setMaximumWidth(95)
        self.stop_button.setEnabled(False)
        for button in (self.configure_button, self.start_button, self.stop_button, self.clear_button):
            button.setMaximumWidth(95)
        self.configure_button.setToolTip("Validate the current settings and place the mock digitizer in an armed/ready state. Re-running this updates the armed configuration.")
        self.start_button.setToolTip("Start acquisition using the current settings. If not already configured and armed, settings are validated first.")
        self.stop_button.setToolTip("Request a controlled stop after the current record or batch finishes.")
        self.clear_button.setToolTip("Clear plots, counters, and log messages from the current GUI display.")
        self.continuous.setToolTip("When checked, acquisition repeats until Stop is pressed. When unchecked, Start acquires one finite batch.")
        self.polarity.setToolTip("Detector polarity correction and polarity-specific mass calibration selection.")

        self.configure_button.clicked.connect(self.configure_requested)
        self.start_button.clicked.connect(self.start_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.clear_button.clicked.connect(self.clear_requested)

    def set_running(self, running: bool) -> None:
        """Enable controls for current run state."""
        self.start_button.setEnabled(not running)
        self.configure_button.setEnabled(not running)
        self.clear_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def set_progress(self, shots: int, dropped: int) -> None:
        """Update acquisition counters."""
        return

    def set_state(self, state: str) -> None:
        """Update state label."""
        return

    def detector_polarity(self) -> int:
        """Return detector polarity multiplier for processing."""
        return 1 if self.polarity.currentText() == "positive" else -1
