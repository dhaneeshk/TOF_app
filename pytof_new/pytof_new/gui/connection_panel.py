"""Connection controls for mock or real hardware."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class ConnectionPanel(QtWidgets.QGroupBox):
    """Display mock or real hardware connection state."""

    connect_requested = QtCore.Signal()
    disconnect_requested = QtCore.Signal()
    simulation_toggled = QtCore.Signal(bool)

    def __init__(self) -> None:
        super().__init__("Connection")
        self.simulation_mode = QtWidgets.QCheckBox("Simulation mode")
        self.simulation_mode.setChecked(True)
        self.digitizer_status = QtWidgets.QLabel("Digitizer: disconnected")
        self.bme_status = QtWidgets.QLabel("BME: disconnected")
        self.real_warning = QtWidgets.QLabel(
            "Real mode uses the BME delay generator for trigger pulses.\n"
            "Verify cabling, levels, and termination before starting."
        )
        self.real_warning.setWordWrap(True)
        self.real_warning.setStyleSheet("color: #d18f00; font-weight: bold;")
        self.real_warning.setVisible(False)
        self.service_state_label = QtWidgets.QLabel("")
        self.service_state_label.setStyleSheet("color: #888; font-size: 10px;")
        self.service_state_label.setVisible(False)
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.disconnect_button = QtWidgets.QPushButton("Disconnect")
        self.disconnect_button.setEnabled(False)
        self.connect_button.setMaximumWidth(90)
        self.disconnect_button.setMaximumWidth(90)
        self._update_tooltips()
        self.simulation_mode.toggled.connect(self._on_simulation_toggled)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(4)
        buttons.addWidget(self.connect_button)
        buttons.addWidget(self.disconnect_button)
        buttons.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.simulation_mode)
        layout.addWidget(self.real_warning)
        layout.addWidget(self.service_state_label)
        layout.addWidget(self.digitizer_status)
        layout.addWidget(self.bme_status)
        layout.addLayout(buttons)

        self.connect_button.clicked.connect(self.connect_requested)
        self.disconnect_button.clicked.connect(self.disconnect_requested)

    def _update_tooltips(self) -> None:
        sim = self.simulation_mode.isChecked()
        if sim:
            self.simulation_mode.setToolTip("Uncheck to connect to real Spectrum digitizer hardware.")
            self.digitizer_status.setToolTip("Connection state of the mock digitizer.")
            self.bme_status.setToolTip("Connection state of the mock BME delay generator.")
            self.connect_button.setToolTip("Connect to the mock digitizer and mock BME delay generator.")
            self.disconnect_button.setToolTip("Disconnect mock hardware when acquisition is idle.")
        else:
            self.simulation_mode.setToolTip("Re-check to return to mock simulation mode.")
            self.digitizer_status.setToolTip("Connection state of the real Spectrum digitizer.")
            self.bme_status.setToolTip("Connection state of the real BME delay generator.")
            self.connect_button.setToolTip("Connect to the real Spectrum digitizer and BME delay generator.")
            self.disconnect_button.setToolTip("Disconnect real hardware when acquisition is idle.")

    def _on_simulation_toggled(self, checked: bool) -> None:
        self._update_tooltips()
        self.real_warning.setVisible(not checked)
        self.service_state_label.setVisible(not checked)
        self.bme_status.setVisible(True)
        self.simulation_toggled.emit(checked)

    def set_service_state(self, state: str) -> None:
        self.service_state_label.setText(f"Service: {state}")
        if state == "connected":
            self.service_state_label.setStyleSheet("color: #4caf50; font-size: 10px;")

    def set_bme_service_state(self, state: str) -> None:
        self.bme_status.setText(f"BME: {state}")
        if state in ("connected", "configured", "armed"):
            self.bme_status.setStyleSheet("color: #4caf50;")
        elif state == "error":
            self.bme_status.setStyleSheet("color: #d9534f;")
        else:
            self.bme_status.setStyleSheet("")

    def set_connecting(self, connecting: bool) -> None:
        """Disable connect button during async connection attempt."""
        self.connect_button.setEnabled(not connecting)
        self.connect_button.setText("Connecting..." if connecting else "Connect")
        if connecting:
            self.service_state_label.setText("Service: connecting...")
            self.service_state_label.setStyleSheet("color: #d18f00; font-size: 10px;")
            self.bme_status.setText("BME: connecting...")

    def simulation_is_active(self) -> bool:
        return self.simulation_mode.isChecked()

    def set_connected(self, connected: bool) -> None:
        """Update displayed connection state."""
        text = "connected" if connected else "disconnected"
        self.digitizer_status.setText(f"Digitizer: {text}")
        self.bme_status.setText(f"BME: {text}")
        self.bme_status.setStyleSheet("")
        self.connect_button.setEnabled(not connected)
        self.disconnect_button.setEnabled(connected)
