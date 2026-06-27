"""Mass calibration peak collection panel."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from pytof_new.processing.calibration import fit_mass_calibration, mass_to_tof_ns


DEFAULT_MASS_CALIBRATION = (3.5e-6, 0.0129, 6.27)


class CalibrationPanel(QtWidgets.QGroupBox):
    """Collect observed peak positions and known m/z values for calibration."""

    save_calibration_ini_requested = QtCore.Signal()

    def __init__(self, rows: int = 6) -> None:
        super().__init__("Calibration")
        self.collect_peaks = QtWidgets.QCheckBox("Collect calibration peaks")
        self.observed_header = QtWidgets.QLabel("Observed TOF (us)")
        self.known_header = QtWidgets.QLabel("Known m/z")
        self.status = QtWidgets.QLabel("Collection disabled")
        self.status.setWordWrap(True)
        self.observed_edits: list[QtWidgets.QLineEdit] = []
        self.known_mass_edits: list[QtWidgets.QLineEdit] = []
        self.clear_observed_button = QtWidgets.QPushButton("Clear observed")
        self.clear_all_button = QtWidgets.QPushButton("Clear all")
        self.determine_button = QtWidgets.QPushButton("Determine calibration constants")
        self.save_calibration_ini_button = QtWidgets.QPushButton("Save calib to ini")
        self.a_result = QtWidgets.QLineEdit()
        self.b_result = QtWidgets.QLineEdit()
        self.c_result = QtWidgets.QLineEdit()
        for result in (self.a_result, self.b_result, self.c_result):
            result.setReadOnly(True)
        self._axis_mode = "TOF"
        self._current_coefficients = DEFAULT_MASS_CALIBRATION

        self.collect_peaks.setToolTip("When enabled, right-drag peak fits populate the next empty observed position row.")
        self.observed_header.setToolTip("Observed peak center in the current plot axis units.")
        self.known_header.setToolTip("Known mass-to-charge value corresponding to the observed peak.")
        self.clear_observed_button.setToolTip("Clear only the observed peak position column.")
        self.clear_all_button.setToolTip("Clear both observed peak positions and known m/z entries.")
        self.determine_button.setToolTip("Fit m/z = A*t_ns^2 + B*t_ns + C from at least three completed rows.")
        self.save_calibration_ini_button.setToolTip("Save the determined calibration constants to PyTOF.ini for the current polarity.")
        for button in (self.clear_observed_button, self.clear_all_button, self.determine_button, self.save_calibration_ini_button):
            button.setMaximumWidth(180)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.observed_header, 0, 0)
        grid.addWidget(self.known_header, 0, 1)
        for row in range(rows):
            observed = QtWidgets.QLineEdit()
            known = QtWidgets.QLineEdit()
            observed.setPlaceholderText(f"Peak {row + 1}")
            known.setPlaceholderText(f"m/z {row + 1}")
            self.observed_edits.append(observed)
            self.known_mass_edits.append(known)
            grid.addWidget(observed, row + 1, 0)
            grid.addWidget(known, row + 1, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.clear_observed_button)
        buttons.addWidget(self.clear_all_button)
        buttons.addStretch(1)

        results = QtWidgets.QFormLayout()
        results.addRow("A (m/z/ns^2)", self.a_result)
        results.addRow("B (m/z/ns)", self.b_result)
        results.addRow("C (m/z)", self.c_result)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.collect_peaks)
        layout.addLayout(grid)
        layout.addLayout(buttons)
        layout.addWidget(self.determine_button)
        layout.addLayout(results)
        layout.addWidget(self.status)
        layout.addWidget(self.save_calibration_ini_button)

        self.collect_peaks.toggled.connect(self._update_collection_status)
        self.clear_observed_button.clicked.connect(self.clear_observed)
        self.clear_all_button.clicked.connect(self.clear_all)
        self.determine_button.clicked.connect(self.determine_calibration_constants)
        self.save_calibration_ini_button.clicked.connect(self.save_calibration_ini_requested)

    def add_peak_fit(self, selection: object) -> None:
        """Add a fitted peak center to the next empty observed row."""
        if not self.collect_peaks.isChecked():
            return
        for edit in self.observed_edits:
            if not edit.text().strip():
                edit.setText(f"{selection.center:.9g}")
                self.status.setText(f"Added {selection.axis_mode} peak center: {selection.center:.9g}")
                return
        self.status.setText("Calibration table is full")

    def handle_axis_changed(self, axis_mode: str) -> None:
        """Clear observed positions when the plotted x-axis units change."""
        self.clear_observed()
        self._axis_mode = axis_mode
        if axis_mode == "Mass":
            self.observed_header.setText("Observed m/z")
        else:
            self.observed_header.setText("Observed TOF (us)")
        self.status.setText("Observed positions cleared because plot axis changed")

    def clear_observed(self) -> None:
        """Clear observed peak positions only."""
        for edit in self.observed_edits:
            edit.clear()

    def clear_all(self) -> None:
        """Clear observed positions and known m/z values."""
        self.clear_observed()
        for edit in self.known_mass_edits:
            edit.clear()
        for edit in (self.a_result, self.b_result, self.c_result):
            edit.clear()
        self.status.setText("Calibration table cleared")

    def determine_calibration_constants(self) -> tuple[float, float, float] | None:
        """Fit and display calibration constants from completed table rows."""
        observed, known_mz = self._completed_rows()
        if observed.size < 3:
            self.status.setText("At least three completed calibration rows are required")
            return None
        try:
            if self._axis_mode == "Mass":
                tof_ns = mass_to_tof_ns(observed, self._current_coefficients)
            else:
                tof_ns = observed * 1000.0
            coefficients = fit_mass_calibration(tof_ns, known_mz)
        except ValueError as exc:
            self.status.setText(str(exc))
            return None
        a, b, c = coefficients
        self.a_result.setText(f"{a:.9g}")
        self.b_result.setText(f"{b:.9g}")
        self.c_result.setText(f"{c:.9g}")
        self.status.setText("Calibration constants determined")
        return coefficients

    def set_current_coefficients(self, coefficients: tuple[float, float, float]) -> None:
        """Set current coefficients used when observed positions are in m/z."""
        self._current_coefficients = coefficients

    def current_result_coefficients(self) -> tuple[float, float, float] | None:
        """Return currently displayed fit constants, if present and numeric."""
        try:
            return (float(self.a_result.text()), float(self.b_result.text()), float(self.c_result.text()))
        except ValueError:
            return None

    def _completed_rows(self) -> tuple[np.ndarray, np.ndarray]:
        observed_values: list[float] = []
        known_values: list[float] = []
        for observed_edit, known_edit in zip(self.observed_edits, self.known_mass_edits, strict=True):
            observed_text = observed_edit.text().strip()
            known_text = known_edit.text().strip()
            if not observed_text and not known_text:
                continue
            if not observed_text or not known_text:
                continue
            try:
                observed_values.append(float(observed_text))
                known_values.append(float(known_text))
            except ValueError:
                self.status.setText("Calibration entries must be numeric")
                return np.array([]), np.array([])
        return np.asarray(observed_values, dtype=np.float64), np.asarray(known_values, dtype=np.float64)

    def _update_collection_status(self, enabled: bool) -> None:
        self.status.setText("Collection enabled" if enabled else "Collection disabled")
