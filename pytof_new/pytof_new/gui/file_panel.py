"""File and metadata settings."""

from __future__ import annotations

from pathlib import Path
from PySide6 import QtCore, QtWidgets

from pytof_new.config.models import StorageConfig


class FilePanel(QtWidgets.QGroupBox):
    """Collect output filename and metadata."""

    save_cumulative_requested = QtCore.Signal()
    save_reference_requested = QtCore.Signal()
    load_spectrum_requested = QtCore.Signal()
    active_loaded_source_requested = QtCore.Signal(object)
    remove_loaded_spectrum_requested = QtCore.Signal(int)
    load_ini_requested = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__("File and Metadata")
        self.output = QtWidgets.QLineEdit(str(Path("pytof_new") / _timestamped_pytof_name("mock", "")))
        self.ini_path = QtWidgets.QLineEdit(str(_default_ini_path()))
        self.browse_ini_button = QtWidgets.QPushButton("Browse ini")
        self.load_ini_button = QtWidgets.QPushButton("Load ini")
        self.molecule = QtWidgets.QLineEdit("mock")
        self.surface = QtWidgets.QLineEdit("")
        self.q1 = QtWidgets.QLineEdit("")
        self.q2 = QtWidgets.QLineEdit("")
        self.uv = QtWidgets.QLineEdit("")
        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setMaximumHeight(70)
        self.save_cumulative_button = QtWidgets.QPushButton("Save spectrum")
        self.save_reference_button = QtWidgets.QPushButton("Save reference")
        self.load_spectrum_button = QtWidgets.QPushButton("Load spectrum")
        self.loaded_spectra_menu_button = QtWidgets.QToolButton()
        self.loaded_spectra_menu_button.setText("Loaded spectra")
        self.loaded_spectra_menu_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.loaded_spectra_menu = QtWidgets.QMenu(self.loaded_spectra_menu_button)
        self.loaded_spectra_menu_button.setMenu(self.loaded_spectra_menu)
        self.loaded_spectra_menu_button.setEnabled(False)
        self.output.setToolTip("Output .pytof file path for the cumulative spectrum.")
        self.ini_path.setToolTip("Path to the legacy PyTOF.ini file used for defaults and calibration constants.")
        self.browse_ini_button.setToolTip("Select a PyTOF.ini file.")
        self.load_ini_button.setToolTip("Load SaveDir, N_Average, and polarity-specific calibration constants from PyTOF.ini.")
        self.molecule.setToolTip("Molecule name used in metadata and default .pytof filename.")
        self.surface.setToolTip("Optional surface name used in metadata and default .pytof filename.")
        self.q1.setToolTip("Q1 electronic setting saved in .pytof metadata.")
        self.q2.setToolTip("Q2 electronic setting saved in .pytof metadata.")
        self.uv.setToolTip("UV setting saved in .pytof metadata.")
        self.notes.setToolTip("Operator notes stored in run metadata.")
        self.save_cumulative_button.setToolTip("Save the current top cumulative average spectrum to the selected .pytof file.")
        self.save_reference_button.setToolTip("Save the current blank cumulative average as a reference for later subtraction.")
        self.load_spectrum_button.setToolTip("Load a saved .pytof spectrum overlay for comparison and calibration.")
        self.loaded_spectra_menu_button.setToolTip("Select the active loaded spectrum or remove loaded spectra.")
        for button in (
            self.browse_ini_button,
            self.load_ini_button,
            self.save_cumulative_button,
            self.save_reference_button,
            self.load_spectrum_button,
            self.loaded_spectra_menu_button,
        ):
            button.setMaximumWidth(180)

        form = QtWidgets.QFormLayout(self)
        form.addRow("Output file", self.output)
        ini_buttons = QtWidgets.QHBoxLayout()
        ini_buttons.addWidget(self.browse_ini_button)
        ini_buttons.addWidget(self.load_ini_button)
        form.addRow("PyTOF.ini file", self.ini_path)
        form.addRow("", ini_buttons)
        metadata_row = QtWidgets.QHBoxLayout()
        metadata_row.setSpacing(8)
        for label, widget in (
            ("Molecule", self.molecule),
            ("Surface", self.surface),
            ("Q1", self.q1),
            ("Q2", self.q2),
            ("UV", self.uv),
        ):
            widget.setMaximumWidth(110)
            metadata_row.addWidget(QtWidgets.QLabel(label))
            metadata_row.addWidget(widget)
        metadata_row.addStretch(1)
        form.addRow(metadata_row)
        form.addRow("Notes", self.notes)
        save_buttons = QtWidgets.QHBoxLayout()
        save_buttons.setSpacing(12)
        save_buttons.addWidget(self.save_cumulative_button)
        save_buttons.addWidget(self.save_reference_button)
        save_buttons.addWidget(self.load_spectrum_button)
        save_buttons.addWidget(self.loaded_spectra_menu_button)
        save_buttons.addStretch(1)
        form.addRow(save_buttons)
        self.save_cumulative_button.clicked.connect(self.save_cumulative_requested)
        self.save_reference_button.clicked.connect(self.save_reference_requested)
        self.load_spectrum_button.clicked.connect(self.load_spectrum_requested)
        self.browse_ini_button.clicked.connect(self._browse_ini)
        self.load_ini_button.clicked.connect(self.load_ini_requested)

    def config(self) -> StorageConfig:
        """Return immutable storage settings."""
        return StorageConfig(
            output_path=Path(self.output.text()),
            molecule=self.molecule.text(),
            surface=self.surface.text(),
            q1=self.q1.text(),
            q2=self.q2.text(),
            uv=self.uv.text(),
            notes=self.notes.toPlainText(),
            save_raw_segments=True,
            save_processed=True,
        )

    def update_default_output_path(self, save_dir: Path) -> None:
        """Set a timestamped .pytof output path from current metadata fields."""
        self.output.setText(str(Path(save_dir) / _timestamped_pytof_name(self.molecule.text(), self.surface.text())))

    def next_spectrum_output_path(self) -> Path:
        """Return a new collision-safe .pytof output path from current metadata."""
        parent = Path(self.output.text()).parent
        return _unique_path(parent / _timestamped_pytof_name(self.molecule.text(), self.surface.text()))

    def set_locked_for_acquisition(self, locked: bool) -> None:
        """Lock config controls while keeping spectrum saving available."""
        self.setEnabled(True)
        for widget in (
            self.output,
            self.ini_path,
            self.browse_ini_button,
            self.load_ini_button,
            self.save_reference_button,
        ):
            widget.setEnabled(not locked)
        for widget in (self.molecule, self.surface, self.q1, self.q2, self.uv, self.notes):
            widget.setEnabled(True)
        self.save_cumulative_button.setEnabled(True)
        self.load_spectrum_button.setEnabled(True)
        self.loaded_spectra_menu_button.setEnabled(bool(self.loaded_spectra_menu.actions()))

    def update_loaded_spectra_menu(self, spectra: list[tuple[int, str]], active_source: object) -> None:
        """Refresh the loaded-spectra menu from current plot state."""
        self.loaded_spectra_menu.clear()
        if not spectra:
            self.loaded_spectra_menu_button.setEnabled(False)
            return
        live_action = self.loaded_spectra_menu.addAction("Live")
        live_action.setCheckable(True)
        live_action.setChecked(active_source == "live")
        live_action.triggered.connect(lambda _checked=False: self.active_loaded_source_requested.emit("live"))
        self.loaded_spectra_menu.addSeparator()
        for spectrum_id, label in spectra:
            row = _LoadedSpectrumMenuRow(spectrum_id, label, active_source == spectrum_id, self.loaded_spectra_menu)
            row.selected.connect(self.active_loaded_source_requested)
            row.remove_requested.connect(self.remove_loaded_spectrum_requested)
            action = QtWidgets.QWidgetAction(self.loaded_spectra_menu)
            action.setDefaultWidget(row)
            self.loaded_spectra_menu.addAction(action)
        self.loaded_spectra_menu_button.setEnabled(True)

    def _browse_ini(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Select PyTOF.ini", self.ini_path.text(), "INI files (*.ini);;All files (*)")
        if path:
            self.ini_path.setText(path)


def _default_ini_path() -> Path:
    cwd = Path.cwd()
    candidates = (cwd / "PyTOF.ini", cwd.parent / "PyTOF.ini")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return cwd / "PyTOF.ini"


def _timestamped_pytof_name(molecule: str, surface: str) -> str:
    from datetime import datetime

    safe_molecule = _safe_name_part(molecule) or "pytof"
    safe_surface = _safe_name_part(surface)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if safe_surface:
        return f"{safe_molecule}_{safe_surface}_{timestamp}.pytof"
    return f"{safe_molecule}_{timestamp}.pytof"


def _unique_path(path: Path) -> Path:
    """Return path, or add a numeric suffix if it already exists."""
    path = Path(path)
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused output path for {path}")


def _safe_name_part(value: str) -> str:
    return "_".join(value.strip().split())


class _LoadedSpectrumMenuRow(QtWidgets.QWidget):
    """Menu row with active-source selection and delete button."""

    selected = QtCore.Signal(object)
    remove_requested = QtCore.Signal(int)

    def __init__(self, spectrum_id: int, label: str, active: bool, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.spectrum_id = spectrum_id
        self.select_button = QtWidgets.QRadioButton(label)
        self.select_button.setChecked(active)
        self.delete_button = QtWidgets.QPushButton("x")
        self.delete_button.setMaximumWidth(24)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(self.select_button)
        layout.addStretch(1)
        layout.addWidget(self.delete_button)
        self.select_button.toggled.connect(lambda checked: self.selected.emit(spectrum_id) if checked else None)
        self.delete_button.clicked.connect(lambda _checked=False: self.remove_requested.emit(spectrum_id))
