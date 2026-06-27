"""Allow ``python -m pytof_new`` to launch the GUI directly."""

import sys
from pathlib import Path

from PySide6 import QtWidgets

from pytof_new.gui.main_window import MainWindow
from pytof_new.logging_config import configure_logging


def main() -> int:
    configure_logging()
    app = QtWidgets.QApplication(sys.argv)
    ini_path = _select_startup_ini()
    if ini_path is None:
        return 0
    window = MainWindow()
    window.file_panel.ini_path.setText(str(ini_path))
    if not window.load_pytof_ini():
        window.close()
        return 1
    window.show()
    return app.exec()


def _select_startup_ini() -> Path | None:
    start_dir = str(_default_ini_dir())
    path, _filter = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select PyTOF.ini", start_dir, "INI files (*.ini);;All files (*)"
    )
    if not path:
        return None
    return Path(path)


def _default_ini_dir() -> Path:
    cwd = Path.cwd()
    for candidate in (cwd / "PyTOF.ini", cwd.parent / "PyTOF.ini"):
        if candidate.exists():
            return candidate.parent
    return cwd


if __name__ == "__main__":
    raise SystemExit(main())
