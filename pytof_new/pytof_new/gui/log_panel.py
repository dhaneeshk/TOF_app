"""Application log display."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class LogPanel(QtWidgets.QGroupBox):
    """Timestamped text log panel."""

    def __init__(self) -> None:
        super().__init__("Log")
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.text.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.text)

    def append(self, message: str) -> None:
        """Append one or more prefixed log lines."""
        lines = str(message).splitlines() or [""]
        self.text.appendPlainText("\n".join(f">> {line}" for line in lines))
