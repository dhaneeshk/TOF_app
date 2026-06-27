"""Shared GUI interaction guards."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class GuardedInteractionFilter(QtCore.QObject):
    """Prevent accidental value changes from hover-only wheel and label clicks."""

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if isinstance(watched, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
            if event.type() == QtCore.QEvent.Type.Wheel and not watched.hasFocus():
                event.ignore()
                return True

        if isinstance(watched, QtWidgets.QCheckBox):
            if event.type() in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseButtonRelease):
                if not _checkbox_indicator_rect(watched).contains(event.position().toPoint()):
                    event.ignore()
                    return True
        return super().eventFilter(watched, event)


def install_guarded_interactions(root: QtWidgets.QWidget, event_filter: GuardedInteractionFilter) -> None:
    """Install guarded wheel/click behavior on editable controls under root."""
    widgets = [
        *root.findChildren(QtWidgets.QAbstractSpinBox),
        *root.findChildren(QtWidgets.QComboBox),
        *root.findChildren(QtWidgets.QCheckBox),
    ]
    for widget in widgets:
        widget.installEventFilter(event_filter)
        if isinstance(widget, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
            widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)


def set_compact_widths(root: QtWidgets.QWidget, width: int = 105) -> None:
    """Constrain common value-entry controls to a compact width."""
    for widget in root.findChildren(QtWidgets.QAbstractSpinBox):
        widget.setMaximumWidth(width)
    for widget in root.findChildren(QtWidgets.QComboBox):
        widget.setMaximumWidth(width)
    for widget in root.findChildren(QtWidgets.QLineEdit):
        if not widget.isReadOnly():
            widget.setMaximumWidth(width)


def _checkbox_indicator_rect(checkbox: QtWidgets.QCheckBox) -> QtCore.QRect:
    option = QtWidgets.QStyleOptionButton()
    checkbox.initStyleOption(option)
    return checkbox.style().subElementRect(QtWidgets.QStyle.SubElement.SE_CheckBoxIndicator, option, checkbox)
