"""pyqtgraph spectrum display."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from pytof_new.processing.filtering import smooth_savgol
from pytof_new.processing.peaks import PeakFitResult, fit_peak_window


@dataclass(frozen=True)
class PeakFitSelection:
    """Peak fit result emitted after an interactive plot selection."""

    plot_name: str
    axis_mode: str
    center: float
    fwhm: float | None
    fitted: bool


@dataclass
class LoadedSpectrumView:
    """Plot state for a loaded .pytof spectrum overlay."""

    spectrum_id: int
    label: str
    mass_axis: np.ndarray
    trace: np.ndarray
    cumulative_curve: object
    live_curve: object


class XRangeZoomViewBox(pg.ViewBox):
    """ViewBox that uses mouse drags for x-zooming and peak fitting."""

    def __init__(self, x_range_selected: object, peak_range_selected: object, reset_requested: object) -> None:
        super().__init__()
        self._x_range_selected = x_range_selected
        self._peak_range_selected = peak_range_selected
        self._reset_requested = reset_requested

    def mouseDragEvent(self, ev: object, axis: object = None) -> None:  # noqa: N802 - pyqtgraph override
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self._handle_left_drag(ev)
            return
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            self._handle_right_drag(ev)
            return
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseDragEvent(ev, axis=axis)
            return

    def _handle_left_drag(self, ev: object) -> None:
        ev.accept()
        start = ev.buttonDownPos(QtCore.Qt.MouseButton.LeftButton)
        end = ev.pos()
        if ev.isStart():
            self._update_x_zoom_box(start, start)
            return
        self._update_x_zoom_box(start, end)
        if ev.isFinish():
            self.rbScaleBox.hide()
            data_rect = self.childGroup.mapRectFromParent(self._x_zoom_rect(start, end))
            x_min, x_max = sorted((data_rect.left(), data_rect.right()))
            if abs(x_max - x_min) > 1e-12:
                self._x_range_selected(x_min, x_max)

    def _handle_right_drag(self, ev: object) -> None:
        ev.accept()
        start = ev.buttonDownPos(QtCore.Qt.MouseButton.RightButton)
        end = ev.pos()
        if ev.isStart():
            self.updateScaleBox(start, start)
            return
        self.updateScaleBox(start, end)
        if ev.isFinish():
            self.rbScaleBox.hide()
            data_rect = self.childGroup.mapRectFromParent(QtCore.QRectF(start, end).normalized())
            x_min, x_max = sorted((data_rect.left(), data_rect.right()))
            if abs(x_max - x_min) > 1e-12:
                self._peak_range_selected(x_min, x_max)

    def mouseClickEvent(self, ev: object) -> None:  # noqa: N802 - pyqtgraph override
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            return
        super().mouseClickEvent(ev)

    def mouseDoubleClickEvent(self, ev: object) -> None:  # noqa: N802 - pyqtgraph override
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.rbScaleBox.hide()
            self._reset_requested()
            return
        super().mouseDoubleClickEvent(ev)

    def _update_x_zoom_box(self, start: QtCore.QPointF, end: QtCore.QPointF) -> None:
        rect = self._x_zoom_rect(start, end)
        self.updateScaleBox(rect.topLeft(), rect.bottomRight())

    def _x_zoom_rect(self, start: QtCore.QPointF, end: QtCore.QPointF) -> QtCore.QRectF:
        bounds = self.boundingRect()
        return QtCore.QRectF(QtCore.QPointF(start.x(), bounds.top()), QtCore.QPointF(end.x(), bounds.bottom())).normalized()


class SpectrumPlot(QtWidgets.QGroupBox):
    """Cumulative and live TOF or mass spectrum plots."""

    peak_fit_completed = QtCore.Signal(object)
    loaded_spectra_changed = QtCore.Signal(object, object)

    def __init__(self) -> None:
        super().__init__("Spectra")
        self.axis_mode = QtWidgets.QComboBox()
        self.axis_mode.addItems(["TOF", "Mass"])
        self.reset_cumulative = QtWidgets.QPushButton("Reset cumulative")
        self.mz_min = QtWidgets.QLineEdit()
        self.mz_max = QtWidgets.QLineEdit()
        self.cumulative_count_label = QtWidgets.QLabel("Cumulative records: 0")
        mz_validator = QtGui.QDoubleValidator(0.0, 1_000_000_000.0, 6, self)
        mz_validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        self.mz_min.setValidator(mz_validator)
        self.mz_max.setValidator(mz_validator)
        self.mz_min.setPlaceholderText("min")
        self.mz_max.setPlaceholderText("max")
        self.mz_min.setMaximumWidth(75)
        self.mz_max.setMaximumWidth(75)
        self.axis_mode.setToolTip("Choose whether plots use TOF or calibrated m/z when mass calibration is available.")
        self.reset_cumulative.setToolTip("Clear the cumulative average without changing the latest record plot.")
        self.mz_min.setToolTip("Minimum displayed x value in m/z. Leave blank to use the full range.")
        self.mz_max.setToolTip("Maximum displayed x value in m/z. Leave blank to use the full range.")
        self.cumulative_count_label.setToolTip("Number of records or raw segments accumulated in the upper plot.")

        cumulative_viewbox = XRangeZoomViewBox(
            self._set_x_range,
            lambda x_min, x_max: self._fit_peak_range("cumulative", x_min, x_max),
            self._reset_zoom,
        )
        live_viewbox = XRangeZoomViewBox(
            self._set_x_range,
            lambda x_min, x_max: self._fit_peak_range("live", x_min, x_max),
            self._reset_zoom,
        )

        self.cumulative_plot = pg.PlotWidget(title="Cumulative Average", viewBox=cumulative_viewbox)
        self.cumulative_plot.setLabel("left", "Signal", units="V")
        self.cumulative_plot.setLabel("bottom", "TOF", units="us")
        self.cumulative_curve = self.cumulative_plot.plot(pen=pg.mkPen("y", width=1.5), name="Cumulative")

        self.live_plot = pg.PlotWidget(title="Latest Batch Average", viewBox=live_viewbox)
        self.live_plot.setLabel("left", "Signal", units="V")
        self.live_plot.setLabel("bottom", "TOF", units="us")
        self.live_average_curve = self.live_plot.plot(pen=pg.mkPen("c", width=1.5), name="Batch average")
        self.latest_curve = self.live_plot.plot(pen=pg.mkPen((180, 180, 180, 120), width=1), name="Latest raw segment")
        self.cumulative_plot.setXLink(self.live_plot)

        self._cumulative_sum: np.ndarray | None = None
        self._cumulative_count = 0
        self._last_axis: np.ndarray | None = None
        self._last_tof_axis_us: np.ndarray | None = None
        self._last_mass_axis: np.ndarray | None = None
        self._cumulative_display_trace: np.ndarray | None = None
        self._last_cumulative_display: tuple[np.ndarray, np.ndarray] | None = None
        self._last_live_average_display: tuple[np.ndarray, np.ndarray] | None = None
        self._last_latest_display: tuple[np.ndarray, np.ndarray] | None = None
        self._last_cumulative_full_display: tuple[np.ndarray, np.ndarray] | None = None
        self._last_live_average_full_display: tuple[np.ndarray, np.ndarray] | None = None
        self._last_latest_full_display: tuple[np.ndarray, np.ndarray] | None = None
        self._last_live_average_trace: np.ndarray | None = None
        self._last_latest_trace: np.ndarray | None = None
        self._peak_annotations: dict[str, tuple[pg.InfiniteLine, pg.TextItem, PeakFitResult]] = {}
        self._loaded_spectra: list[LoadedSpectrumView] = []
        self._next_loaded_spectrum_id = 1
        self._active_source: str | int = "live"

        self.controls = QtWidgets.QHBoxLayout()
        self.controls.setSpacing(6)
        self.controls.addWidget(self.axis_mode)
        self.controls.addWidget(self.reset_cumulative)
        self.controls.addWidget(QtWidgets.QLabel("m/z min"))
        self.controls.addWidget(self.mz_min)
        self.controls.addWidget(QtWidgets.QLabel("m/z max"))
        self.controls.addWidget(self.mz_max)
        self._acquisition_control_index = self.controls.count()
        self.controls.addStretch(1)
        self.controls.addWidget(self.cumulative_count_label)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(self.controls)
        layout.addWidget(self.cumulative_plot)
        layout.addWidget(self.live_plot)

        self.reset_cumulative.clicked.connect(self.clear_cumulative)
        self.mz_min.editingFinished.connect(self._apply_mz_window)
        self.mz_max.editingFinished.connect(self._apply_mz_window)
        self.axis_mode.currentTextChanged.connect(lambda _text: self._refresh_axis_mode())
        self.cumulative_plot.getViewBox().sigRangeChanged.connect(lambda *_args: self._update_peak_label_position("cumulative"))
        self.live_plot.getViewBox().sigRangeChanged.connect(lambda *_args: self._update_peak_label_position("live"))

    def add_loaded_spectrum(self, spectrum: object) -> None:
        """Display a loaded .pytof spectrum on both plots."""
        if len(self._loaded_spectra) >= 4:
            raise ValueError("At most four loaded spectra can be displayed")
        spectrum_id = self._next_loaded_spectrum_id
        self._next_loaded_spectrum_id += 1
        label = str(spectrum.label)
        cumulative_curve = self.cumulative_plot.plot(name=label)
        live_curve = self.live_plot.plot(name=label)
        view = LoadedSpectrumView(
            spectrum_id=spectrum_id,
            label=label,
            mass_axis=spectrum.mass_axis.copy(),
            trace=spectrum.trace.copy(),
            cumulative_curve=cumulative_curve,
            live_curve=live_curve,
        )
        self._loaded_spectra.append(view)
        self.axis_mode.setCurrentText("Mass")
        self.set_active_source(spectrum_id)
        self._refresh_loaded_spectrum_curves()
        self._autoscale_or_apply_mz_window()

    def remove_loaded_spectrum(self, spectrum_id: int) -> None:
        """Remove one loaded spectrum overlay."""
        for index, view in enumerate(self._loaded_spectra):
            if view.spectrum_id != spectrum_id:
                continue
            self.cumulative_plot.removeItem(view.cumulative_curve)
            self.live_plot.removeItem(view.live_curve)
            del self._loaded_spectra[index]
            if self._active_source == spectrum_id:
                self.set_active_source("live")
            self._refresh_loaded_spectrum_curves()
            self._autoscale_or_apply_mz_window()
            self._emit_loaded_spectra_changed()
            return

    def set_active_source(self, source: str | int) -> None:
        """Select which source is used for peak fitting and calibration."""
        self._active_source = source
        if source != "live":
            self.axis_mode.setCurrentText("Mass")
        self._clear_peak_annotation("cumulative")
        self._clear_peak_annotation("live")
        self._update_loaded_spectrum_styles()
        self._emit_loaded_spectra_changed()

    def loaded_spectrum_count(self) -> int:
        """Return the number of loaded spectrum overlays."""
        return len(self._loaded_spectra)

    def loaded_spectrum_summaries(self) -> list[tuple[int, str]]:
        """Return id and label pairs for loaded spectrum menu rendering."""
        return [(view.spectrum_id, _short_label(view.label)) for view in self._loaded_spectra]

    def active_source(self) -> str | int:
        """Return the active analysis source."""
        return self._active_source

    def _emit_loaded_spectra_changed(self) -> None:
        self.loaded_spectra_changed.emit(self.loaded_spectrum_summaries(), self._active_source)

    def add_acquisition_controls(
        self,
        arm_button: QtWidgets.QPushButton,
        start_button: QtWidgets.QPushButton,
        stop_button: QtWidgets.QPushButton,
        clear_button: QtWidgets.QPushButton,
        continuous: QtWidgets.QCheckBox,
        polarity: QtWidgets.QComboBox,
    ) -> None:
        """Place acquisition controls in the spectra toolbar."""
        for widget in (arm_button, start_button, stop_button, clear_button, continuous):
            self.controls.insertWidget(self._acquisition_control_index, widget)
            self._acquisition_control_index += 1
        self.controls.insertWidget(self._acquisition_control_index, QtWidgets.QLabel("Detector polarity"))
        self._acquisition_control_index += 1
        self.controls.insertWidget(self._acquisition_control_index, polarity)
        self._acquisition_control_index += 1

    def set_processed(self, processed: object) -> None:
        """Display a processed batch and update cumulative average."""
        self._last_tof_axis_us = processed.tof_axis * 1e6
        self._last_mass_axis = processed.mass_axis.copy() if processed.mass_axis is not None else None
        axis, label = self._current_axis_and_label()
        self._last_axis = axis.copy()

        batch_count = processed.accepted_count
        batch_sum = processed.unfiltered_average_trace.astype(np.float64) * batch_count
        if self._cumulative_sum is None or self._cumulative_sum.shape != batch_sum.shape:
            self._cumulative_sum = np.zeros_like(batch_sum, dtype=np.float64)
            self._cumulative_count = 0
        self._cumulative_sum += batch_sum
        self._cumulative_count += batch_count
        cumulative_average = (self._cumulative_sum / self._cumulative_count).astype(np.float32)
        self._cumulative_display_trace = _maybe_smooth(cumulative_average, processed)
        self._last_cumulative_full_display = (axis.copy(), self._cumulative_display_trace.copy())

        x_cumulative, y_cumulative = _downsample(axis, self._cumulative_display_trace)
        self._last_cumulative_display = (x_cumulative, y_cumulative)
        self.cumulative_curve.setData(x_cumulative, y_cumulative)
        label_name = "records" if processed.record_mode == "hardware_average" else "segments"
        self.cumulative_count_label.setText(f"Cumulative {label_name}: {self._cumulative_count}")

        x_average, average = _downsample(axis, processed.average_trace)
        self._last_live_average_trace = processed.average_trace.copy()
        self._last_live_average_full_display = (axis.copy(), processed.average_trace.copy())
        self._last_live_average_display = (x_average, average)
        self.live_average_curve.setData(x_average, average)
        if processed.record_mode == "raw_segments":
            latest = processed.baseline_corrected_segments[-1]
            self._last_latest_trace = latest.copy()
            x_latest, latest = _downsample(axis, latest)
            self._last_latest_full_display = (axis.copy(), self._last_latest_trace.copy())
            self._last_latest_display = (x_latest, latest)
            self.latest_curve.setData(x_latest, latest)
            self.latest_curve.show()
        else:
            self._last_latest_trace = None
            self._last_latest_full_display = None
            self._last_latest_display = None
            self.latest_curve.clear()
            self.latest_curve.hide()
        self.cumulative_plot.setLabel("bottom", label[0], units=label[1])
        self.live_plot.setLabel("bottom", label[0], units=label[1])
        self._refresh_loaded_spectrum_curves()
        self._apply_mz_window()

    def clear_cumulative(self) -> None:
        """Clear cumulative plot state without changing live data."""
        self._cumulative_sum = None
        self._cumulative_count = 0
        self._cumulative_display_trace = None
        self._last_cumulative_display = None
        self._last_cumulative_full_display = None
        self._clear_peak_annotation("cumulative")
        self.cumulative_curve.setData([], [])
        self.cumulative_count_label.setText("Cumulative records: 0")
        self._refresh_plots()

    def clear_all(self) -> None:
        """Clear cumulative and live plot state."""
        self.clear_cumulative()
        self._last_axis = None
        self._last_tof_axis_us = None
        self._last_mass_axis = None
        self._last_live_average_display = None
        self._last_latest_display = None
        self._last_live_average_full_display = None
        self._last_latest_full_display = None
        self._last_live_average_trace = None
        self._last_latest_trace = None
        self._clear_peak_annotation("live")
        self.live_average_curve.setData([], [])
        self.latest_curve.setData([], [])
        self.latest_curve.hide()
        self._refresh_plots()

    def set_shot_analysis_display(self, processed: object, aligned_average: np.ndarray, last_record: np.ndarray, record_count: int) -> None:
        """Display shot-analysis aligned average and latest analyzed record."""
        self.clear_all()
        self._last_tof_axis_us = processed.tof_axis * 1e6
        self._last_mass_axis = processed.mass_axis.copy() if processed.mass_axis is not None else None
        axis, label = self._current_axis_and_label()
        self._last_axis = axis.copy()

        aligned_average = np.asarray(aligned_average, dtype=np.float32)
        last_record = np.asarray(last_record, dtype=np.float32)
        self._cumulative_sum = aligned_average.astype(np.float64) * record_count
        self._cumulative_count = record_count
        self._cumulative_display_trace = aligned_average.copy()
        self._last_cumulative_full_display = (axis.copy(), aligned_average.copy())
        x_cumulative, y_cumulative = _downsample(axis, aligned_average)
        self._last_cumulative_display = (x_cumulative, y_cumulative)
        self.cumulative_curve.setData(x_cumulative, y_cumulative)
        self.cumulative_count_label.setText(f"Shot analysis records: {record_count}")

        self._last_live_average_trace = last_record.copy()
        self._last_live_average_full_display = (axis.copy(), last_record.copy())
        x_live, y_live = _downsample(axis, last_record)
        self._last_live_average_display = (x_live, y_live)
        self.live_average_curve.setData(x_live, y_live)
        self.cumulative_plot.setLabel("bottom", label[0], units=label[1])
        self.live_plot.setLabel("bottom", label[0], units=label[1])
        self._refresh_loaded_spectrum_curves()
        self._autoscale_or_apply_mz_window()

    def cumulative_spectrum(self) -> tuple[np.ndarray, np.ndarray, int] | None:
        """Return full-resolution cumulative average and record count for saving."""
        if self._cumulative_sum is None or self._cumulative_count == 0 or self._last_axis is None:
            return None
        trace = self._cumulative_display_trace
        if trace is None:
            trace = (self._cumulative_sum / self._cumulative_count).astype(np.float32)
        return self._last_axis.copy(), trace, self._cumulative_count

    def reference_spectrum(self) -> tuple[np.ndarray, np.ndarray, int] | None:
        """Return the unsmoothed cumulative average for reference subtraction."""
        if self._cumulative_sum is None or self._cumulative_count == 0 or self._last_axis is None:
            return None
        trace = (self._cumulative_sum / self._cumulative_count).astype(np.float32)
        return self._last_axis.copy(), trace, self._cumulative_count

    def _autoscale(self) -> None:
        self.cumulative_plot.enableAutoRange()
        self.live_plot.enableAutoRange()

    def _reset_zoom(self) -> None:
        self.cumulative_plot.autoRange()
        self.live_plot.autoRange()

    def _apply_mz_window(self) -> None:
        x_range = self._x_range_from_mz_window()
        if x_range is None:
            return
        self._set_x_range(*x_range)

    def _autoscale_or_apply_mz_window(self) -> None:
        x_range = self._x_range_from_mz_window()
        if x_range is None:
            self._reset_zoom()
        else:
            self._set_x_range(*x_range)

    def _set_x_range(self, x_min: float, x_max: float) -> None:
        self.live_plot.setXRange(x_min, x_max, padding=0.0)
        self.cumulative_plot.setXRange(x_min, x_max, padding=0.0)
        loaded_curves = self._loaded_display_curves()
        self._set_y_range_for_window(self.cumulative_plot, [self._last_cumulative_display, *loaded_curves], x_min, x_max)
        self._set_y_range_for_window(
            self.live_plot,
            [self._last_live_average_display, self._last_latest_display, *loaded_curves],
            x_min,
            x_max,
        )

    def _refresh_axis_mode(self) -> None:
        if self._active_source != "live" and self.axis_mode.currentText() != "Mass":
            self.axis_mode.setCurrentText("Mass")
            return
        if self._last_tof_axis_us is None:
            self._refresh_loaded_spectrum_curves()
            self._autoscale_or_apply_mz_window()
            return
        axis, label = self._current_axis_and_label()
        self._last_axis = axis.copy()
        self._clear_peak_annotation("cumulative")
        self._clear_peak_annotation("live")
        if self._cumulative_display_trace is not None:
            self._last_cumulative_full_display = (axis.copy(), self._cumulative_display_trace.copy())
            x_cumulative, y_cumulative = _downsample(axis, self._cumulative_display_trace)
            self._last_cumulative_display = (x_cumulative, y_cumulative)
            self.cumulative_curve.setData(x_cumulative, y_cumulative)
        if self._last_live_average_trace is not None:
            self._last_live_average_full_display = (axis.copy(), self._last_live_average_trace.copy())
            x_average, average = _downsample(axis, self._last_live_average_trace)
            self._last_live_average_display = (x_average, average)
            self.live_average_curve.setData(x_average, average)
        if self._last_latest_trace is not None:
            self._last_latest_full_display = (axis.copy(), self._last_latest_trace.copy())
            x_latest, latest = _downsample(axis, self._last_latest_trace)
            self._last_latest_display = (x_latest, latest)
            self.latest_curve.setData(x_latest, latest)
        self.cumulative_plot.setLabel("bottom", label[0], units=label[1])
        self.live_plot.setLabel("bottom", label[0], units=label[1])
        self._refresh_loaded_spectrum_curves()
        self._autoscale_or_apply_mz_window()

    def _current_axis_and_label(self) -> tuple[np.ndarray, tuple[str, str]]:
        if self.axis_mode.currentText() == "Mass" and self._last_mass_axis is not None:
            return self._last_mass_axis, ("m/z", "")
        if self._last_tof_axis_us is None:
            return np.array([], dtype=np.float64), ("TOF", "us")
        return self._last_tof_axis_us, ("TOF", "us")

    def _x_range_from_mz_window(self) -> tuple[float, float] | None:
        text_min = self.mz_min.text().strip()
        text_max = self.mz_max.text().strip()
        if not text_min or not text_max:
            return None
        try:
            mz_min = float(text_min)
            mz_max = float(text_max)
        except ValueError:
            return None
        if mz_max <= mz_min:
            return None
        if self.axis_mode.currentText() == "Mass":
            return mz_min, mz_max
        if self._last_mass_axis is None or self._last_axis is None:
            return None
        mask = (self._last_mass_axis >= mz_min) & (self._last_mass_axis <= mz_max)
        if not np.any(mask):
            return None
        x_values = self._last_axis[mask]
        return float(np.min(x_values)), float(np.max(x_values))

    def _set_y_range_for_window(
        self,
        plot: pg.PlotWidget,
        curves: list[tuple[np.ndarray, np.ndarray] | None],
        x_min: float,
        x_max: float,
    ) -> None:
        visible_values: list[np.ndarray] = []
        for curve in curves:
            if curve is None:
                continue
            x_values, y_values = curve
            mask = (x_values >= x_min) & (x_values <= x_max)
            if np.any(mask):
                values = y_values[mask]
                visible_values.append(values[np.isfinite(values)])
        if not visible_values:
            plot.enableAutoRange(axis="y", enable=True)
            return
        combined = np.concatenate(visible_values)
        if combined.size == 0:
            plot.enableAutoRange(axis="y", enable=True)
            return
        y_min = float(np.min(combined))
        y_max = float(np.max(combined))
        if y_min == y_max:
            padding = max(abs(y_min) * 0.05, 1e-6)
        else:
            padding = (y_max - y_min) * 0.08
        plot.setYRange(y_min - padding, y_max + padding, padding=0.0)

    def _fit_peak_range(self, plot_name: str, x_min: float, x_max: float) -> None:
        curve = self._active_fit_curve(plot_name)
        if curve is None:
            return
        axis, trace = curve
        mask = (axis >= x_min) & (axis <= x_max)
        if np.count_nonzero(mask) < 5:
            return
        try:
            result = fit_peak_window(axis[mask], trace[mask])
        except ValueError:
            return
        self._draw_peak_annotation(plot_name, result, axis[mask], trace[mask])
        self.peak_fit_completed.emit(
            PeakFitSelection(
                plot_name=plot_name,
                axis_mode="Mass" if self._active_source != "live" else self.axis_mode.currentText(),
                center=result.center,
                fwhm=result.fwhm,
                fitted=result.fitted,
            )
        )

    def _active_fit_curve(self, plot_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        if self._active_source == "live":
            return self._last_cumulative_full_display if plot_name == "cumulative" else self._last_live_average_full_display
        for view in self._loaded_spectra:
            if view.spectrum_id == self._active_source:
                return view.mass_axis, view.trace
        return None

    def _refresh_loaded_spectrum_curves(self) -> None:
        visible = self.axis_mode.currentText() == "Mass"
        for view in self._loaded_spectra:
            x_values, y_values = _downsample(view.mass_axis, view.trace)
            view.cumulative_curve.setData(x_values, y_values)
            view.live_curve.setData(x_values, y_values)
            view.cumulative_curve.setVisible(visible)
            view.live_curve.setVisible(visible)
        self._update_loaded_spectrum_styles()

    def _loaded_display_curves(self) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.axis_mode.currentText() != "Mass":
            return []
        curves: list[tuple[np.ndarray, np.ndarray]] = []
        for view in self._loaded_spectra:
            curves.append(_downsample(view.mass_axis, view.trace))
        return curves

    def _update_loaded_spectrum_styles(self) -> None:
        inactive_pen = pg.mkPen((150, 150, 150, 170), width=1.2, style=QtCore.Qt.PenStyle.DashLine)
        active_pen = pg.mkPen((255, 90, 210), width=2.0)
        for view in self._loaded_spectra:
            pen = active_pen if view.spectrum_id == self._active_source else inactive_pen
            view.cumulative_curve.setPen(pen)
            view.live_curve.setPen(pen)

    def _draw_peak_annotation(
        self,
        plot_name: str,
        result: PeakFitResult,
        axis_window: np.ndarray,
        trace_window: np.ndarray,
    ) -> None:
        self._clear_peak_annotation(plot_name)
        plot = self.cumulative_plot if plot_name == "cumulative" else self.live_plot
        pen = pg.mkPen((0, 220, 80), width=1.5, style=QtCore.Qt.PenStyle.DashLine)
        line = pg.InfiniteLine(pos=result.center, angle=90, movable=False, pen=pen)
        plot.addItem(line)
        label = pg.TextItem(_peak_label_text(result, self.axis_mode.currentText()), color=(0, 220, 80), anchor=(0.0, 0.0))
        plot.addItem(label)
        self._peak_annotations[plot_name] = (line, label, result)
        self._update_peak_label_position(plot_name)

    def _clear_peak_annotation(self, plot_name: str) -> None:
        annotation = self._peak_annotations.pop(plot_name, None)
        if annotation is None:
            return
        plot = self.cumulative_plot if plot_name == "cumulative" else self.live_plot
        for item in annotation[:2]:
            plot.removeItem(item)

    def _update_peak_label_position(self, plot_name: str) -> None:
        annotation = self._peak_annotations.get(plot_name)
        if annotation is None:
            return
        _line, label, result = annotation
        plot = self.cumulative_plot if plot_name == "cumulative" else self.live_plot
        view = plot.getViewBox()
        x_range, y_range = view.viewRange()
        x_min, x_max = x_range
        y_min, y_max = y_range
        if x_max <= x_min or y_max <= y_min:
            return
        x_padding = (x_max - x_min) * 0.02
        y_padding = (y_max - y_min) * 0.08
        x_position = min(max(result.center + x_padding, x_min + x_padding), x_max - x_padding)
        y_position = y_max - y_padding
        label.setAnchor((0.0, 0.0))
        label.setPos(x_position, y_position)

    def _refresh_plots(self) -> None:
        for plot in (self.cumulative_plot, self.live_plot):
            plot.getViewBox().rbScaleBox.hide()
            plot.getPlotItem().update()
            plot.scene().update()
            plot.viewport().update()
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)


def _downsample(axis: np.ndarray, trace: np.ndarray, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    if len(axis) <= max_points:
        return axis, trace
    step = max(1, len(axis) // max_points)
    return axis[::step], trace[::step]


def _maybe_smooth(trace: np.ndarray, processed: object) -> np.ndarray:
    if not processed.smoothing_enabled:
        return trace
    return smooth_savgol(trace, processed.smoothing_window)


def _short_label(label: str, max_length: int = 18) -> str:
    label = label.strip() or "loaded"
    if len(label) <= max_length:
        return label
    return f"{label[: max_length - 1]}..."


def _peak_label_text(result: PeakFitResult, axis_mode: str) -> str:
    name = "m/z" if axis_mode == "Mass" else "t"
    unit = "" if axis_mode == "Mass" else " us"
    fwhm = "n/a" if result.fwhm is None else f"{result.fwhm:.6g}{unit}"
    fit_note = "fit" if result.fitted else "estimate"
    return f"{name} = {result.center:.6g}{unit}\nFWHM = {fwhm}\n{fit_note}"
