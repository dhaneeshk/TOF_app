import os
import logging
from pathlib import Path
from types import SimpleNamespace
import h5py
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets
from PySide6 import QtCore

import app as gui_app
import numpy as np

from pytof_new.acquisition.models import AcquisitionBatch
from pytof_new.acquisition.worker import QtLogHandler
from pytof_new.acquisition.worker import _display_processed
from pytof_new.config.legacy_ini import PyTOFIniSettings, load_pytof_ini, save_pytof_ini
from pytof_new.config.models import DigitizerConfig, ProcessingConfig
from pytof_new.gui.spectrum_plot import PeakFitSelection
from pytof_new.gui.digitizer_panel import calculate_segment_samples
from pytof_new.gui.main_window import MainWindow
from pytof_new.gui.spectrum_plot import SpectrumPlot
from pytof_new.processing.filtering import smooth_savgol
from pytof_new.processing.pipeline import process_batch


def test_main_window_builds_and_snapshots_config() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    assert window.acquisition_panel.continuous.isChecked()
    assert not hasattr(window.acquisition_panel, "batch_button")
    assert not hasattr(window.acquisition_panel, "arm_button")
    assert not hasattr(window.acquisition_panel, "state")
    assert not hasattr(window.acquisition_panel, "shots")
    assert window.acquisition_panel.polarity.currentText() == "positive"
    assert window.acquisition_panel.configure_button.text() == "Arm"
    assert window.acquisition_panel.configure_button.toolTip()
    assert window.acquisition_panel.configure_button.parent() is window.spectrum_plot
    assert window.acquisition_panel.start_button.parent() is window.spectrum_plot
    assert window.acquisition_panel.stop_button.parent() is window.spectrum_plot
    assert window.acquisition_panel.clear_button.parent() is window.spectrum_plot
    assert not hasattr(window.spectrum_plot, "autoscale")
    assert not hasattr(window.spectrum_plot, "reset_zoom")
    assert not hasattr(window.spectrum_plot, "load_spectrum_button")
    assert not hasattr(window, "hide_settings_button")
    assert window.file_panel.load_spectrum_button.text() == "Load spectrum"
    assert not window.file_panel.loaded_spectra_menu_button.isEnabled()
    assert not hasattr(window.file_panel, "save_raw")
    assert not hasattr(window.file_panel, "save_processed")
    assert not hasattr(window.file_panel, "save_config")
    window.log_panel.append("line one\nline two")
    assert window.log_panel.text.toPlainText() == ">> line one\n>> line two"
    assert window.settings_stack.isHidden()
    window._toggle_settings(0)
    assert not window.settings_stack.isHidden()
    assert window.settings_stack.currentWidget() is window.digitizer_panel
    window._toggle_settings(0)
    assert window.settings_stack.isHidden()
    window._toggle_settings(2)
    assert window.settings_stack.currentWidget() is window.processing_panel
    assert not hasattr(window.processing_panel, "mass_calibration")
    assert not hasattr(window.processing_panel, "peak_finding")
    assert window.processing_panel.shot_analysis_button.text() == "Shot analysis"
    assert window.processing_panel.shot_analysis_records.maximum() == 300
    assert window.processing_panel.shot_analysis_delay_ms.minimum() == 10.0
    window.processing_panel.shot_analysis_records.setValue(7)
    window.processing_panel.shot_analysis_delay_ms.setValue(10.0)
    window.processing_panel.shot_analysis_max_lag_ns.setValue(125.0)
    assert window.processing_panel.shot_analysis_min_mz.value() == 50.0
    assert window.processing_panel.shot_analysis_settings() == pytest.approx((7, 0.010, 125e-9, 50.0))
    assert "0.009 us" in window.processing_panel.smoothing_interval.text()
    window.digitizer_panel.sample_rate.setCurrentIndex(1)
    assert "0.018 us" in window.processing_panel.smoothing_interval.text()
    window.digitizer_panel.sample_rate.setCurrentIndex(0)
    window.processing_panel.high_pass.setChecked(True)
    window.processing_panel.high_pass_cutoff_mhz.setValue(0.25)
    window.processing_panel.low_pass.setChecked(True)
    window.processing_panel.low_pass_cutoff_mhz.setValue(25.0)
    window.processing_panel.absolute_signal.setChecked(True)
    window.processing_panel.reference_path.setText("reference_smoke.h5")
    window._toggle_settings(3)
    assert window.settings_stack.currentWidget() is window.calibration_panel
    assert len(window.calibration_panel.observed_edits) == 6
    window._toggle_settings(4)
    assert window.settings_stack.currentWidget() is window.mock_spectra_panel
    window.digitizer_panel.tof_window_us.setValue(50.0)
    assert window.digitizer_panel.segment_samples() == calculate_segment_samples(50e-6, 1.25e9, 32)
    assert not window.bme_panel.advanced_checkbox.isChecked()
    assert not window.bme_panel.advanced_group.isEnabled()
    assert window.bme_panel.extraction_fill_us.value() == pytest.approx(55.0)
    assert window.bme_panel.repetition_us.value() == pytest.approx(105.0)
    assert window.bme_panel.digitizer_width_us.value() == pytest.approx(50.0)
    assert window.bme_panel.push_width_us.value() == pytest.approx(50.0)
    assert window.bme_panel.pull_width_us.value() == pytest.approx(50.0)
    assert window.bme_panel.digitizer_channel.currentText() == "A"
    assert window.bme_panel.push_channel.currentText() == "C"
    assert window.bme_panel.pull_channel.currentText() == "F"
    assert window.bme_panel.pull_polarity.currentText() == "NEG"
    assert window.bme_panel.validation_message.text() == "OK"
    window.bme_panel.advanced_checkbox.setChecked(True)
    window.bme_panel.push_channel.setCurrentText("A")
    assert "unique" in window.bme_panel.validation_message.text()
    window.bme_panel.push_channel.setCurrentText("C")
    window.bme_panel.advanced_checkbox.setChecked(False)
    window.digitizer_panel.tof_window_us.setValue(60.0)
    assert window.bme_panel.repetition_us.value() == pytest.approx(115.0)
    assert window.bme_panel.digitizer_width_us.value() == pytest.approx(60.0)
    window.digitizer_panel.tof_window_us.setValue(50.0)
    window.bme_panel.repetition_us.setValue(10.0)
    assert "exceeds" in window.digitizer_panel.window_warning.text()
    assert "\n" in window.digitizer_panel.window_warning.text()
    window.bme_panel.repetition_us.setValue(105.0)
    window.mock_spectra_panel.ion_peaks_enabled.setChecked(False)
    window.mock_spectra_panel.noise_rms_mv.setValue(4.5)
    window.mock_spectra_panel.ringing_enabled.setChecked(True)
    window.mock_spectra_panel.ringing_amplitude_mv.setValue(15.0)
    window.mock_spectra_panel.ringing_frequency_mhz.setValue(18.0)
    window.mock_spectra_panel.ringing_decay_us.setValue(2.5)
    window.mock_spectra_panel.ringing_phase_deg.setValue(45.0)
    window.mock_spectra_panel.ringing_follows_timing_jitter.setChecked(True)
    config = window._snapshot_config()
    config.validate()
    assert config.processing.high_pass_enabled
    assert config.processing.high_pass_cutoff_hz == pytest.approx(250_000.0)
    assert config.processing.low_pass_enabled
    assert config.processing.low_pass_cutoff_hz == pytest.approx(25_000_000.0)
    assert config.processing.absolute_signal_enabled
    assert config.processing.mass_calibration_enabled
    assert config.processing.mass_calibration == window._active_calibration()
    assert not config.processing.peak_finding_enabled
    assert config.processing.reference_path is not None
    assert str(config.processing.reference_path).endswith("reference_smoke.h5")
    assert not config.mock_spectra.ion_peaks_enabled
    assert config.mock_spectra.resolving_power == window.mock_spectra_panel.resolving_power.value()
    assert config.mock_spectra.noise_rms_v == pytest.approx(0.0045)
    assert config.mock_spectra.ringing_enabled
    assert config.mock_spectra.ringing_amplitude_v == 0.015
    assert config.mock_spectra.ringing_phase_rad > 0.0
    assert config.mock_spectra.ringing_follows_timing_jitter
    window.close()
    assert app is not None


def test_qt_log_handler_accepts_callable_sink() -> None:
    messages: list[str] = []
    handler = QtLogHandler(messages.append)
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", (), None)
    handler.emit(record)
    assert messages == ["hello"]


def test_cumulative_spectrum_saves_displayed_smoothed_trace() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    plot = SpectrumPlot()
    raw = np.zeros((1, 64), dtype=np.int16)
    raw[0, 30] = 3000
    batch = AcquisitionBatch(raw, None, sample_rate_hz=10.0, pretrigger_samples=2, first_trigger_index=0)
    digitizer = DigitizerConfig(segment_samples=64, number_of_segments=1, pretrigger_samples=2, input_range_v=1.0)
    processing = ProcessingConfig(baseline_start=0, baseline_stop=2, smoothing_enabled=True, smoothing_window=11)
    processed = process_batch(batch, digitizer, processing)
    plot.set_processed(processed)
    spectrum = plot.cumulative_spectrum()
    assert spectrum is not None
    _axis, trace, count = spectrum
    assert count == 1
    np.testing.assert_allclose(trace, smooth_savgol(processed.unfiltered_average_trace, 11), rtol=1e-6)
    plot.close()
    assert app is not None


def test_axis_mode_change_replots_existing_data() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    plot = SpectrumPlot()
    raw = np.zeros((1, 64), dtype=np.int16)
    raw[0, 30] = 3000
    batch = AcquisitionBatch(raw, None, sample_rate_hz=10.0, pretrigger_samples=2, first_trigger_index=0)
    digitizer = DigitizerConfig(segment_samples=64, number_of_segments=1, pretrigger_samples=2, input_range_v=1.0)
    processing = ProcessingConfig(
        baseline_start=0,
        baseline_stop=2,
        mass_calibration_enabled=True,
        mass_calibration=(3.5e-6, 0.0129, 6.27),
    )
    processed = process_batch(batch, digitizer, processing)
    plot.set_processed(processed)
    tof_x, _tof_y = plot.live_average_curve.getData()
    plot.axis_mode.setCurrentText("Mass")
    mass_x, _mass_y = plot.live_average_curve.getData()
    np.testing.assert_allclose(tof_x, processed.tof_axis * 1e6)
    np.testing.assert_allclose(mass_x, processed.mass_axis)
    plot.close()
    assert app is not None


def test_mz_window_limits_tof_axis_display_range() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    plot = SpectrumPlot()
    raw = np.zeros((1, 128), dtype=np.int16)
    raw[0, 60] = 3000
    batch = AcquisitionBatch(raw, None, sample_rate_hz=1.0e6, pretrigger_samples=0, first_trigger_index=0)
    digitizer = DigitizerConfig(sample_rate_hz=1.0e6, segment_samples=128, number_of_segments=1, pretrigger_samples=0, input_range_v=1.0)
    processing = ProcessingConfig(
        baseline_start=0,
        baseline_stop=2,
        mass_calibration_enabled=True,
        mass_calibration=(0.0, 1.0, 0.0),
    )
    processed = process_batch(batch, digitizer, processing)
    plot.set_processed(processed)
    plot.mz_min.setText("20000")
    plot.mz_max.setText("50000")
    plot._apply_mz_window()
    tof_min = processed.tof_axis[20] * 1e6
    tof_max = processed.tof_axis[50] * 1e6
    x_range, _y_range = plot.cumulative_plot.getViewBox().viewRange()
    assert x_range[0] == pytest.approx(tof_min)
    assert x_range[1] == pytest.approx(tof_max)
    plot.axis_mode.setCurrentText("Mass")
    x_range, _y_range = plot.cumulative_plot.getViewBox().viewRange()
    assert x_range[0] == pytest.approx(20000.0)
    assert x_range[1] == pytest.approx(50000.0)
    plot.close()
    assert app is not None


def test_cumulative_uses_full_count_for_aggregated_display_updates() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    plot = SpectrumPlot()
    raw = np.zeros((4, 64), dtype=np.int16)
    raw[:, 30] = 3000
    batch = AcquisitionBatch(raw, None, sample_rate_hz=10.0, pretrigger_samples=2, first_trigger_index=0)
    digitizer = DigitizerConfig(segment_samples=64, number_of_segments=4, pretrigger_samples=2, input_range_v=1.0)
    processing = ProcessingConfig(baseline_start=0, baseline_stop=2)
    processed = process_batch(batch, digitizer, processing)
    pending_sum = processed.accepted_baseline_corrected_segments.sum(axis=0, dtype=np.float64)
    display = _display_processed(processed, pending_sum, processed.accepted_count)
    plot.set_processed(display)
    spectrum = plot.reference_spectrum()
    assert spectrum is not None
    _axis, trace, count = spectrum
    assert count == 4
    np.testing.assert_allclose(trace, processed.unfiltered_average_trace, rtol=1e-6)
    plot.close()
    assert app is not None


def test_peak_fit_annotation_is_created_from_selected_range() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    plot = SpectrumPlot()
    raw = np.zeros((1, 512), dtype=np.int16)
    axis_index = np.arange(512)
    raw[0] = np.rint(12000 * np.exp(-0.5 * ((axis_index - 240) / 8.0) ** 2)).astype(np.int16)
    batch = AcquisitionBatch(raw, None, sample_rate_hz=1.0e6, pretrigger_samples=0, first_trigger_index=0)
    digitizer = DigitizerConfig(sample_rate_hz=1.0e6, segment_samples=512, number_of_segments=1, pretrigger_samples=0, input_range_v=1.0)
    processing = ProcessingConfig(baseline_start=0, baseline_stop=10, subtract_baseline=False)
    processed = process_batch(batch, digitizer, processing)
    plot.set_processed(processed)
    plot._fit_peak_range("live", 220.0, 260.0)
    assert "live" in plot._peak_annotations
    line, label, _result = plot._peak_annotations["live"]
    assert 235.0 <= line.value() <= 245.0
    assert "FWHM" in label.toPlainText()
    plot.close()
    assert app is not None


def test_loaded_spectrum_becomes_active_mass_source_for_peak_fits(tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    plot = SpectrumPlot()
    mass_axis = np.linspace(1.0, 100.0, 512)
    trace = np.exp(-0.5 * ((mass_axis - 42.0) / 2.0) ** 2).astype(np.float32)
    spectrum = SimpleNamespace(path=tmp_path / "loaded.pytof", label="Loaded", mass_axis=mass_axis, trace=trace)
    selections = []
    plot.peak_fit_completed.connect(selections.append)

    plot.add_loaded_spectrum(spectrum)
    assert plot.axis_mode.currentText() == "Mass"
    assert plot.loaded_spectrum_count() == 1
    assert plot._active_source != "live"
    plot._fit_peak_range("cumulative", 35.0, 50.0)

    assert selections
    assert selections[-1].axis_mode == "Mass"
    assert selections[-1].center == pytest.approx(42.0, abs=0.5)
    plot.set_active_source("live")
    assert plot._active_source == "live"
    plot.close()
    assert app is not None


def test_loaded_spectrum_limit_and_delete_falls_back_to_live(tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    plot = SpectrumPlot()
    mass_axis = np.linspace(1.0, 10.0, 16)
    trace = np.ones(16, dtype=np.float32)
    for index in range(4):
        plot.add_loaded_spectrum(SimpleNamespace(path=tmp_path / f"{index}.pytof", label=f"L{index}", mass_axis=mass_axis, trace=trace))
    assert plot.loaded_spectrum_count() == 4
    with pytest.raises(ValueError, match="At most four"):
        plot.add_loaded_spectrum(SimpleNamespace(path=tmp_path / "extra.pytof", label="extra", mass_axis=mass_axis, trace=trace))
    active = plot._active_source
    assert active != "live"
    plot.remove_loaded_spectrum(active)
    assert plot.loaded_spectrum_count() == 3
    assert plot._active_source == "live"
    plot.close()
    assert app is not None


def test_loaded_spectrum_peak_fit_populates_calibration_as_mass(tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    mass_axis = np.linspace(1.0, 100.0, 512)
    trace = np.exp(-0.5 * ((mass_axis - 55.0) / 2.0) ** 2).astype(np.float32)
    spectrum = SimpleNamespace(path=tmp_path / "cal.pytof", label="Cal", mass_axis=mass_axis, trace=trace)
    window.calibration_panel.collect_peaks.setChecked(True)
    window.spectrum_plot.add_loaded_spectrum(spectrum)
    window.spectrum_plot._fit_peak_range("cumulative", 48.0, 62.0)
    assert window.calibration_panel.observed_header.text() == "Observed m/z"
    assert float(window.calibration_panel.observed_edits[0].text()) == pytest.approx(55.0, abs=0.5)
    window.close()
    assert app is not None


def test_loaded_spectra_menu_selects_and_deletes_overlay(tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    mass_axis = np.linspace(1.0, 20.0, 64)
    trace = np.ones(64, dtype=np.float32)
    window.spectrum_plot.add_loaded_spectrum(SimpleNamespace(path=tmp_path / "one.pytof", label="One", mass_axis=mass_axis, trace=trace))
    spectrum_id = window.spectrum_plot.loaded_spectrum_summaries()[0][0]
    assert window.file_panel.loaded_spectra_menu_button.isEnabled()
    assert window.file_panel.loaded_spectra_menu.actions()[0].text() == "Live"
    assert window.spectrum_plot.active_source() == spectrum_id

    window.file_panel.active_loaded_source_requested.emit("live")
    assert window.spectrum_plot.active_source() == "live"
    window.file_panel.active_loaded_source_requested.emit(spectrum_id)
    assert window.spectrum_plot.active_source() == spectrum_id
    window.file_panel.remove_loaded_spectrum_requested.emit(spectrum_id)
    assert window.spectrum_plot.loaded_spectrum_count() == 0
    assert window.spectrum_plot.active_source() == "live"
    assert not window.file_panel.loaded_spectra_menu_button.isEnabled()
    window.close()
    assert app is not None


def test_main_window_shot_analysis_completes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.digitizer_panel.tof_window_us.setValue(0.384)
    window.processing_panel.baseline_stop.setValue(16)
    window.processing_panel.shot_analysis_records.setValue(3)
    window.processing_panel.shot_analysis_delay_ms.setValue(10.0)
    window.processing_panel.shot_analysis_max_lag_ns.setValue(50.0)
    window.processing_panel.shot_analysis_min_mz.setValue(0.0)

    loop = QtCore.QEventLoop()
    timed_out = {"value": False}
    window.start_shot_analysis()
    assert window._shot_analysis_worker is not None
    assert window._shot_analysis_worker.config.bme.repetition_period_s == pytest.approx(0.010)
    assert window._shot_analysis_worker.config.digitizer.hardware_averages_per_record == 1
    window._shot_analysis_worker.finished.connect(loop.quit)
    timer = QtCore.QTimer()
    timer.setSingleShot(True)

    def on_timeout() -> None:
        timed_out["value"] = True
        window.stop()
        loop.quit()

    timer.timeout.connect(on_timeout)
    timer.start(10_000)
    loop.exec()
    timer.stop()
    assert not timed_out["value"]
    assert window._shot_analysis_worker is None
    assert "RMS" in window.processing_panel.shot_analysis_result.text()
    assert window.spectrum_plot.cumulative_count_label.text() == "Shot analysis records: 3"
    cumulative_x, cumulative_y = window.spectrum_plot.cumulative_curve.getData()
    live_x, live_y = window.spectrum_plot.live_average_curve.getData()
    assert len(cumulative_x) == window.digitizer_panel.segment_samples()
    assert len(live_x) == window.digitizer_panel.segment_samples()
    assert np.any(np.asarray(cumulative_y) != 0.0)
    assert np.any(np.asarray(live_y) != 0.0)
    window.close()
    assert app is not None


def test_calibration_panel_collects_peak_fits_and_handles_axis_changes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    selection = PeakFitSelection("live", "TOF", 12.3456789, 0.1, True)
    window.spectrum_plot.peak_fit_completed.emit(selection)
    assert window.calibration_panel.observed_edits[0].text() == ""
    window.calibration_panel.collect_peaks.setChecked(True)
    window.spectrum_plot.peak_fit_completed.emit(selection)
    assert window.calibration_panel.observed_edits[0].text() == "12.3456789"
    window.calibration_panel.known_mass_edits[0].setText("28.0")
    window.spectrum_plot.axis_mode.setCurrentText("Mass")
    assert window.calibration_panel.observed_edits[0].text() == ""
    assert window.calibration_panel.known_mass_edits[0].text() == "28.0"
    assert window.calibration_panel.observed_header.text() == "Observed m/z"
    window.calibration_panel.clear_all()
    assert window.calibration_panel.known_mass_edits[0].text() == ""
    window.close()
    assert app is not None


def test_calibration_panel_determines_constants_from_tof_rows() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    coefficients = (3.5e-6, 0.0129, 6.27)
    tof_us = np.array([1.0, 2.0, 3.5, 5.0])
    tof_ns = tof_us * 1000.0
    mz = coefficients[0] * tof_ns**2 + coefficients[1] * tof_ns + coefficients[2]
    for index, (observed, known) in enumerate(zip(tof_us, mz, strict=True)):
        window.calibration_panel.observed_edits[index].setText(f"{observed}")
        window.calibration_panel.known_mass_edits[index].setText(f"{known}")
    fitted = window.calibration_panel.determine_calibration_constants()
    assert fitted is not None
    np.testing.assert_allclose(fitted, coefficients, rtol=1e-10, atol=1e-10)
    assert window.calibration_panel.a_result.text()
    window.close()
    assert app is not None


def test_calibration_panel_determines_constants_from_mass_axis_rows() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    current = (3.5e-6, 0.0129, 6.27)
    target = (4.0e-6, 0.010, 7.0)
    tof_ns = np.array([1000.0, 2000.0, 3500.0, 5000.0])
    observed_mz = current[0] * tof_ns**2 + current[1] * tof_ns + current[2]
    known_mz = target[0] * tof_ns**2 + target[1] * tof_ns + target[2]
    window.calibration_panel.handle_axis_changed("Mass")
    for index, (observed, known) in enumerate(zip(observed_mz, known_mz, strict=True)):
        window.calibration_panel.observed_edits[index].setText(f"{observed}")
        window.calibration_panel.known_mass_edits[index].setText(f"{known}")
    fitted = window.calibration_panel.determine_calibration_constants()
    assert fitted is not None
    np.testing.assert_allclose(fitted, target, rtol=1e-10, atol=1e-10)
    window.close()
    assert app is not None


def test_main_window_loads_ini_and_saves_active_polarity_calibration(monkeypatch, tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ini_path = tmp_path / "PyTOF.ini"
    save_dir = tmp_path / "data"
    save_pytof_ini(
        ini_path,
        PyTOFIniSettings(
            save_dir=save_dir,
            n_average=3000,
            positive_calibration=(1.0, 2.0, 3.0),
            negative_calibration=(4.0, 5.0, 6.0),
        ),
    )
    window = MainWindow()
    window.file_panel.ini_path.setText(str(ini_path))
    window.load_pytof_ini()
    assert window.digitizer_panel.averages_per_record.value() == 3000
    assert str(window.file_panel.output.text()).startswith(str(save_dir))
    assert window.file_panel.output.text().endswith(".pytof")
    assert window._snapshot_config().processing.mass_calibration == (1.0, 2.0, 3.0)
    window.acquisition_panel.polarity.setCurrentText("negative")
    assert window._snapshot_config().processing.detector_polarity == -1
    assert window._snapshot_config().processing.mass_calibration == (4.0, 5.0, 6.0)
    window.calibration_panel.a_result.setText("7.0")
    window.calibration_panel.b_result.setText("8.0")
    window.calibration_panel.c_result.setText("9.0")
    window.digitizer_panel.averages_per_record.setValue(123)
    window.file_panel.output.setText(str(save_dir / "manual.pytof"))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes)
    window.save_calibration_to_ini()
    loaded = load_pytof_ini(ini_path)
    assert loaded.positive_calibration == (1.0, 2.0, 3.0)
    assert loaded.negative_calibration == (7.0, 8.0, 9.0)
    assert loaded.n_average == 123
    assert loaded.save_dir == save_dir
    window.close()
    assert app is not None


def test_repeated_spectrum_saves_use_unique_surface_filenames(tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    raw = np.zeros((1, 64), dtype=np.int16)
    raw[0, 30] = 3000
    batch = AcquisitionBatch(raw, None, sample_rate_hz=10.0, pretrigger_samples=2, first_trigger_index=0)
    digitizer = DigitizerConfig(segment_samples=64, number_of_segments=1, pretrigger_samples=2, input_range_v=1.0)
    processing = ProcessingConfig(baseline_start=0, baseline_stop=2)
    processed = process_batch(batch, digitizer, processing)
    window.spectrum_plot.set_processed(processed)
    window.file_panel.output.setText(str(tmp_path / "placeholder.pytof"))
    window.file_panel.molecule.setText("Cr6")
    window.file_panel.surface.setText("Ag111")

    window.save_cumulative_spectrum()
    first = window.file_panel.output.text()
    window.save_cumulative_spectrum()
    second = window.file_panel.output.text()

    assert first != second
    assert "Cr6_Ag111_" in first
    assert "Cr6_Ag111_" in second
    assert len(list(tmp_path.glob("Cr6_Ag111_*.pytof"))) == 2
    window.close()
    assert app is not None


def test_file_panel_keeps_spectrum_save_enabled_when_config_locked() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window._set_config_locked(True)
    assert window.file_panel.isEnabled()
    assert window.file_panel.save_cumulative_button.isEnabled()
    assert not window.file_panel.save_reference_button.isEnabled()
    assert not window.file_panel.output.isEnabled()
    assert window.file_panel.molecule.isEnabled()
    assert window.file_panel.surface.isEnabled()
    assert window.file_panel.q1.isEnabled()
    assert window.file_panel.q2.isEnabled()
    assert window.file_panel.uv.isEnabled()
    assert window.file_panel.notes.isEnabled()
    assert window.file_panel.load_spectrum_button.isEnabled()
    window._set_config_locked(False)
    assert window.file_panel.save_reference_button.isEnabled()
    assert window.file_panel.output.isEnabled()
    window.close()
    assert app is not None


def test_calibration_ini_save_button_lives_in_calibration_drawer() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    assert not hasattr(window.file_panel, "save_calibration_ini_button")
    assert window.calibration_panel.save_calibration_ini_button.text() == "Save calib to ini"
    window.close()
    assert app is not None


def test_save_calibration_to_ini_requires_confirmation(monkeypatch, tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ini_path = tmp_path / "PyTOF.ini"
    save_pytof_ini(
        ini_path,
        PyTOFIniSettings(
            save_dir=tmp_path,
            n_average=100,
            positive_calibration=(1.0, 2.0, 3.0),
            negative_calibration=(4.0, 5.0, 6.0),
        ),
    )
    window = MainWindow()
    window.file_panel.ini_path.setText(str(ini_path))
    window.load_pytof_ini()
    window.calibration_panel.a_result.setText("7.0")
    window.calibration_panel.b_result.setText("8.0")
    window.calibration_panel.c_result.setText("9.0")

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.No)
    window.save_calibration_to_ini()
    assert load_pytof_ini(ini_path).positive_calibration == (1.0, 2.0, 3.0)

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes)
    window.save_calibration_to_ini()
    assert load_pytof_ini(ini_path).positive_calibration == (7.0, 8.0, 9.0)
    window.close()
    assert app is not None


def test_startup_ini_selector_returns_selected_path_and_cancel(monkeypatch, tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ini_path = tmp_path / "PyTOF.ini"

    monkeypatch.setattr(gui_app.QtWidgets.QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(ini_path), "INI files (*.ini)"))
    assert gui_app.select_startup_ini() == ini_path

    monkeypatch.setattr(gui_app.QtWidgets.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("", ""))
    assert gui_app.select_startup_ini() is None
    assert app is not None


def test_main_window_continuous_acquisition_stops_after_multiple_batches(tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.digitizer_panel.tof_window_us.setValue(0.384)
    window.digitizer_panel.averages_per_record.setValue(2)
    window.processing_panel.baseline_stop.setValue(16)
    window.file_panel.output.setText(str(tmp_path / "continuous.pytof"))
    window.acquisition_panel.continuous.setChecked(True)

    loop = QtCore.QEventLoop()
    batches_seen = {"count": 0}
    timed_out = {"value": False}

    def on_progress(shots: int, _dropped: int) -> None:
        batches_seen["count"] += 1
        if shots >= 4:
            window.stop()

    window.start()
    assert window._worker is not None
    window._worker.progress.connect(on_progress)
    window._worker.finished.connect(loop.quit)

    timer = QtCore.QTimer()
    timer.setSingleShot(True)

    def on_timeout() -> None:
        timed_out["value"] = True
        window.stop()
        loop.quit()

    timer.timeout.connect(on_timeout)
    timer.start(10_000)
    loop.exec()
    timer.stop()
    log_text = window.log_panel.text.toPlainText()
    assert "acquired 1 mock averaged records" not in log_text
    assert not (tmp_path / "continuous.pytof").exists()
    window.save_cumulative_spectrum()
    saved_path = Path(window.file_panel.output.text())
    text = saved_path.read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("## pyTOF data saved on ")
    assert text[14]
    window.save_reference_spectrum()
    assert window.processing_panel.reference_subtraction.isChecked()
    assert window.spectrum_plot.cumulative_spectrum() is None
    with h5py.File(saved_path.with_name(f"{saved_path.stem}.reference.h5"), "r") as handle:
        assert handle["reference/trace"].shape[0] == window.digitizer_panel.segment_samples()
        assert handle["reference"].attrs["record_count"] >= 2
    window.close()
    assert not timed_out["value"]
    assert batches_seen["count"] >= 2
    assert app is not None


def test_simulation_toggle_connect_shows_error() -> None:
    """Uncheck simulation, click Connect, expect graceful error dialog (no crash)."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()

    assert window.connection_panel.simulation_mode.isChecked()
    assert window._simulation_mode

    loop = QtCore.QEventLoop()
    error_shown: dict[str, bool] = {"value": False}
    safety_timed_out: dict[str, bool] = {"value": False}

    safety = QtCore.QTimer()
    safety.setSingleShot(True)
    safety.timeout.connect(lambda: _safety_timeout(window, loop, safety_timed_out))
    safety.start(10_000)

    window.connection_panel.simulation_mode.setChecked(False)
    assert not window._simulation_mode
    assert not window.connection_panel.real_warning.isHidden()
    assert not window.connection_panel.bme_status.isHidden()

    poller = QtCore.QTimer()
    poller.timeout.connect(lambda: _close_active_message_boxes(error_shown, loop))
    poller.start(100)

    window.connection_panel.connect_button.click()

    quit_timer = QtCore.QTimer()
    quit_timer.setSingleShot(True)
    quit_timer.timeout.connect(loop.quit)
    quit_timer.start(6000)

    loop.exec()
    safety.stop()
    poller.stop()
    quit_timer.stop()

    assert error_shown["value"], "Expected an error dialog when connecting without real hardware"
    assert not safety_timed_out["value"]
    assert not window._connected
    assert window._service is None
    assert window._service_thread is None
    window.close()


def _close_active_message_boxes(error_shown: dict[str, bool], loop: QtCore.QEventLoop) -> None:
    """Close any active QMessageBox and record that one was found."""
    for widget in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(widget, QtWidgets.QMessageBox):
            error_shown["value"] = True
            widget.accept()
            loop.quit()
            return


def _safety_timeout(window: MainWindow, loop: QtCore.QEventLoop, timed_out: dict[str, bool]) -> None:
    timed_out["value"] = True
    window.close()
    loop.quit()
