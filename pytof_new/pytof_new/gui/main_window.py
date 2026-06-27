"""Main window for Phase 4 mock acquisition GUI."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from pytof_new.acquisition.real_coordinator import RealBatchCoordinator
from pytof_new.acquisition.real_worker import RealAcquisitionWorker, RealShotAnalysisWorker
from pytof_new.acquisition.worker import AcquisitionWorker, QtLogHandler, ShotAnalysisWorker
from pytof_new.config.legacy_ini import PyTOFIniSettings, load_pytof_ini, save_pytof_ini
from pytof_new.config.models import RunConfig
from pytof_new.gui.acquisition_panel import AcquisitionPanel
from pytof_new.gui.bme_panel import BMEPanel
from pytof_new.gui.calibration_panel import CalibrationPanel
from pytof_new.gui.connection_panel import ConnectionPanel
from pytof_new.gui.digitizer_panel import DigitizerPanel
from pytof_new.gui.file_panel import FilePanel
from pytof_new.gui.interaction import GuardedInteractionFilter, install_guarded_interactions
from pytof_new.gui.log_panel import LogPanel
from pytof_new.gui.mock_spectra_panel import MockSpectraPanel
from pytof_new.gui.processing_panel import ProcessingPanel
from pytof_new.gui.spectrum_plot import SpectrumPlot
from pytof_new.hardware.acquisition_planner import AcquisitionRunPlan, format_existing_plan, plan_acquisition
from pytof_new.hardware.bme_service import BMEDelayGeneratorService, BMEServiceState
from pytof_new.hardware.spectrum_limits import default_m4i2210_info
from pytof_new.hardware.spectrum_models import SpectrumAcquisitionMode, SpectrumAcquisitionPlan, SpectrumAcquisitionRequest, SpectrumHardwareInfo, SpectrumTriggerSource
from pytof_new.hardware.spectrum_service import ServiceState, SpectrumAcquisitionService
from pytof_new.storage.hdf5_writer import save_reference_spectrum
from pytof_new.storage.pytof_reader import load_pytof_spectrum
from pytof_new.storage.pytof_writer import save_pytof_spectrum


DEFAULT_POSITIVE_CALIBRATION = (3.5e-6, 0.0129, 6.27)
DEFAULT_NEGATIVE_CALIBRATION = (3.5e-6, 0.0129, 6.27)


class MainWindow(QtWidgets.QMainWindow):
    """Minimal mock-only TOF acquisition GUI."""

    _request_spectrum_configure = QtCore.Signal(object)
    _request_bme_configure = QtCore.Signal(object)
    _request_spectrum_disconnect = QtCore.Signal()
    _request_bme_disconnect = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pytof_new mock acquisition")
        self.resize(1320, 860)
        self._connected = False
        self._configured = False
        self._armed = False
        self._simulation_mode = True
        self._thread: QtCore.QThread | None = None
        self._worker: AcquisitionWorker | RealAcquisitionWorker | None = None
        self._shot_analysis_thread: QtCore.QThread | None = None
        self._shot_analysis_worker: ShotAnalysisWorker | RealShotAnalysisWorker | None = None
        self._service: SpectrumAcquisitionService | None = None
        self._service_thread: QtCore.QThread | None = None
        self._bme_service: BMEDelayGeneratorService | None = None
        self._bme_service_thread: QtCore.QThread | None = None
        self._real_spectrum_connected = False
        self._real_bme_connected = False
        self._armed_run_plan: AcquisitionRunPlan | None = None
        self._armed_config: RunConfig | None = None
        self._log_emitter = _LogEmitter(self)
        self._log_handler: QtLogHandler | None = None
        self._ini_settings = PyTOFIniSettings(
            save_dir=Path("pytof_new"),
            n_average=100,
            positive_calibration=DEFAULT_POSITIVE_CALIBRATION,
            negative_calibration=DEFAULT_NEGATIVE_CALIBRATION,
        )

        self.connection_panel = ConnectionPanel()
        self.digitizer_panel = DigitizerPanel()
        self.bme_panel = BMEPanel()
        self.acquisition_panel = AcquisitionPanel()
        self.processing_panel = ProcessingPanel()
        self.calibration_panel = CalibrationPanel()
        self.mock_spectra_panel = MockSpectraPanel()
        self.file_panel = FilePanel()
        self.spectrum_plot = SpectrumPlot()
        self.log_panel = LogPanel()
        self._active_settings_index: int | None = None
        self._interaction_filter = GuardedInteractionFilter(self)

        self._build_layout()
        install_guarded_interactions(self, self._interaction_filter)
        self._connect_signals()
        self._install_log_handler()
        self._set_config_locked(False)

    def _build_layout(self) -> None:
        settings_widget = self._build_settings_drawer()
        self.spectrum_plot.add_acquisition_controls(
            self.acquisition_panel.configure_button,
            self.acquisition_panel.start_button,
            self.acquisition_panel.stop_button,
            self.acquisition_panel.clear_button,
            self.acquisition_panel.continuous,
            self.acquisition_panel.polarity,
        )

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(self.connection_panel)
        left_layout.addWidget(self.log_panel)
        left_layout.addWidget(settings_widget)
        left_layout.addStretch(1)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(285)

        right_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.spectrum_plot, 1)
        right_split.addWidget(right_container)
        right_split.addWidget(self.file_panel)
        right_split.setSizes([620, 200])

        main_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_split.addWidget(left_scroll)
        main_split.addWidget(right_split)
        main_split.setSizes([300, 1020])
        self.setCentralWidget(main_split)

    def _build_settings_drawer(self) -> QtWidgets.QWidget:
        """Build a compact settings drawer above the plot."""
        self.digitizer_settings_button = QtWidgets.QPushButton("Digitizer")
        self.bme_settings_button = QtWidgets.QPushButton("BME")
        self.processing_settings_button = QtWidgets.QPushButton("Processing")
        self.calibration_settings_button = QtWidgets.QPushButton("Calibration")
        self.mock_spectra_settings_button = QtWidgets.QPushButton("Mock Spectra")
        self.digitizer_settings_button.setToolTip("Open or close digitizer acquisition settings.")
        self.bme_settings_button.setToolTip("Open or close BME delay-generator timing settings.")
        self.processing_settings_button.setToolTip("Open or close processing settings.")
        self.calibration_settings_button.setToolTip("Open or close calibration peak collection settings.")
        self.mock_spectra_settings_button.setToolTip("Open or close mock spectrum generation settings.")

        for button in (
            self.digitizer_settings_button,
            self.bme_settings_button,
            self.processing_settings_button,
            self.calibration_settings_button,
            self.mock_spectra_settings_button,
        ):
            button.setCheckable(True)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.digitizer_settings_button)
        bar.addWidget(self.bme_settings_button)
        bar.addWidget(self.processing_settings_button)
        bar.addWidget(self.calibration_settings_button)
        bar.addWidget(self.mock_spectra_settings_button)
        bar.addStretch(1)

        self.settings_stack = QtWidgets.QStackedWidget()
        self.settings_stack.addWidget(self.digitizer_panel)
        self.settings_stack.addWidget(self.bme_panel)
        self.settings_stack.addWidget(self.processing_panel)
        self.settings_stack.addWidget(self.calibration_panel)
        self.settings_stack.addWidget(self.mock_spectra_panel)
        self.settings_stack.setVisible(False)
        self.settings_stack.setMaximumHeight(430)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.settings_stack)

        self.digitizer_settings_button.clicked.connect(lambda: self._toggle_settings(0))
        self.bme_settings_button.clicked.connect(lambda: self._toggle_settings(1))
        self.processing_settings_button.clicked.connect(lambda: self._toggle_settings(2))
        self.calibration_settings_button.clicked.connect(lambda: self._toggle_settings(3))
        self.mock_spectra_settings_button.clicked.connect(lambda: self._toggle_settings(4))
        return container

    def _toggle_settings(self, index: int) -> None:
        """Show a settings page, or collapse it if the same page is clicked."""
        if self._active_settings_index == index and not self.settings_stack.isHidden():
            self._hide_settings()
            return
        self._active_settings_index = index
        self.settings_stack.setCurrentIndex(index)
        self.settings_stack.setVisible(True)
        self.digitizer_settings_button.setChecked(index == 0)
        self.bme_settings_button.setChecked(index == 1)
        self.processing_settings_button.setChecked(index == 2)
        self.calibration_settings_button.setChecked(index == 3)
        self.mock_spectra_settings_button.setChecked(index == 4)

    def _hide_settings(self) -> None:
        """Collapse the settings drawer."""
        self._active_settings_index = None
        self.settings_stack.setVisible(False)
        self.digitizer_settings_button.setChecked(False)
        self.bme_settings_button.setChecked(False)
        self.processing_settings_button.setChecked(False)
        self.calibration_settings_button.setChecked(False)
        self.mock_spectra_settings_button.setChecked(False)

    def _connect_signals(self) -> None:
        self.connection_panel.connect_requested.connect(self._on_connect)
        self.connection_panel.disconnect_requested.connect(self._on_disconnect)
        self.connection_panel.simulation_toggled.connect(self._on_simulation_toggled)
        self.acquisition_panel.configure_requested.connect(self.configure_run)
        self.acquisition_panel.start_requested.connect(self.start)
        self.acquisition_panel.stop_requested.connect(self.stop)
        self.acquisition_panel.clear_requested.connect(self.clear_display)
        self.file_panel.save_cumulative_requested.connect(self.save_cumulative_spectrum)
        self.file_panel.save_reference_requested.connect(self.save_reference_spectrum)
        self.file_panel.load_spectrum_requested.connect(self.load_saved_spectrum)
        self.file_panel.active_loaded_source_requested.connect(self.spectrum_plot.set_active_source)
        self.file_panel.remove_loaded_spectrum_requested.connect(self.spectrum_plot.remove_loaded_spectrum)
        self.file_panel.load_ini_requested.connect(self.load_pytof_ini)
        self.calibration_panel.save_calibration_ini_requested.connect(self.save_calibration_to_ini)
        self.spectrum_plot.loaded_spectra_changed.connect(self.file_panel.update_loaded_spectra_menu)
        self.acquisition_panel.polarity.currentTextChanged.connect(lambda _text: self._on_polarity_changed())
        self.processing_panel.reference_subtraction.toggled.connect(self._on_reference_subtraction_toggled)
        self.processing_panel.shot_analysis_requested.connect(self.start_shot_analysis)
        self.spectrum_plot.peak_fit_completed.connect(self.calibration_panel.add_peak_fit)
        self.spectrum_plot.axis_mode.currentTextChanged.connect(self.calibration_panel.handle_axis_changed)
        self.bme_panel.settings_changed.connect(self._update_digitizer_repetition_warning)
        self.bme_panel.settings_changed.connect(self._invalidate_armed_plan)
        self.bme_panel.settings_changed.connect(self._refresh_plan_preview)
        self.digitizer_panel.tof_window_us.valueChanged.connect(lambda value: self.bme_panel.set_tof_window_us(float(value)))
        self.digitizer_panel.sample_rate.currentIndexChanged.connect(lambda _value: self._update_smoothing_interval())
        self.digitizer_panel.settings_changed.connect(self._invalidate_armed_plan)
        self.digitizer_panel.settings_changed.connect(self._refresh_plan_preview)
        self.bme_panel.set_tof_window_us(float(self.digitizer_panel.tof_window_us.value()))
        self._update_digitizer_repetition_warning()
        self._update_smoothing_interval()
        self._refresh_plan_preview()

    def _install_log_handler(self) -> None:
        self._log_emitter.message.connect(self.log_panel.append)
        self._log_handler = QtLogHandler(self._log_emitter.message)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(self._log_handler)

    def _update_digitizer_repetition_warning(self) -> None:
        """Update digitizer TOF-window warning from current BME repetition period."""
        self.digitizer_panel.set_repetition_period_s(self.bme_panel.config().repetition_period_s)

    def _update_smoothing_interval(self) -> None:
        """Update processing smoothing interval from current digitizer sample rate."""
        self.processing_panel.set_sample_rate_hz(float(self.digitizer_panel.sample_rate.currentData()))

    def _on_polarity_changed(self) -> None:
        """Update calibration inversion constants when detector polarity changes."""
        self.calibration_panel.set_current_coefficients(self._active_calibration())
        self.digitizer_panel.set_calibration(self._active_calibration())
        self._invalidate_armed_plan()
        self._refresh_plan_preview()

    def _invalidate_armed_plan(self) -> None:
        """Require re-arming after settings change."""
        if self._armed_run_plan is not None:
            self.log_panel.append("Settings changed; re-arm before Start to use the new plan")
        self._armed_run_plan = None
        self._armed_config = None
        self._configured = False
        self._armed = False

    def _refresh_plan_preview(self) -> None:
        """Refresh the Basic-mode plan preview without configuring hardware."""
        try:
            config = self._snapshot_config()
            plan = self._plan_request(config)
        except Exception as exc:
            self.digitizer_panel.set_plan_error(str(exc))
            return
        self.digitizer_panel.set_plan_preview(plan.summary_lines, plan.warnings)

    @QtCore.Slot()
    def _on_connect(self) -> None:
        """Connect mock or real hardware based on simulation mode."""
        if self._simulation_mode:
            self._connect_mock_hardware()
        else:
            self._connect_real_hardware()

    @QtCore.Slot()
    def _on_disconnect(self) -> None:
        """Disconnect mock or real hardware."""
        if self._simulation_mode:
            self._disconnect_mock_hardware()
        else:
            self._disconnect_real_hardware()

    @QtCore.Slot(bool)
    def _on_simulation_toggled(self, checked: bool) -> None:
        self._simulation_mode = checked
        if not checked:
            self.log_panel.append("Simulation mode disabled — next connect will attempt real Spectrum and BME hardware")
        else:
            self.log_panel.append("Simulation mode enabled — next connect will use mock hardware")

    def _connect_mock_hardware(self) -> None:
        """Mark mock devices as connected."""
        self._connected = True
        self.connection_panel.set_connected(True)
        self.acquisition_panel.set_state("CONNECTED")
        self.log_panel.append("Connected to mock digitizer and mock BME")

    def _disconnect_mock_hardware(self) -> None:
        """Disconnect mock devices if acquisition is idle."""
        if self._thread is not None:
            self.stop()
            return
        self._connected = False
        self._configured = False
        self._armed = False
        self.connection_panel.set_connected(False)
        self.acquisition_panel.set_state("DISCONNECTED")
        self.log_panel.append("Disconnected mock hardware")

    def _connect_real_hardware(self) -> None:
        """Connect to real Spectrum and BME services."""
        if self._service is not None or self._bme_service is not None:
            return
        self._real_spectrum_connected = False
        self._real_bme_connected = False
        self.connection_panel.set_connecting(True)
        self._service = SpectrumAcquisitionService()
        self._service_thread = QtCore.QThread(self)
        self._service.moveToThread(self._service_thread)
        self._service_thread.started.connect(self._service.connect_card)
        self._service.hardware_info_ready.connect(self._on_spectrum_hardware_info)
        self._service.error_occurred.connect(self._on_real_error)
        self._service.log_message.connect(self.log_panel.append)
        self._service.state_changed.connect(self._on_service_state_changed)
        self._request_spectrum_configure.connect(self._service.configure, QtCore.Qt.ConnectionType.BlockingQueuedConnection)
        self._request_spectrum_disconnect.connect(self._service.disconnect_card, QtCore.Qt.ConnectionType.BlockingQueuedConnection)
        self._service_thread.start()

    def _start_bme_service(self) -> None:
        if self._bme_service is not None:
            return
        self._bme_service = BMEDelayGeneratorService()
        self._bme_service_thread = QtCore.QThread(self)
        self._bme_service.moveToThread(self._bme_service_thread)
        self._bme_service_thread.started.connect(self._bme_service.connect_card)
        self._bme_service.hardware_info_ready.connect(self._on_bme_hardware_info)
        self._bme_service.error_occurred.connect(self._on_real_error)
        self._bme_service.log_message.connect(self.log_panel.append)
        self._bme_service.state_changed.connect(self._on_bme_service_state_changed)
        self._request_bme_configure.connect(self._bme_service.configure, QtCore.Qt.ConnectionType.BlockingQueuedConnection)
        self._request_bme_disconnect.connect(self._bme_service.disconnect_card, QtCore.Qt.ConnectionType.BlockingQueuedConnection)
        self._bme_service_thread.start()

    @QtCore.Slot(object)
    def _on_spectrum_hardware_info(self, info: object) -> None:
        self._real_spectrum_connected = True
        self._start_bme_service()
        self._update_real_connection_ready()
        self.log_panel.append(f"Connected to real Spectrum digitizer: {info}")

    @QtCore.Slot(object)
    def _on_bme_hardware_info(self, info: object) -> None:
        self._real_bme_connected = True
        self._update_real_connection_ready()
        self.log_panel.append(f"Connected to real BME delay generator: {info}")

    def _update_real_connection_ready(self) -> None:
        if not (self._real_spectrum_connected and self._real_bme_connected):
            return
        self._connected = True
        self.connection_panel.set_connecting(False)
        self.connection_panel.set_connected(True)
        self.acquisition_panel.set_state("CONNECTED")

    @QtCore.Slot(str)
    def _on_service_state_changed(self, state: str) -> None:
        self.connection_panel.set_service_state(state)

    @QtCore.Slot(str)
    def _on_bme_service_state_changed(self, state: str) -> None:
        self.connection_panel.set_bme_service_state(state)

    @QtCore.Slot(str)
    def _on_real_error(self, message: str) -> None:
        self._show_error(message)
        self.connection_panel.set_connecting(False)
        if not self._connected:
            self._cleanup_real_service()

    def _disconnect_real_hardware(self) -> None:
        """Disconnect the real digitizer service."""
        if self._thread is not None:
            self.stop()
            return
        self._cleanup_real_service()
        self._connected = False
        self._configured = False
        self._armed = False
        self.connection_panel.set_connected(False)
        self.acquisition_panel.set_state("DISCONNECTED")
        self.log_panel.append("Disconnected real hardware")

    def _cleanup_real_service(self) -> None:
        try:
            self._request_spectrum_configure.disconnect()
        except (RuntimeError, TypeError):
            pass
        if self._bme_service is not None:
            try:
                self._request_bme_configure.disconnect()
            except (RuntimeError, TypeError):
                pass
        if self._bme_service is not None and self._bme_service_thread is not None and self._bme_service_thread.isRunning():
            self._request_bme_disconnect.emit()
        if self._service is not None and self._service_thread is not None and self._service_thread.isRunning():
            self._request_spectrum_disconnect.emit()
        try:
            self._request_spectrum_disconnect.disconnect()
        except (RuntimeError, TypeError):
            pass
        if self._bme_service is not None:
            try:
                self._request_bme_disconnect.disconnect()
            except (RuntimeError, TypeError):
                pass
        if self._bme_service_thread is not None:
            if self._bme_service_thread.isRunning():
                self._bme_service_thread.quit()
                self._bme_service_thread.wait(3000)
            self._bme_service_thread = None
        self._bme_service = None
        if self._service_thread is not None:
            if self._service_thread.isRunning():
                self._service_thread.quit()
                self._service_thread.wait(3000)
            self._service_thread = None
        self._service = None
        self._real_spectrum_connected = False
        self._real_bme_connected = False

    @QtCore.Slot()
    def configure_run(self) -> None:
        """Validate current settings and arm the digitizer."""
        if not self._connected:
            self._on_connect()
            if not self._connected:
                return
        if not self._simulation_mode and not self._real_services_ready_for_configure():
            spectrum_state = self._service.state.value if self._service is not None else "missing"
            bme_state = self._bme_service.state.value if self._bme_service is not None else "missing"
            self.log_panel.append(f"Waiting for hardware connection (Spectrum: {spectrum_state}, BME: {bme_state}) — try Configure again")
            return
        try:
            config = self._snapshot_config()
            config.validate()
            run_plan = self._plan_request(config)
            if not self._simulation_mode:
                config = self._config_with_request(config, run_plan.primary_request)
            self.log_panel.append("=== Acquisition Plan ===")
            for line in run_plan.summary_lines:
                self.log_panel.append(line)
            self.log_panel.append("========================")
            if not self._simulation_mode and self._service is not None:
                if config.digitizer.advanced_mode:
                    run_plan = self._plan_from_advanced_config(config)
                    config = self._config_with_request(config, run_plan.primary_request)
                self._configure_real_services(run_plan.primary_request, config)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._armed_run_plan = run_plan
        self._armed_config = config
        if not self._simulation_mode:
            self._sync_digitizer_panel_to_request(run_plan.primary_request)
        self._configured = True
        self._armed = True
        self.acquisition_panel.set_state("ARMED")
        label = "mock" if self._simulation_mode else "real"
        self.log_panel.append(f"Configuration validated and {label} digitizer armed")

    def _real_services_ready_for_configure(self) -> bool:
        if self._service is None or self._bme_service is None:
            return False
        spectrum_ready = self._service.state in (ServiceState.CONNECTED, ServiceState.CONFIGURED)
        bme_ready = self._bme_service.state in (BMEServiceState.CONNECTED, BMEServiceState.CONFIGURED)
        return spectrum_ready and bme_ready

    def _configure_real_services(self, request: SpectrumAcquisitionRequest, config: RunConfig) -> None:
        if self._service is None or self._bme_service is None:
            raise RuntimeError("Real hardware services are not connected")
        self._request_bme_configure.emit(config.bme)
        self._request_spectrum_configure.emit(request)

    def _build_real_request(self, config: RunConfig) -> SpectrumAcquisitionRequest:
        """Direct mapping from config to request (advanced mode / manual override)."""
        mode_map = {
            "hardware_average": SpectrumAcquisitionMode.AVERAGE_32BIT,
            "raw_segments": SpectrumAcquisitionMode.RAW_MULTI,
        }
        trigger_map = {
            "external0": SpectrumTriggerSource.EXTERNAL0,
            "software": SpectrumTriggerSource.SOFTWARE,
            "channel0": SpectrumTriggerSource.CHANNEL0,
        }
        mode = mode_map.get(config.digitizer.record_mode, SpectrumAcquisitionMode.RAW_MULTI)
        trigger = trigger_map.get(config.digitizer.trigger_source, SpectrumTriggerSource.SOFTWARE)
        if (mode == SpectrumAcquisitionMode.AVERAGE_32BIT or mode == SpectrumAcquisitionMode.AVERAGE_16BIT) and trigger == SpectrumTriggerSource.SOFTWARE:
            self.log_panel.append(
                "WARNING: Block average mode does not support software trigger. "
                "Forcing trigger source to external0."
            )
            trigger = SpectrumTriggerSource.EXTERNAL0
        return SpectrumAcquisitionRequest(
            mode=mode,
            sample_rate_hz=config.digitizer.sample_rate_hz,
            segment_samples=config.digitizer.segment_samples,
            pretrigger_samples=config.digitizer.pretrigger_samples,
            number_of_segments=config.digitizer.number_of_segments,
            averages_per_segment=config.digitizer.hardware_averages_per_record,
            trigger_source=trigger,
            input_range_v=config.digitizer.input_range_v,
            trigger_level_v=config.digitizer.trigger_level_v,
            timeout_s=config.digitizer.timeout_s,
            coupling=config.digitizer.coupling,
            bandwidth_limit_enabled=config.digitizer.bandwidth_limit_enabled,
            trigger_edge=config.digitizer.trigger_edge,
            trigger_termination_ohm=config.digitizer.trigger_termination_ohm,
        )

    def _plan_request(self, config: RunConfig) -> AcquisitionRunPlan:
        """Use the automatic planner to generate a request and plan text."""
        hardware_info = self._service._digitizer.hardware_info if self._service else default_m4i2210_info()
        if hardware_info is None:
            hardware_info = default_m4i2210_info()
        repetition_period_s = config.bme.repetition_period_s
        calibration = self._active_calibration()
        return plan_acquisition(
            acquisition_workflow=config.digitizer.acquisition_workflow,
            acquisition_priority=config.digitizer.acquisition_priority,
            tof_window_us=config.digitizer.tof_window_us,
            target_display_interval_s=config.digitizer.target_update_interval_s,
            total_shots=config.digitizer.total_shots,
            pretrigger_samples=config.digitizer.pretrigger_samples,
            averages_per_segment=config.digitizer.hardware_averages_per_record,
            hardware_info=hardware_info,
            repetition_period_s=repetition_period_s,
            calibration=calibration,
            advanced_mode=config.digitizer.advanced_mode,
            manual_sample_rate_hz=config.digitizer.sample_rate_hz if config.digitizer.advanced_mode else None,
            manual_accumulator_mode=config.digitizer.accumulator_mode,
            manual_fpga_sums_per_batch=config.digitizer.fpga_sums_per_batch if config.digitizer.advanced_mode else None,
            manual_raw_shots_per_batch=config.digitizer.raw_shots_per_batch if config.digitizer.advanced_mode else None,
            manual_segment_samples=config.digitizer.segment_samples if config.digitizer.advanced_mode and config.digitizer.override_segment_samples else None,
            input_range_v=config.digitizer.input_range_v,
            coupling=config.digitizer.coupling,
            bandwidth_limit_enabled=config.digitizer.bandwidth_limit_enabled,
            trigger_source=config.digitizer.trigger_source if config.digitizer.advanced_mode else "external0",
            trigger_edge=config.digitizer.trigger_edge,
            trigger_level_v=config.digitizer.trigger_level_v,
            trigger_termination_ohm=config.digitizer.trigger_termination_ohm,
            timeout_s=config.digitizer.timeout_s,
        )

    def _plan_from_advanced_config(self, config: RunConfig) -> AcquisitionRunPlan:
        return self._plan_request(config)

    def _config_with_request(self, config: RunConfig, request: SpectrumAcquisitionRequest) -> RunConfig:
        digitizer = replace(
            config.digitizer,
            sample_rate_hz=request.sample_rate_hz,
            segment_samples=request.segment_samples,
            pretrigger_samples=request.pretrigger_samples,
            number_of_segments=request.number_of_segments,
            hardware_averages_per_record=request.averages_per_segment,
            record_mode="hardware_average" if request.mode in (SpectrumAcquisitionMode.AVERAGE_32BIT, SpectrumAcquisitionMode.AVERAGE_16BIT) else "raw_segments",
            input_range_v=request.input_range_v,
            trigger_source=request.trigger_source.value,
            trigger_edge=request.trigger_edge,
            trigger_level_v=request.trigger_level_v,
            trigger_termination_ohm=request.trigger_termination_ohm,
            timeout_s=request.timeout_s,
        )
        return replace(config, digitizer=digitizer)

    def _sync_digitizer_panel_to_request(self, request: SpectrumAcquisitionRequest) -> None:
        """Reflect planner-derived sample rate in calculated UI readouts without invalidating Arm."""
        combo = self.digitizer_panel.sample_rate
        previous_blocked = combo.blockSignals(True)
        for index in range(combo.count()):
            if float(combo.itemData(index)) == float(request.sample_rate_hz):
                combo.setCurrentIndex(index)
                break
        combo.blockSignals(previous_blocked)
        self.digitizer_panel._update_calculated_segment_samples()
        self._update_smoothing_interval()

    @QtCore.Slot()
    def start(self) -> None:
        """Start finite or continuous acquisition."""
        if not self._configured or not self._armed:
            self.configure_run()
            if not self._armed:
                return
        if self._armed_run_plan is None:
            self._show_error("Settings changed after arming. Re-arm before Start.")
            return
        self._start_worker(continuous=self._armed_run_plan.continuous)

    def _start_worker(self, continuous: bool) -> None:
        if self._thread is not None:
            return
        if not self._connected:
            self._on_connect()
            if not self._connected:
                return
        try:
            if self._armed_config is None or self._armed_run_plan is None:
                raise RuntimeError("No armed acquisition plan is available")
            config = self._armed_config
            run_plan = self._armed_run_plan
            config.validate()
            if not self._simulation_mode and self._service is not None:
                self._configure_real_services(run_plan.primary_request, config)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._configured = True
        self._armed = True
        self._set_config_locked(True)
        self.acquisition_panel.set_running(True)
        self.acquisition_panel.set_state("ACQUIRING")
        if not self._simulation_mode:
            label = "real"
            self.bme_panel.set_outputs_enabled(False)
            self.log_panel.append("Real acquisition will arm Spectrum first, then start the BME trigger sequence.")
        else:
            label = "mock"
            self.bme_panel.set_outputs_enabled(True)

        self._thread = QtCore.QThread(self)
        if self._simulation_mode:
            self._worker = AcquisitionWorker(config, continuous=continuous)
        else:
            if self._service is None or self._bme_service is None:
                self._show_error("Real hardware services are not connected")
                return
            coordinator = RealBatchCoordinator(self._service, self._bme_service, parent=self)
            coordinator.state_changed.connect(lambda state: self.log_panel.append(f"Coordinator: {state}"))
            self._worker = RealAcquisitionWorker(self._service, config, run_plan, coordinator=coordinator, continuous=continuous)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.batch_ready.connect(self._on_batch_ready)
        self._worker.progress.connect(self.acquisition_panel.set_progress)
        self._worker.log_message.connect(self.log_panel.append)
        self._worker.warning.connect(self.log_panel.append)
        self._worker.error.connect(self._show_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()
        self.log_panel.append(f"{label} acquisition worker started")

    @QtCore.Slot()
    def stop(self) -> None:
        """Request controlled acquisition stop."""
        if self._worker is not None:
            self.acquisition_panel.set_state("STOPPING")
            self._worker.request_stop()
        if self._shot_analysis_worker is not None:
            self._shot_analysis_worker.request_stop()

    @QtCore.Slot(object, object)
    def _on_batch_ready(self, _batch: object, processed: object) -> None:
        self.spectrum_plot.set_processed(processed)

    @QtCore.Slot()
    def clear_display(self) -> None:
        """Clear plots, counters, and log messages."""
        self.spectrum_plot.clear_all()
        self.log_panel.text.clear()
        self.acquisition_panel.set_progress(0, 0)

    @QtCore.Slot()
    def save_cumulative_spectrum(self) -> None:
        """Save the current cumulative average spectrum on user request."""
        spectrum = self.spectrum_plot.cumulative_spectrum()
        if spectrum is None:
            self._show_error("No cumulative spectrum is available to save.")
            return
        axis, trace, record_count = spectrum
        try:
            output_path = self.file_panel.next_spectrum_output_path()
            self.file_panel.output.setText(str(output_path))
            config = self._snapshot_config()
            save_pytof_spectrum(config.storage.output_path, axis, trace, config, axis_mode=self.spectrum_plot.axis_mode.currentText())
        except Exception as exc:
            self._show_error(f"Could not save cumulative spectrum: {exc}")
            return
        self.log_panel.append(f"Saved cumulative spectrum to {config.storage.output_path}")

    @QtCore.Slot()
    def save_reference_spectrum(self) -> None:
        """Save the current unsmoothed cumulative average as a subtraction reference."""
        spectrum = self.spectrum_plot.reference_spectrum()
        if spectrum is None:
            self._show_error("No cumulative spectrum is available to save as a reference.")
            return
        axis, trace, record_count = spectrum
        try:
            config = self._snapshot_config()
            reference_path = _reference_output_path(config.storage.output_path)
            save_reference_spectrum(reference_path, axis, trace, record_count, config)
        except Exception as exc:
            self._show_error(f"Could not save reference spectrum: {exc}")
            return
        self.processing_panel.set_reference_loaded(str(reference_path), record_count)
        self.spectrum_plot.clear_all()
        self.acquisition_panel.set_progress(0, 0)
        self.log_panel.append(f"Saved reference spectrum to {reference_path}")
        self.log_panel.append("Reference subtraction enabled; cleared plots to avoid mixing corrected and uncorrected records")

    @QtCore.Slot()
    def load_saved_spectrum(self) -> None:
        """Load a saved .pytof spectrum overlay for comparison and calibration."""
        start_dir = str(Path(self.file_panel.output.text()).parent)
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Load .pytof spectrum", start_dir, "pyTOF spectra (*.pytof);;All files (*)")
        if not path:
            return
        try:
            spectrum = load_pytof_spectrum(Path(path))
            self.spectrum_plot.add_loaded_spectrum(spectrum)
        except Exception as exc:
            self._show_error(f"Could not load .pytof spectrum: {exc}")
            return
        self.log_panel.append(f"Loaded spectrum overlay: {path}")

    @QtCore.Slot(bool)
    def _on_reference_subtraction_toggled(self, enabled: bool) -> None:
        """Clear accumulation when reference subtraction state changes."""
        if not enabled:
            return
        if not self.processing_panel.reference_path.text().strip():
            return
        self.spectrum_plot.clear_all()
        self.acquisition_panel.set_progress(0, 0)
        self.log_panel.append("Reference subtraction enabled; cleared plots")

    @QtCore.Slot()
    def start_shot_analysis(self) -> None:
        """Run delayed one-record cross-correlation timing-jitter analysis."""
        if self._thread is not None or self._shot_analysis_thread is not None:
            self._show_error("Stop the current acquisition or shot analysis before starting shot analysis.")
            return
        if not self._connected:
            self._on_connect()
            if not self._connected:
                return
        try:
            base_config = self._snapshot_config()
            # Force raw-segments record mode: shot analysis needs individual shots
            digitizer = replace(
                base_config.digitizer,
                record_mode="raw_segments",
                hardware_averages_per_record=1,
                number_of_segments=1,
            )
            record_count, delay_s, max_lag_s, min_mz = self.processing_panel.shot_analysis_settings()
            bme = replace(base_config.bme, repetition_period_s=delay_s)
            config = replace(base_config, digitizer=digitizer, bme=bme)
            config.validate()
            if not self._simulation_mode and self._service is not None:
                if not self._real_services_ready_for_configure():
                    spectrum_state = self._service.state.value if self._service is not None else "missing"
                    bme_state = self._bme_service.state.value if self._bme_service is not None else "missing"
                    self.log_panel.append(
                        f"Waiting for hardware connection (Spectrum: {spectrum_state}, BME: {bme_state}) — try Shot Analysis again"
                    )
                    return
                if config.digitizer.advanced_mode:
                    request = self._build_real_request(config)
                else:
                    # Bypass the generic planner: shot analysis needs a single
                    # raw shot per worker iteration at max time resolution.
                    request = SpectrumAcquisitionRequest(
                        mode=SpectrumAcquisitionMode.RAW_MULTI,
                        sample_rate_hz=1.25e9,
                        segment_samples=config.digitizer.segment_samples,
                        pretrigger_samples=config.digitizer.pretrigger_samples,
                        number_of_segments=1,
                        trigger_source=SpectrumTriggerSource.EXTERNAL0,
                        input_range_v=config.digitizer.input_range_v,
                        trigger_level_v=config.digitizer.trigger_level_v,
                        timeout_s=config.digitizer.timeout_s,
                    )
                    self.log_panel.append("=== Shot Analysis Plan ===")
                    self.log_panel.append("Mode: RAW_MULTI (1 shot per batch)")
                    self.log_panel.append("Sample rate: 1250 MS/s")
                    self.log_panel.append(f"Segment size: {request.segment_samples} samples (including {request.pretrigger_samples} pretrigger)")
                    self.log_panel.append("Number of segments: 1")
                    self.log_panel.append("External trigger required")
                    self.log_panel.append("==========================")
                self._configure_real_services(request, config)
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._set_config_locked(True)
        self.processing_panel.setEnabled(True)
        self.processing_panel.set_shot_analysis_running(True)
        self.acquisition_panel.set_running(True)
        self._shot_analysis_thread = QtCore.QThread(self)
        if self._simulation_mode:
            self._shot_analysis_worker = ShotAnalysisWorker(config, record_count, delay_s, max_lag_s, min_mz)
        else:
            self._shot_analysis_worker = RealShotAnalysisWorker(
                self._service, config, record_count, delay_s, max_lag_s, min_mz,
            )
        self._shot_analysis_worker.moveToThread(self._shot_analysis_thread)
        self._shot_analysis_thread.started.connect(self._shot_analysis_worker.run)
        self._shot_analysis_worker.progress.connect(self.processing_panel.set_shot_analysis_progress)
        self._shot_analysis_worker.display_ready.connect(self.spectrum_plot.set_shot_analysis_display)
        self._shot_analysis_worker.result_ready.connect(self._on_shot_analysis_result)
        self._shot_analysis_worker.log_message.connect(self.log_panel.append)
        self._shot_analysis_worker.warning.connect(self.log_panel.append)
        self._shot_analysis_worker.error.connect(self._show_error)
        self._shot_analysis_worker.finished.connect(self._on_shot_analysis_finished)
        self._shot_analysis_worker.finished.connect(self._shot_analysis_thread.quit)
        self._shot_analysis_worker.finished.connect(self._shot_analysis_worker.deleteLater)
        self._shot_analysis_thread.finished.connect(self._shot_analysis_thread.deleteLater)
        self._shot_analysis_thread.start()
        self.log_panel.append("Shot analysis: 1 raw shot per batch at 1.25 GS/s")

    @QtCore.Slot(object)
    def _on_shot_analysis_result(self, result: object) -> None:
        self.processing_panel.set_shot_analysis_result(result)
        self.log_panel.append(
            "Shot analysis result: "
            f"RMS {result.rms_jitter_s * 1e9:.3g} ns, "
            f"mean {result.mean_shift_s * 1e9:.3g} ns, "
            f"uncertainty {result.uncertainty_s * 1e9:.3g} ns, "
            f"N={result.record_count}"
        )

    @QtCore.Slot()
    def _on_shot_analysis_finished(self) -> None:
        self._shot_analysis_thread = None
        self._shot_analysis_worker = None
        self.processing_panel.set_shot_analysis_running(False)
        self.acquisition_panel.set_running(False)
        self._set_config_locked(False)

    @QtCore.Slot()
    def _on_worker_finished(self) -> None:
        self._thread = None
        self._worker = None
        self.bme_panel.set_outputs_enabled(False)
        self.acquisition_panel.set_running(False)
        self._set_config_locked(False)
        if self._connected:
            self.acquisition_panel.set_state("CONFIGURED")
        self.log_panel.append("Acquisition worker finished")

    def _snapshot_config(self) -> RunConfig:
        return RunConfig(
            digitizer=self.digitizer_panel.config(),
            bme=self.bme_panel.config(),
            processing=self.processing_panel.config(
                detector_polarity=self.acquisition_panel.detector_polarity(),
                mass_calibration=self._active_calibration(),
            ),
            storage=self.file_panel.config(),
            mock_spectra=self.mock_spectra_panel.config(),
        )

    @QtCore.Slot()
    def load_pytof_ini(self) -> bool:
        """Load legacy PyTOF.ini defaults and calibration constants."""
        try:
            settings = load_pytof_ini(Path(self.file_panel.ini_path.text()))
        except Exception as exc:
            self._show_error(f"Could not load PyTOF.ini: {exc}")
            return False
        self._ini_settings = settings
        self.digitizer_panel.averages_per_record.setValue(settings.n_average)
        self.digitizer_panel.set_calibration(self._active_calibration())
        self.file_panel.update_default_output_path(settings.save_dir)
        self.calibration_panel.set_current_coefficients(self._active_calibration())
        self.log_panel.append(f"Loaded PyTOF.ini from {self.file_panel.ini_path.text()}")
        return True

    @QtCore.Slot()
    def save_calibration_to_ini(self) -> None:
        """Save current determined calibration constants to PyTOF.ini for the active polarity."""
        coefficients = self.calibration_panel.current_result_coefficients()
        if coefficients is None:
            self._show_error("No determined calibration constants are available to save.")
            return
        if not self._confirm_save_calibration_to_ini(coefficients):
            self.log_panel.append("Calibration save cancelled")
            return
        save_dir = Path(self.file_panel.output.text()).parent
        settings = replace(self._ini_settings, save_dir=save_dir, n_average=int(self.digitizer_panel.averages_per_record.value()))
        if self.acquisition_panel.polarity.currentText() == "positive":
            settings = replace(settings, positive_calibration=coefficients)
        else:
            settings = replace(settings, negative_calibration=coefficients)
        try:
            save_pytof_ini(Path(self.file_panel.ini_path.text()), settings)
        except Exception as exc:
            self._show_error(f"Could not save calibration to PyTOF.ini: {exc}")
            return
        self._ini_settings = settings
        self.calibration_panel.set_current_coefficients(self._active_calibration())
        self.log_panel.append(f"Saved {self.acquisition_panel.polarity.currentText()} calibration to {self.file_panel.ini_path.text()}")

    def _confirm_save_calibration_to_ini(self, coefficients: tuple[float, float, float]) -> bool:
        polarity = self.acquisition_panel.polarity.currentText()
        ini_path = self.file_panel.ini_path.text()
        a, b, c = coefficients
        message = (
            f"Save {polarity} calibration constants to PyTOF.ini?\n\n"
            f"File: {ini_path}\n"
            f"A: {a:.9g}\n"
            f"B: {b:.9g}\n"
            f"C: {c:.9g}"
        )
        result = QtWidgets.QMessageBox.question(
            self,
            "Confirm calibration save",
            message,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return result == QtWidgets.QMessageBox.StandardButton.Yes

    def _active_calibration(self) -> tuple[float, float, float]:
        if self.acquisition_panel.polarity.currentText() == "negative":
            return self._ini_settings.negative_calibration
        return self._ini_settings.positive_calibration

    def _set_config_locked(self, locked: bool) -> None:
        self.digitizer_panel.setEnabled(not locked)
        self.bme_panel.setEnabled(not locked)
        self.processing_panel.setEnabled(not locked)
        self.calibration_panel.setEnabled(not locked)
        self.mock_spectra_panel.setEnabled(not locked)
        self.file_panel.set_locked_for_acquisition(locked)
        self.connection_panel.disconnect_button.setEnabled(self._connected and not locked)

    def _show_error(self, message: str) -> None:
        self.log_panel.append(f"ERROR: {message}")
        QtWidgets.QMessageBox.critical(self, "pytof_new error", message)

    def closeEvent(self, event: object) -> None:
        """Request a controlled stop when the window closes."""
        if self._worker is not None:
            self.stop()
            QtWidgets.QApplication.processEvents()
        if self._shot_analysis_worker is not None:
            self._shot_analysis_worker.request_stop()
            QtWidgets.QApplication.processEvents()
        if self._thread is not None:
            self._thread.wait(5000)
        if self._shot_analysis_thread is not None:
            self._shot_analysis_thread.wait(5000)
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
        super().closeEvent(event)


class _LogEmitter(QtCore.QObject):
    """Owns a Qt signal used to marshal logs into the GUI thread."""

    message = QtCore.Signal(str)


def _reference_output_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}.reference.h5")
