# pytof_new

`pytof_new` is the modular Python/PySide6 rewrite of the legacy `pyTOF-v23.py` time-of-flight mass spectrometry acquisition application.

The legacy root-level scripts and vendor files are kept as hardware-call references. The active application lives under `pytof_new/pytof_new` and should be edited there unless a hardware probe or legacy comparison specifically requires the old files.

## Current State

Implemented:

- Immutable run configuration models and legacy `PyTOF.ini` load/save helpers.
- Mock digitizer and mock delay-generator path for development without hardware.
- Processing pipeline for baseline subtraction, filtering, reference subtraction, shot rejection, averaging, smoothing, mass conversion, peak fitting, calibration, and jitter analysis.
- HDF5 run/reference storage and legacy text `.pytof` spectrum import/export.
- PySide6 GUI with Basic/Advanced digitizer controls, live plan preview, acquisition controls, plot display, calibration tools, metadata entry, and saved-spectrum overlays.
- Finite-batch Spectrum M4i.2210-x8 acquisition planner for Live Averaged Spectrum, Live Raw-Shot Spectrum, and Finite Shot Analysis.
- Real Spectrum service path for finite non-FIFO `RAW_MULTI`, `AVERAGE_32BIT`, and `AVERAGE_16BIT` batches.
- Real BME SG08p service path with lazy `DelayGenerator.dll` loading, safe configuration while inactive, finite trigger-count arming, and coordinated Spectrum+BME batch start.

Known limitations:

- Real BME output timing and electrical levels still require oscilloscope validation on the target hardware before connecting BME outputs to the Spectrum trigger input or extraction pulser.
- Shot analysis still uses the existing Spectrum-only real worker after configuring BME timing. The normal Start path uses coordinated Spectrum+BME batches.

## Dependencies

The package metadata is in `pyproject.toml`.

Runtime dependencies:

- Python `>=3.12`
- `numpy`
- `scipy`
- `h5py`
- `PySide6`
- `pyqtgraph`

Test dependency:

- `pytest`

Hardware/runtime-only dependencies for real operation:

- Spectrum SDK / `pyspcm` wrapper available on `PYTHONPATH`
- Spectrum register constants from the SDK environment
- A Spectrum M4i.2210-x8 card reachable as `/dev/spcm0` by default, or another device path passed to diagnostics/probe scripts
- BME `DelayGenerator.dll` matching the Python process architecture
- BME SG08p card reachable through the BME DLL
- Vendor support DLLs required by the Spectrum and BME stacks

Local root-level files such as `pyspcm.py`, `spcm_tools.py`, `DelayGenerator.dll`, `DG_DLL_1.h`, `DG_DLL_1.c`, and `PlxApi720_x64.dll` are hardware/vendor support files. Do not move or rewrite them unless you are deliberately updating vendor integration.

## Common Commands

Run the GUI:

```powershell
py -3.12 pytof_new\app.py
```

Run all tests:

```powershell
py -3.12 -m pytest pytof_new\tests
```

Run a mock acquisition script:

```powershell
py -3.12 pytof_new\scripts\mock_acquire_batch.py --output pytof_new\mock_run.h5 --averages-per-record 100
```

Run Spectrum hardware diagnostics from the installed package command path:

```powershell
py -3.12 -m pytof_new.cli diagnose --device /dev/spcm0 --mode raw_multi
```

The CLI hardware diagnostic is gated by `PYTOF_RUN_HARDWARE_TESTS=1`.

Run the standalone Spectrum probe script:

```powershell
py -3.12 pytof_new\scripts\probe_spectrum.py --info
py -3.12 pytof_new\scripts\probe_spectrum.py --raw-multi --segs 8
py -3.12 pytof_new\scripts\probe_spectrum.py --average --avg-records 8
py -3.12 pytof_new\scripts\probe_spectrum.py --average16 --avg-records 8
```

Run the standalone BME probe script safely:

```powershell
py -3.12 pytof_new\scripts\probe_bme.py --info
py -3.12 pytof_new\scripts\probe_bme.py --configure --enabled-channels A --tof-window-us 50 --fill-us 55
py -3.12 pytof_new\scripts\probe_bme.py --arm --enabled-channels A --pulse-count 1
```

BME output activation is gated:

```powershell
$env:PYTOF_RUN_HARDWARE_TESTS=1
py -3.12 pytof_new\scripts\probe_bme.py --pulse-test --enabled-channels A --pulse-count 1 --tof-window-us 50 --fill-us 55
py -3.12 pytof_new\scripts\probe_bme.py --pulse-test --enabled-channels A,C,F --pulse-count 1 --tof-window-us 50 --fill-us 55
```

The first pulse test should use only channel A into an oscilloscope/safe load. Add C and F only after A is verified. The pulse test polls the BME trigger counter and exits nonzero if the expected count is not reached.

Run the combined Spectrum+BME probe after oscilloscope setup is safe:

```powershell
$env:PYTOF_RUN_HARDWARE_TESTS=1
py -3.12 pytof_new\scripts\probe_spectrum_bme.py --mode raw_multi
py -3.12 pytof_new\scripts\probe_spectrum_bme.py --mode average_32bit --averages 32
py -3.12 pytof_new\scripts\probe_spectrum_bme.py --mode average_16bit --averages 32
py -3.12 pytof_new\scripts\probe_spectrum_bme.py --mode all
```

The combined probe first uses a dedicated one-trigger RAW acquisition to verify that BME connect/configure/arm does not trigger Spectrum. It then activates BME and looks for pickup/ringing events from A/C/F pulse edges at `0`, `5`, `7`, `10`, `12`, and `17 us`. Expected edge times are derived from the configured delays and widths. It saves `.csv`, `.npy`, `.npz`, and `.json` outputs for visual inspection and metadata review.

To DMA averaged records, external trigger pulses must be present:

```powershell
py -3.12 pytof_new\scripts\probe_spectrum.py --average --avg-records 8 --average-acquire
py -3.12 pytof_new\scripts\probe_spectrum.py --average16 --avg-records 8 --average-acquire
```

## Repository Layout

Top-level active project:

```text
pytof_new/
  app.py                         GUI launcher used during development
  pyproject.toml                 package metadata, dependencies, pytest config
  README.md                      this guide
  scripts/
    mock_acquire_batch.py        mock acquisition command-line smoke script
    probe_spectrum.py            direct Spectrum SDK hardware probe
  tests/                         pytest suite
  pytof_new/
    __main__.py                  installed package entry point
    cli.py                       CLI diagnostics entry point
    config/                      immutable configuration and legacy INI support
    acquisition/                 acquisition controllers, Qt workers, data batches
    hardware/                    digitizer/delay abstractions, mock hardware, Spectrum path
    processing/                  signal processing pipeline and algorithms
    storage/                     HDF5, reference, JSON config, and .pytof I/O
    gui/                         PySide6 panels, main window, plotting widgets
    vendor/                      placeholders/import-safe vendor namespaces
```

Legacy/root-level reference files outside `pytof_new/`:

```text
pyTOF-v23.py                     legacy application, used only as reference
pyspcm.py                        Spectrum Python wrapper reference/vendor file
spcm_tools.py                    Spectrum helper reference/vendor file
PyTOF.ini                        legacy operator settings/calibration file
*.dll                            hardware/vendor DLLs
*.pdf                            hardware manuals/datasheets
```

## Application Architecture

The app separates these concerns:

- User intent: acquisition workflow, priority, TOF window, input range, display interval, total shots.
- Hardware batch: exactly one finite Spectrum acquisition request that fits onboard memory.
- Display cadence: how often the GUI redraws accumulated data.

No FIFO acquisition is used. Continuous operation is implemented by repeatedly acquiring finite hardware batches.

## Configuration Models

Edit configuration schema in:

```text
pytof_new/config/models.py
```

Important models:

- `RunConfig`: complete immutable snapshot for one acquisition run.
- `DigitizerConfig`: digitizer/planner settings.
- `BMEConfig`: delay generator timing settings.
- `ProcessingConfig`: processing pipeline settings.
- `StorageConfig`: output file and metadata settings.
- `MockSpectraConfig`: mock signal generation settings.
- `AcquisitionWorkflow`: user-facing acquisition workflow enum.
- `AcquisitionPriority`: Basic-mode planner preference enum.

Defaults also live in:

```text
pytof_new/config/defaults.json
```

Legacy `PyTOF.ini` compatibility lives in:

```text
pytof_new/config/legacy_ini.py
```

If you add a new persistent setting, update `models.py`, relevant GUI panel `config()` methods, tests, and any JSON/INI conversion code.

## GUI Structure

Main GUI orchestration lives in:

```text
pytof_new/gui/main_window.py
```

`MainWindow` owns the panels, connects signals, snapshots config, builds acquisition plans, arms hardware, starts workers, handles saving/loading, and updates plots. It also passes the digitizer TOF window into the BME panel so Basic BME timing stays derived from the current TOF window.

Panel files:

```text
gui/connection_panel.py          simulation/real connection controls
gui/digitizer_panel.py           Basic/Advanced digitizer controls and plan preview
gui/bme_panel.py                 BME timing/settings panel
gui/acquisition_panel.py         Arm/Start/Stop/Clear controls and polarity
gui/processing_panel.py          processing and shot-analysis controls
gui/calibration_panel.py         calibration peak collection/fitting UI
gui/file_panel.py                output path, metadata, saved spectrum controls
gui/mock_spectra_panel.py        mock signal/noise/ringing controls
gui/spectrum_plot.py             pyqtgraph plot widget and spectrum overlays
gui/log_panel.py                 GUI log display
gui/interaction.py               UI interaction helpers/guards
```

If a control is visually wrong, edit the relevant `gui/*_panel.py` file. If control actions are wrong, start in `gui/main_window.py` and follow the connected signal.

`gui/bme_panel.py` has a Basic/Advanced split. Basic mode exposes `Extraction region fill time (us)`, default `55 us`, and derives:

```text
BME repetition period = digitizer TOF window + extraction region fill time
```

Advanced mode exposes the derived settings for manual override: repetition period, digitizer/PUSH/PULL channels, per-output polarity, per-output width, per-output delay, and trigger termination.

## GUI To Hardware Flow

The normal real-hardware flow is:

1. User edits GUI controls.
2. `MainWindow._snapshot_config()` builds an immutable `RunConfig` from panel values.
3. `MainWindow._plan_request()` calls `hardware/acquisition_planner.py` to build an `AcquisitionRunPlan`.
4. Pressing Arm stores `_armed_run_plan` and `_armed_config`.
5. For real hardware, `BMEDelayGeneratorService.configure()` applies BME timing while inactive and `SpectrumAcquisitionService.configure()` sends the plan's `primary_request` to the Spectrum service thread.
6. Pressing Start uses the already armed plan. It does not silently recalculate a different plan.
7. `RealAcquisitionWorker` requests finite batches. In normal real mode it uses `RealBatchCoordinator` to sequence Spectrum and BME. In Spectrum-only test paths it can still call `SpectrumAcquisitionService.acquire()` directly.
8. Returned `SpectrumAcquisitionResult` data is converted to an `AcquisitionBatch` by `hardware/spectrum_converter.py`.
9. `processing/pipeline.py` processes the batch.
10. The worker emits processed display data back to the GUI thread.

The coordinated real batch order is:

1. Stop/deactivate BME.
2. Arm BME for the expected finite trigger count.
3. Prepare Spectrum DMA.
4. Start Spectrum and enable its trigger engine.
5. Start BME output generation.
6. Wait for Spectrum DMA completion.
7. Read BME trigger count and status.
8. Stop/deactivate BME.
9. Verify BME trigger count and Spectrum record count before emitting the result.

The normal mock flow is similar, but uses `AcquisitionWorker`, `MockDigitizer`, and `MockDelayGenerator` instead of the Spectrum service. Mock mode deliberately preserves the development/test behavior and does not require real hardware.

## Threading Model

The GUI thread owns widgets only.

Real Spectrum hardware calls are serialized by:

```text
hardware/spectrum_service.py
```

`SpectrumAcquisitionService` is a `QObject` that lives on a dedicated `QThread`. It owns one persistent `SpectrumDigitizer` handle and exposes slots for:

- connect
- configure
- acquire
- abort
- disconnect
- recover

Real BME calls are serialized by:

```text
hardware/bme_service.py
```

`BMEDelayGeneratorService` is also a `QObject` on its own `QThread`. It owns one `BMEDelayGenerator` session and exposes slots for:

- connect
- configure
- arm
- start
- stop
- emergency stop
- read trigger count
- read status
- disconnect

`acquisition/real_coordinator.py` is the signal-driven state machine that coordinates both services for one finite hardware batch. It is the place to change safe batch ordering or synchronization checks.

Real acquisition workers live in:

```text
acquisition/real_worker.py
```

Mock acquisition workers live in:

```text
acquisition/worker.py
```

Do not call blocking hardware functions from GUI widget code. Route them through the service/worker layer.

## Acquisition Workflows

The user-facing workflows are defined in `config/models.py` and planned in `hardware/acquisition_planner.py`.

### Live Averaged Spectrum

Purpose:

- continuously display and accumulate a high-SNR spectrum;
- use Spectrum FPGA Block Averaging;
- repeatedly acquire finite block-average batches;
- normalize FPGA sums to per-shot averages in Python.

Hardware mode:

- `AVERAGE_32BIT` or `AVERAGE_16BIT`;
- external trigger required;
- software trigger is rejected.

Important batch terms:

- `N`: shots per FPGA sum, stored as `averages_per_segment`.
- `k`: FPGA output records per hardware batch, stored as `number_of_segments`.
- physical triggers per batch: `N * k`.
- transfer records per batch: `k`.

### Live Raw-Shot Spectrum

Purpose:

- continuously acquire individual physical shots;
- preserve raw segments for rejection/inspection;
- update the GUI at the requested cadence while hardware batches may be smaller.

Hardware mode:

- `RAW_MULTI`;
- Basic mode uses external trigger;
- Advanced/testing mode may use software trigger.

### Finite Shot Analysis

Purpose:

- acquire a user-defined total number of raw shots;
- preserve every physical shot independently;
- split the run into one or more finite hardware batches;
- support timing-jitter/cross-correlation/alignment analysis.

Hardware mode:

- always `RAW_MULTI`;
- no hardware averaging;
- final partial batch is supported.

## Acquisition Planner

Planner code lives in:

```text
pytof_new/hardware/acquisition_planner.py
```

The main planner entry point is:

```python
plan_acquisition(...)
```

It returns:

```python
AcquisitionRunPlan
```

The run plan contains:

- workflow and priority;
- primary finite hardware request;
- optional final partial request;
- continuous/finite flag;
- total requested shots;
- physical shots per batch;
- output records per batch;
- full batch count;
- final partial batch size;
- requested and estimated display interval;
- transfer size;
- onboard memory fraction;
- actual post-trigger window;
- warnings and summary text.

Memory rule:

```text
No finite hardware batch may exceed 50% of discovered onboard card memory.
```

The planner uses byte widths:

```text
RAW_MULTI:      1 byte/sample
AVERAGE_16BIT: 2 bytes/sample
AVERAGE_32BIT: 4 bytes/sample
```

Segment size rule:

```text
segment_samples = pretrigger_samples + requested posttrigger samples, rounded up to valid hardware alignment
```

The summary's `Segment size` is total samples including pretrigger. The `Actual recorded post-trigger window` subtracts pretrigger back out.

If you want to change automatic sample-rate choice, edit:

```text
_candidate_rates()
```

If you want to change Basic-mode FPGA averaging count `N`, edit:

```text
_basic_averages_per_segment()
```

If you want to change finite raw batch sizing, edit:

```text
_try_raw_candidate()
```

If you want to change plan-preview wording, edit:

```text
_summary_lines()
```

If you want to change memory limits, edit:

```text
MAX_MEMORY_FRACTION
```

## Spectrum Hardware Path

Real Spectrum hardware integration is split across these files:

```text
hardware/spectrum_models.py      import-safe dataclasses and enums
hardware/spectrum_limits.py      request validation and transfer planning
hardware/spectrum_driver.py      lazy wrapper around pyspcm SDK calls
hardware/spectrum_digitizer.py   persistent card handle, register writes, DMA acquisition
hardware/spectrum_service.py     Qt service that owns the digitizer on a worker thread
hardware/spectrum_converter.py   Spectrum result to AcquisitionBatch conversion
```

`SpectrumAcquisitionRequest` represents exactly one finite hardware acquisition. It includes:

- acquisition mode;
- sample rate;
- segment samples;
- pretrigger samples;
- number of output records;
- FPGA averages per output record;
- trigger source;
- input range;
- trigger level;
- timeout;
- coupling;
- bandwidth limit;
- trigger edge;
- trigger termination.

`SpectrumDigitizer.configure_request()` writes Spectrum registers. It is the main place to update if the hardware needs a different register sequence.

Core register sequence currently mirrors `scripts/probe_spectrum.py`:

```text
SPC_CHENABLE
SPC_AMP0
SPC_CARDMODE
SPC_MEMSIZE
SPC_POSTTRIGGER
SPC_SEGMENTSIZE
SPC_AVERAGES, for block averaging
SPC_CLOCKMODE
SPC_SAMPLERATE
SPC_CLOCKOUT
SPC_TIMEOUT, if available
trigger registers
```

DMA acquisition is in `SpectrumDigitizer.acquire_configured()`.

The split acquisition methods used by the coordinator are:

```text
prepare_configured_acquisition()
start_prepared_acquisition()
wait_for_prepared_result()
```

Do not replace this with FIFO acquisition. Continuous GUI acquisition is repeated finite batches.

16-bit block-average mode symbols are checked in this order:

```text
SPC_REC_STD_AVERAGE_16BIT
SPC_REC_STD_AVERAGE16
SPC_REC_STD_AVERAGE_16
```

## Spectrum Probe Script

Direct hardware probe code lives in:

```text
scripts/probe_spectrum.py
```

Use this when you want to verify raw SDK calls outside the GUI/service abstraction.

It can test:

- card identity and feature flags;
- `RAW_MULTI` with software trigger;
- `AVERAGE_32BIT` config readback;
- `AVERAGE_16BIT` config readback;
- multi-record block-average batches with `--avg-records`;
- optional DMA acquisition with `--average-acquire`, when external trigger pulses are present.

The probe intentionally uses direct `pyspcm` calls and is Python 3.8-compatible style. It is not the app runtime path.

If actual hardware behavior disagrees with the app, first reproduce the successful register sequence in `probe_spectrum.py`, then port the change to `hardware/spectrum_digitizer.py` and update tests.

## BME Hardware Path

Real BME integration is split across these files:

```text
hardware/bme_constants.py          verified constants and export names
hardware/bme_driver.py             lazy ctypes wrapper around DelayGenerator.dll
hardware/bme_delay_generator.py    SG08p session/configure/arm/start/stop logic
hardware/bme_service.py            Qt service that owns BME calls on a worker thread
acquisition/trigger_counts.py      derives expected BME triggers from Spectrum requests
acquisition/real_coordinator.py    safe Spectrum+BME batch sequencing
```

The BME DLL is never loaded at module import time. Real hardware mode loads it only when the BME service connects.

The implementation intentionally binds only flat exported functions with complete signatures from `DG_DLL_1.h`. It does not use structure-based helper functions such as `Set_BME_G08(...)` because that path activates the delay generator internally and would bypass the safe start order.

Current default channel intent:

- channel A: Spectrum trigger output
- channel C: PUSH extraction trigger, positive polarity
- channel F: PULL extraction trigger, negative polarity
- channels B/D/E: disabled

Current Basic timing defaults:

- TOF window: from the digitizer panel
- extraction region fill time: `55 us`
- repetition period: `TOF window + 55 us`
- digitizer/PUSH/PULL delays: `0 us`
- digitizer/PUSH/PULL widths: follow the TOF window
- trigger termination: `50 Ohms`

The Spectrum-trigger output channel must remain enabled with `GoSignal = LocalPrimary`, `OutputModulo = 1`, and `OutputOffset = 0` so one accepted BME trigger produces one Spectrum trigger pulse.

If you change BME timing, channel mapping, trigger mode, or output safety behavior, update `BMEConfig`, `BMEDelayGenerator`, `BMEDelayGeneratorService`, coordinator tests, and hardware oscilloscope procedures together.

## Mock Hardware Path

Mock hardware exists to develop without the Spectrum card or BME hardware.

Important files:

```text
hardware/mock_digitizer.py          synthetic TOF waveforms and ADC conversion
hardware/mock_delay_generator.py    mock delay generator state
acquisition/controller.py           high-level mock acquisition controller
acquisition/worker.py               Qt worker used by the GUI in Simulation mode
scripts/mock_acquire_batch.py       simple command-line mock acquisition
```

If you want to change synthetic peak shapes, noise, ringing, timing jitter, saturation, or dropped trigger behavior, edit:

```text
hardware/mock_digitizer.py
gui/mock_spectra_panel.py
config/models.py, MockSpectraConfig
```

## Processing Pipeline

Processing entry point:

```text
processing/pipeline.py
```

Supporting modules:

```text
processing/baseline.py       baseline estimation/subtraction
processing/filtering.py      Butterworth filters and smoothing
processing/reference.py      reference spectrum load/subtract
processing/quality.py        shot quality and rejection
processing/averaging.py      averaging helpers
processing/conversion.py     ADC voltage, TOF axis, mass conversion
processing/peaks.py          peak fitting
processing/calibration.py    calibration fitting/inversion
processing/jitter.py         cross-correlation jitter and time-aligned average
```

If a displayed spectrum looks wrong, start with `processing/pipeline.py`, then follow the helper module for the specific processing stage.

## Storage And File Formats

Storage code lives in:

```text
storage/hdf5_writer.py       HDF5 run and reference files
storage/pytof_writer.py      legacy text .pytof export
storage/pytof_reader.py      legacy text .pytof import
storage/config_io.py         JSON config save/load helpers
storage/metadata.py          timestamps and metadata helpers
```

Current GUI behavior saves cumulative spectra as legacy `.pytof` text files. Reference spectra are saved as HDF5 files. The HDF5 run writer is used by mock scripts and tests.

If you change metadata fields, update:

```text
config/models.py, StorageConfig
gui/file_panel.py
storage/pytof_writer.py
storage/hdf5_writer.py
tests/test_storage.py
```

## Where To Edit Common Behaviors

Change Basic digitizer controls or layout:

```text
gui/digitizer_panel.py
```

Change how Arm/Start/Stop behave:

```text
gui/main_window.py
acquisition/worker.py
acquisition/real_worker.py
```

Change acquisition planning policy:

```text
hardware/acquisition_planner.py
tests/test_acquisition_planner.py
```

Change Spectrum register writes:

```text
hardware/spectrum_digitizer.py
tests/test_spectrum_digitizer.py
scripts/probe_spectrum.py, if validating on real hardware first
```

Change Spectrum request validation or hardware limits:

```text
hardware/spectrum_limits.py
tests/test_spectrum_limits.py
```

Change real hardware service/thread behavior:

```text
hardware/spectrum_service.py
hardware/bme_service.py
acquisition/real_coordinator.py
acquisition/real_worker.py
tests/test_spectrum_service.py
tests/test_bme_service.py
tests/test_real_coordinator.py
tests/test_real_workers.py
```

Change BME DLL bindings or SG08p configuration:

```text
hardware/bme_driver.py
hardware/bme_constants.py
hardware/bme_delay_generator.py
tests/test_bme_driver.py
tests/test_bme_constants.py
tests/test_bme_delay_generator.py
```

Change mock acquisition behavior:

```text
hardware/mock_digitizer.py
hardware/mock_delay_generator.py
acquisition/controller.py
acquisition/worker.py
tests/test_mock_acquisition.py
```

Change processing results:

```text
processing/pipeline.py
processing/*.py
tests/test_processing.py
```

Change plotting behavior:

```text
gui/spectrum_plot.py
tests/test_gui_smoke.py
```

Change saved file format:

```text
storage/pytof_writer.py
storage/pytof_reader.py
storage/hdf5_writer.py
tests/test_storage.py
```

Change app startup:

```text
app.py
pytof_new/__main__.py
pytof_new/cli.py
```

## Testing Strategy

Run the full suite before and after behavior changes:

```powershell
py -3.12 -m pytest pytof_new\tests
```

Useful targeted tests:

```powershell
py -3.12 -m pytest pytof_new\tests\test_acquisition_planner.py
py -3.12 -m pytest pytof_new\tests\test_gui_smoke.py
py -3.12 -m pytest pytof_new\tests\test_spectrum_digitizer.py
py -3.12 -m pytest pytof_new\tests\test_spectrum_service.py
py -3.12 -m pytest pytof_new\tests\test_bme_service.py
py -3.12 -m pytest pytof_new\tests\test_real_coordinator.py
py -3.12 -m pytest pytof_new\tests\test_real_workers.py
py -3.12 -m pytest pytof_new\tests\test_processing.py
py -3.12 -m pytest pytof_new\tests\test_storage.py
```

Known warning:

- `tests/test_real_workers.py::test_shot_analysis_acquires_n_records` may emit RuntimeWarnings from fake all-constant traces during jitter analysis. The test still passes and the warning is caused by synthetic fake data, not a hardware path failure.

## Hardware Safety Notes

- The planner limits each finite Spectrum hardware batch to at most 50% of onboard memory.
- The planner never silently shortens the requested post-trigger TOF window.
- Basic block-average modes require external trigger.
- Raw software trigger is allowed only for Advanced/testing workflows.
- Finite Shot Analysis always preserves individual raw segments.
- Continuous acquisition is repeated finite batches, not FIFO.
- Real hardware mode requires both Spectrum and BME services. If either one cannot connect, real mode fails instead of falling back to a partial hardware path.
- Normal real acquisition starts BME only after Spectrum is prepared and armed.
- BME configuration and arming must not generate physical pulses; physical output generation starts only at explicit BME activation.
- Verify BME TTL amplitude, Spectrum Ext0 termination/range, and cabling with an oscilloscope before connecting production hardware.

## First Hardware Run Checklist

Before connecting BME outputs to the Spectrum trigger input or extraction electronics:

1. Confirm the Spectrum card is detected with `probe_spectrum.py --info`.
2. Confirm safe BME discovery with `py -3.12 -m pytof_new.cli diagnose-bme`.
3. Leave BME outputs disconnected from production hardware and connected only to an oscilloscope/known-safe load.
4. Confirm BME connect/configure/arm do not emit pulses.
5. Run one gated A-only BME pulse test with `PYTOF_RUN_HARDWARE_TESTS=1` and `probe_bme.py --pulse-test --enabled-channels A --pulse-count 1`.
6. Add C and F with `--enabled-channels A,C,F` only after A is verified.
7. Verify channel A is the Spectrum trigger pulse, channel C is PUSH, and channel F is PULL.
8. Verify default polarities: A POS, C POS, F NEG.
9. Verify all delays default to `0 us` and widths follow the TOF window in Basic mode.
10. Verify repetition period equals `TOF window + extraction region fill time` in Basic mode.
11. Verify pulse amplitude and termination at the selected load; avoid double termination.
12. Connect BME channel A to Spectrum Ext0 only after the trigger pulse is verified safe for the Spectrum input range/termination.
13. Run `probe_spectrum_bme.py --mode raw_multi` and inspect the saved trace for the six expected pickup events.
14. Run `probe_spectrum_bme.py --mode average_32bit` and `--mode average_16bit` when 16-bit averaging is supported.
15. Run a one-record coordinated GUI acquisition and verify Spectrum record count equals BME accepted trigger count.

Troubleshooting quick checks:

- BME count too low: BME stopped early, preset count misconfigured, or internal trigger sequence interrupted.
- BME count correct but Spectrum timeout: check cabling, Ext0 threshold, pulse width, termination, and trigger edge.
- Spectrum completes but BME count differs: check counter reset/preset bookkeeping and coordinator metadata.
- Incorrect PUSH/PULL polarity: verify advanced BME polarity settings and actual SG08p output behavior on the oscilloscope.

## Recommended Workflow For Hardware Changes

1. Reproduce Spectrum register behavior in `scripts/probe_spectrum.py` when changing Spectrum configuration.
2. Confirm BME configuration behavior with an oscilloscope before connecting BME outputs to other hardware.
3. Confirm BME configure/arm calls do not emit pulses.
4. Confirm BME activation emits the expected channel pulses and preset counts stop output.
5. Port confirmed Spectrum behavior into `hardware/spectrum_digitizer.py`.
6. Port confirmed BME behavior into `hardware/bme_delay_generator.py`.
7. Update request validation, trigger-count derivation, and coordinator logic if needed.
8. Add or update fake-driver/service/coordinator tests.
9. Run the full test suite.
10. Test through the GUI in Simulation mode first.
11. Test through the GUI in real mode with Spectrum and BME connected, monitored, and safely terminated.
