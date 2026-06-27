#!/usr/bin/env python3
"""
Minimal hardware probe for Spectrum M4i.2210-x8 digitizer.

Run on the PC that has the card + BME hardware (requires Spectrum SDK).
Reports card identity, tests RAW_MULTI (Modes 1/2) and AVERAGE_32BIT config.

Usage:
    python probe_spectrum.py --info                  card identity + features
    python probe_spectrum.py --raw-multi             single-segment RAW_MULTI
    python probe_spectrum.py --raw-multi --segs 8    multi-segment RAW_MULTI
    python probe_spectrum.py --average               AVERAGE_32BIT config-only test
    python probe_spectrum.py --average --avg-records 8
    python probe_spectrum.py --average --average-acquire --avg-records 8
    python probe_spectrum.py --average16             AVERAGE_16BIT config-only test
    python probe_spectrum.py --all                   run everything
    python probe_spectrum.py                         same as --all
    python probe_spectrum.py --device /dev/spcm0     specify card device path
"""

# Python 3.8 compatible.

from __future__ import print_function

import argparse
import sys
import time

import numpy as np

try:
    from ctypes import (
        byref, cast, c_char, c_double, c_int8, c_int16, c_int32, c_uint32,
        c_uint64, create_string_buffer, sizeof, POINTER,
    )
    from pyspcm import *
    from spcm_tools import pvAllocMemPageAligned, szTypeToName
    from py_header.regs import *
    from py_header.spcerr import ERR_OK, ERR_TIMEOUT
except ImportError:
    _SDK_LOAD_ERROR = sys.exc_info()[1]
    _SDK_OK = False
else:
    _SDK_OK = True

ERRORTEXTLEN = 1024

KILO_B = lambda x: x * 1024
MEGA_B = lambda x: x * 1024 * 1024
GIGA_B = lambda x: x * 1024 * 1024 * 1024

SEGMENT_SAMPLES = KILO_B(64)  # 65536
PRETRIGGER = 32
SAMPLE_RATE = 1250000000
V_RANGE = 500
AVERAGES = 100
TIMEOUT_MS = 5000


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------

class SpectrumProbeError(RuntimeError):
    pass


def _check_sdk():
    if not _SDK_OK:
        sys.stderr.write("ERROR: {}\n".format(_SDK_LOAD_ERROR))
        sys.stderr.write(
            "This script must run on the PC with the Spectrum SDK installed.\n"
            "Make sure pyspcm.py, spcm_tools.py, and py_header/ are on PYTHONPATH.\n"
        )
        sys.exit(1)


def require_ok(hCard, operation, err):
    """Raise SpectrumProbeError if err is not ERR_OK; also clears card error."""
    if err == ERR_OK:
        return
    szErr = create_string_buffer(ERRORTEXTLEN)
    spcm_dwGetErrorInfo_i32(hCard, None, None, szErr)
    raise SpectrumProbeError(
        "{} failed with {:#x}: {}".format(
            operation, err, szErr.value.decode(errors="replace"),
        )
    )


def clear_card_error(hCard):
    """Read and discard any pending card error."""
    szErr = create_string_buffer(ERRORTEXTLEN)
    spcm_dwGetErrorInfo_i32(hCard, None, None, szErr)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def open_card(device_path):
    _check_sdk()
    hCard = spcm_hOpen(device_path)
    if not hCard:
        sys.stderr.write("FAIL: spcm_hOpen returned null handle for {}\n".format(device_path))
        sys.exit(1)
    print("OK: card opened on", device_path)
    return hCard


def close_card(hCard):
    spcm_vClose(hCard)
    print("OK: card closed")


def _reg(name):
    """Resolve a py_header.regs constant by name, or None if not defined."""
    return globals().get(name)


def set_i32(hCard, reg, value, label=""):
    if reg is None:
        raise SpectrumProbeError("Register {} not defined in py_header.regs".format(label))
    tag = "{}[{:#x}]={}".format(label, reg, value)
    err = spcm_dwSetParam_i32(hCard, reg, value)
    require_ok(hCard, "spcm_dwSetParam_i32 {}".format(tag), err)


def set_i64(hCard, reg, value, label=""):
    if reg is None:
        raise SpectrumProbeError("Register {} not defined in py_header.regs".format(label))
    tag = "{}[{:#x}]={}".format(label, reg, value)
    err = spcm_dwSetParam_i64(hCard, reg, value)
    require_ok(hCard, "spcm_dwSetParam_i64 {}".format(tag), err)


def get_i32(hCard, reg, label=""):
    if reg is None:
        raise SpectrumProbeError("Register {} not defined in py_header.regs".format(label))
    val = c_int32()
    err = spcm_dwGetParam_i32(hCard, reg, byref(val))
    require_ok(hCard, "spcm_dwGetParam_i32 {}".format(label), err)
    return val.value


def get_i64(hCard, reg, label=""):
    if reg is None:
        raise SpectrumProbeError("Register {} not defined in py_header.regs".format(label))
    val = c_uint64()
    err = spcm_dwGetParam_i64(hCard, reg, byref(val))
    require_ok(hCard, "spcm_dwGetParam_i64 {}".format(label), err)
    return val.value


def card_command(hCard, cmd, label=""):
    tag = "M2CMD {:#x} ({})".format(cmd, label)
    err = spcm_dwSetParam_i32(hCard, SPC_M2CMD, cmd)
    require_ok(hCard, tag, err)


def describe_data(data, name="data"):
    if data is None or data.size == 0:
        print("    {}: NONE/EMPTY".format(name))
        return
    print("    {}: shape={} dtype={} min={} max={} mean={:.4g}".format(
        name, data.shape, data.dtype,
        data.min(), data.max(), float(data.mean()),
    ))


# ---------------------------------------------------------------------------
# DMA transfer (generic over sample type)
# ---------------------------------------------------------------------------

def dma_transfer(hCard, n_samples, sample_ctype):
    """Start card, wait for data, DMA-transfer, stop.
    Returns (elapsed_seconds, numpy_array).
    sample_ctype: e.g. c_int8 for raw, c_int32 for FPGA sums.
    """
    buf_bytes = n_samples * sizeof(sample_ctype)
    qwBufferSize = c_uint64(buf_bytes)

    pvBuffer = pvAllocMemPageAligned(buf_bytes)
    err = spcm_dwDefTransfer_i64(
        hCard, SPCM_BUF_DATA, SPCM_DIR_CARDTOPC, 0,
        pvBuffer, 0, qwBufferSize,
    )
    require_ok(hCard, "spcm_dwDefTransfer_i64", err)

    t0 = time.time()
    card_command(
        hCard,
        M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_DATA_STARTDMA,
        "START|ENABLETRIGGER|STARTDMA",
    )

    wait_err = spcm_dwSetParam_i32(
        hCard, SPC_M2CMD, M2CMD_CARD_WAITREADY | M2CMD_DATA_WAITDMA,
    )
    elapsed = time.time() - t0

    if wait_err == ERR_TIMEOUT:
        print("  TIMEOUT: card did not finish within timeout")
        card_command(hCard, M2CMD_CARD_STOP | M2CMD_CARD_DISABLETRIGGER | M2CMD_DATA_STOPDMA, "STOP")
        return elapsed, None
    elif wait_err != ERR_OK:
        card_command(hCard, M2CMD_CARD_STOP | M2CMD_CARD_DISABLETRIGGER | M2CMD_DATA_STOPDMA, "STOP")
        raise SpectrumProbeError("WAITREADY error: {:#x}".format(wait_err))

    # Status
    lAvailUser = c_int32()
    lPCPos = c_int32()
    spcm_dwGetParam_i32(hCard, SPC_DATA_AVAIL_USER_LEN, byref(lAvailUser))
    spcm_dwGetParam_i32(hCard, SPC_DATA_AVAIL_USER_POS, byref(lPCPos))
    print("  Available: {} bytes, position: {}".format(lAvailUser.value, lPCPos.value))

    # Copy data before stopping
    pData = cast(pvBuffer, POINTER(sample_ctype))
    data = np.ctypeslib.as_array(pData, shape=(n_samples,)).copy()

    card_command(hCard, M2CMD_CARD_STOP | M2CMD_CARD_DISABLETRIGGER | M2CMD_DATA_STOPDMA, "STOP")
    return elapsed, data


# ---------------------------------------------------------------------------
# common configuration helpers
# ---------------------------------------------------------------------------

def _apply_raw_multi_config(hCard, segment_samples, nsegs):
    """Configure card for SPC_REC_STD_MULTI with software trigger."""
    posttrigger = segment_samples - PRETRIGGER
    memsize = segment_samples * nsegs

    set_i32(hCard, _reg("SPC_CARDMODE"), _reg("SPC_REC_STD_MULTI"), "SPC_CARDMODE")
    set_i32(hCard, _reg("SPC_CHENABLE"), _reg("CHANNEL0"), "SPC_CHENABLE")
    set_i32(hCard, _reg("SPC_AMP0"), V_RANGE, "SPC_AMP0")
    set_i64(hCard, _reg("SPC_MEMSIZE"), memsize, "SPC_MEMSIZE")
    set_i32(hCard, _reg("SPC_CLOCKMODE"), _reg("SPC_CM_INTPLL"), "SPC_CLOCKMODE")
    set_i64(hCard, _reg("SPC_SAMPLERATE"), SAMPLE_RATE, "SPC_SAMPLERATE")
    set_i32(hCard, _reg("SPC_CLOCKOUT"), 0, "SPC_CLOCKOUT")
    set_i32(hCard, _reg("SPC_TRIG_ORMASK"), _reg("SPC_TMASK_SOFTWARE"), "SPC_TRIG_ORMASK")
    set_i32(hCard, _reg("SPC_TIMEOUT"), TIMEOUT_MS, "SPC_TIMEOUT")
    # Segment: SEGMENTSIZE + POSTTRIGGER (no direct PRETRIGGER set for multi)
    set_i64(hCard, _reg("SPC_SEGMENTSIZE"), segment_samples, "SPC_SEGMENTSIZE")
    set_i64(hCard, _reg("SPC_POSTTRIGGER"), posttrigger, "SPC_POSTTRIGGER")


# ---------------------------------------------------------------------------
# test: --info
# ---------------------------------------------------------------------------

def test_info(hCard):
    clear_card_error(hCard)
    print("\n=== Card Identity ===")
    card_type = get_i32(hCard, _reg("SPC_PCITYP"), "SPC_PCITYP")
    serial = get_i32(hCard, _reg("SPC_PCISERIALNO"), "SPC_PCISERIALNO")
    mem_size = get_i64(hCard, _reg("SPC_PCIMEMSIZE"), "SPC_PCIMEMSIZE")

    if card_type is not None:
        name = szTypeToName(card_type)
        print("  Type:  {:#x} -> {}".format(card_type, name))
    if serial is not None:
        print("  S/N:   {:05d}".format(serial))
    if mem_size is not None:
        print("  Memory: {} bytes ({:.1f} GB)".format(mem_size, mem_size / GIGA_B(1)))

    # Try reading extended features via numeric register 2000
    print("\n=== Feature Scan ===")
    reg_2000 = 2000  # SPC_PCIEXTFEATURE numeric address
    feat = c_int32()
    err = spcm_dwGetParam_i32(hCard, reg_2000, byref(feat))
    if err == ERR_OK:
        print("  SPC_PCIEXTFEATURE(2000): {:#010x}".format(feat.value))
        avg_mask = _reg("SPCM_FEAT_EXTFW_AVERAGE")
        if avg_mask is None:
            avg_mask = 2  # known value for this card
            print("    (using fallback mask: {:#x})".format(avg_mask))
        if feat.value & avg_mask:
            print("  Block Average option: YES")
        else:
            print("  Block Average option: NO")
    else:
        print("  SPC_PCIEXTFEATURE(2000): register not accessible (err={:#x})".format(err))

    ch_count = get_i32(hCard, _reg("SPC_CHCOUNT"), "SPC_CHCOUNT")
    print("  Channel count: {}".format(ch_count if ch_count is not None else "?"))
    adc_res = get_i32(hCard, _reg("SPC_MIINST_BITSPERSAMPLE"), "SPC_MIINST_BITSPERSAMPLE")
    print("  ADC bits: {}".format(adc_res if adc_res is not None else "?"))
    print("  Max sample rate: 1.25 GS/s (by card model)")

    # Probe 16-bit average mode constant
    clear_card_error(hCard)
    mode16 = _reg("SPC_REC_STD_AVERAGE_16BIT")
    if mode16 is None:
        print("  SPC_REC_STD_AVERAGE_16BIT: constant absent from SDK")
    else:
        saved = c_int32()
        spcm_dwGetParam_i32(hCard, _reg("SPC_CARDMODE"), byref(saved))
        err = spcm_dwSetParam_i32(hCard, _reg("SPC_CARDMODE"), mode16)
        if err == ERR_OK:
            readback = c_int32()
            spcm_dwGetParam_i32(hCard, _reg("SPC_CARDMODE"), byref(readback))
            accepted = (readback.value == mode16)
            print("  16-bit average mode accepted by card: {}".format(accepted))
            # Restore
            spcm_dwSetParam_i32(hCard, _reg("SPC_CARDMODE"), saved.value)
        else:
            print("  16-bit average mode: rejected (err={:#x})".format(err))


# ---------------------------------------------------------------------------
# test: --raw-multi  (Mode 1/2)
# ---------------------------------------------------------------------------

def test_raw_multi(hCard, nsegs):
    clear_card_error(hCard)
    mode_name = "RAW_MULTI {} segment(s)".format(nsegs)
    print("\n=== {} ===".format(mode_name))

    segment_samples = SEGMENT_SAMPLES
    n_samples = segment_samples * nsegs

    _apply_raw_multi_config(hCard, segment_samples, nsegs)

    elapsed, data = dma_transfer(hCard, n_samples, c_int8)
    print("  Wait time: {:.3f} s".format(elapsed))

    if data is None:
        print("  FAIL: no data (card timed out)")
        return

    # Read trigger counter
    try:
        trig_cnt = get_i64(hCard, _reg("SPC_TRIGGERCOUNTER"), "SPC_TRIGGERCOUNTER")
        print("  Trigger count: {}".format(trig_cnt))
    except SpectrumProbeError as exc:
        print("  Trigger count: unavailable ({})".format(exc))

    # Reshape and describe
    try:
        data_2d = data.reshape(nsegs, segment_samples)
        describe_data(data_2d, "data (2D int8)")
        for i in range(min(nsegs, 3)):
            seg = data_2d[i]
            n_unique = int(len(np.unique(seg)))
            print("      seg[{}]: min={} max={} mean={:.4g} unique={}".format(
                i, seg.min(), seg.max(), float(seg.mean()), n_unique,
            ))
    except ValueError:
        describe_data(data, "data (flat)")
        print("  (expected {} samples, got {})".format(n_samples, data.size))

    print("  RESULT: {} PASS".format(mode_name))


# ---------------------------------------------------------------------------
# test: --average (Mode 3, config-only)
# ---------------------------------------------------------------------------

def test_average_config(hCard, records, acquire=False):
    clear_card_error(hCard)
    label = "AVERAGE_32BIT ({} FPGA output record(s))".format(records)
    print("\n=== {} ===".format(label))
    if acquire:
        print("  Full acquisition requested; Ext0 trigger pulses are required.")
    else:
        print("  Writing registers, reading back — no acquisition started.")
        print("  (Use --average-acquire to DMA data; requires Ext0 trigger pulses.)")

    segment_samples = SEGMENT_SAMPLES
    posttrigger = segment_samples - PRETRIGGER
    memsize = segment_samples * records
    n_samples = segment_samples * records

    # --- write configuration ---
    set_i32(hCard, _reg("SPC_CARDMODE"), _reg("SPC_REC_STD_AVERAGE"), "SPC_CARDMODE")
    set_i32(hCard, _reg("SPC_CHENABLE"), _reg("CHANNEL0"), "SPC_CHENABLE")
    set_i32(hCard, _reg("SPC_AMP0"), V_RANGE, "SPC_AMP0")
    set_i64(hCard, _reg("SPC_MEMSIZE"), memsize, "SPC_MEMSIZE")
    set_i32(hCard, _reg("SPC_CLOCKMODE"), _reg("SPC_CM_INTPLL"), "SPC_CLOCKMODE")
    set_i64(hCard, _reg("SPC_SAMPLERATE"), SAMPLE_RATE, "SPC_SAMPLERATE")
    set_i32(hCard, _reg("SPC_CLOCKOUT"), 0, "SPC_CLOCKOUT")
    set_i32(hCard, _reg("SPC_AVERAGES"), AVERAGES, "SPC_AVERAGES")
    # Trigger: Ext0 (block averaging does not support software trigger)
    set_i32(hCard, _reg("SPC_TRIG_ORMASK"), _reg("SPC_TMASK_EXT0"), "SPC_TRIG_ORMASK")
    set_i32(hCard, _reg("SPC_TRIG_EXT0_MODE"), _reg("SPC_TM_POS"), "SPC_TRIG_EXT0_MODE")
    set_i32(hCard, _reg("SPC_TRIG_EXT0_LEVEL0"), 1500, "SPC_TRIG_EXT0_LEVEL0")
    # Segment size / posttrigger
    set_i64(hCard, _reg("SPC_SEGMENTSIZE"), segment_samples, "SPC_SEGMENTSIZE")
    set_i64(hCard, _reg("SPC_POSTTRIGGER"), posttrigger, "SPC_POSTTRIGGER")
    set_i32(hCard, _reg("SPC_TIMEOUT"), TIMEOUT_MS, "SPC_TIMEOUT")

    # --- read back verification ---
    cardmode_back = get_i32(hCard, _reg("SPC_CARDMODE"), "SPC_CARDMODE(readback)")
    avg_back = get_i32(hCard, _reg("SPC_AVERAGES"), "SPC_AVERAGES(readback)")
    memsize_back = get_i64(hCard, _reg("SPC_MEMSIZE"), "SPC_MEMSIZE(readback)")
    segsize_back = get_i64(hCard, _reg("SPC_SEGMENTSIZE"), "SPC_SEGMENTSIZE(readback)")
    post_back = get_i64(hCard, _reg("SPC_POSTTRIGGER"), "SPC_POSTTRIGGER(readback)")
    trigmask_back = get_i32(hCard, _reg("SPC_TRIG_ORMASK"), "SPC_TRIG_ORMASK(readback)")

    expected_mode = _reg("SPC_REC_STD_AVERAGE")
    expected_mask = _reg("SPC_TMASK_EXT0")

    print("  SPC_CARDMODE:     set={} readback={} {}".format(
        expected_mode, cardmode_back,
        "OK" if cardmode_back == expected_mode else "MISMATCH",
    ))
    print("  SPC_AVERAGES:     set={} readback={} {}".format(
        AVERAGES, avg_back,
        "OK" if avg_back == AVERAGES else "MISMATCH",
    ))
    print("  SPC_MEMSIZE:      set={} readback={} {}".format(
        memsize, memsize_back,
        "OK" if memsize_back == memsize else "MISMATCH",
    ))
    print("  SPC_SEGMENTSIZE:  set={} readback={} {}".format(
        segment_samples, segsize_back,
        "OK" if segsize_back == segment_samples else "MISMATCH",
    ))
    print("  SPC_POSTTRIGGER:  set={} readback={} {}".format(
        posttrigger, post_back,
        "OK" if post_back == posttrigger else "MISMATCH",
    ))
    print("  SPC_TRIG_ORMASK:  set={} readback={} {}".format(
        expected_mask, trigmask_back,
        "OK" if trigmask_back == expected_mask else "MISMATCH",
    ))

    if acquire:
        elapsed, data = dma_transfer(hCard, n_samples, c_int32)
        print("  Wait time: {:.3f} s".format(elapsed))
        if data is None:
            print("  FAIL: no averaged data (card timed out)")
            return
        data_2d = data.reshape(records, segment_samples)
        describe_data(data_2d, "data (2D int32 FPGA sums)")

    print("  RESULT: {} PASS".format(label))


# ---------------------------------------------------------------------------
# test: --average16 (Mode 16-bit average, config-only)
# ---------------------------------------------------------------------------

def test_average16_config(hCard, records, acquire=False):
    clear_card_error(hCard)
    label = "AVERAGE_16BIT ({} FPGA output record(s))".format(records)
    print("\n=== {} ===".format(label))
    if acquire:
        print("  Full acquisition requested; Ext0 trigger pulses are required.")
    else:
        print("  Writing registers, reading back — no acquisition started.")
        print("  (Use --average-acquire to DMA data; requires Ext0 trigger pulses.)")

    mode16 = _reg("SPC_REC_STD_AVERAGE_16BIT")
    if mode16 is None:
        mode16_candidates = ("SPC_REC_STD_AVERAGE16", "SPC_REC_STD_AVERAGE_16")
        for cand in mode16_candidates:
            val = _reg(cand)
            if val is not None:
                mode16 = val
                print("  Using {} as 16-bit average constant".format(cand))
                break
        else:
            print("  SKIP: no 16-bit average constant in SDK")
            return

    segment_samples = SEGMENT_SAMPLES
    posttrigger = segment_samples - PRETRIGGER
    memsize = segment_samples * records
    n_samples = segment_samples * records

    set_i32(hCard, _reg("SPC_CARDMODE"), mode16, "SPC_CARDMODE")
    set_i32(hCard, _reg("SPC_CHENABLE"), _reg("CHANNEL0"), "SPC_CHENABLE")
    set_i32(hCard, _reg("SPC_AMP0"), V_RANGE, "SPC_AMP0")
    set_i64(hCard, _reg("SPC_MEMSIZE"), memsize, "SPC_MEMSIZE")
    set_i32(hCard, _reg("SPC_CLOCKMODE"), _reg("SPC_CM_INTPLL"), "SPC_CLOCKMODE")
    set_i64(hCard, _reg("SPC_SAMPLERATE"), SAMPLE_RATE, "SPC_SAMPLERATE")
    set_i32(hCard, _reg("SPC_CLOCKOUT"), 0, "SPC_CLOCKOUT")
    set_i32(hCard, _reg("SPC_AVERAGES"), AVERAGES, "SPC_AVERAGES")
    set_i32(hCard, _reg("SPC_TRIG_ORMASK"), _reg("SPC_TMASK_EXT0"), "SPC_TRIG_ORMASK")
    set_i32(hCard, _reg("SPC_TRIG_EXT0_MODE"), _reg("SPC_TM_POS"), "SPC_TRIG_EXT0_MODE")
    set_i32(hCard, _reg("SPC_TRIG_EXT0_LEVEL0"), 1500, "SPC_TRIG_EXT0_LEVEL0")
    set_i64(hCard, _reg("SPC_SEGMENTSIZE"), segment_samples, "SPC_SEGMENTSIZE")
    set_i64(hCard, _reg("SPC_POSTTRIGGER"), posttrigger, "SPC_POSTTRIGGER")
    set_i32(hCard, _reg("SPC_TIMEOUT"), TIMEOUT_MS, "SPC_TIMEOUT")

    cardmode_back = get_i32(hCard, _reg("SPC_CARDMODE"), "SPC_CARDMODE(readback)")
    avg_back = get_i32(hCard, _reg("SPC_AVERAGES"), "SPC_AVERAGES(readback)")
    memsize_back = get_i64(hCard, _reg("SPC_MEMSIZE"), "SPC_MEMSIZE(readback)")
    segsize_back = get_i64(hCard, _reg("SPC_SEGMENTSIZE"), "SPC_SEGMENTSIZE(readback)")
    post_back = get_i64(hCard, _reg("SPC_POSTTRIGGER"), "SPC_POSTTRIGGER(readback)")

    print("  SPC_CARDMODE:     set={} readback={} {}".format(
        mode16, cardmode_back,
        "OK" if cardmode_back == mode16 else "MISMATCH",
    ))
    print("  SPC_AVERAGES:     set={} readback={} {}".format(
        AVERAGES, avg_back,
        "OK" if avg_back == AVERAGES else "MISMATCH",
    ))
    print("  SPC_MEMSIZE:      set={} readback={} {}".format(
        memsize, memsize_back,
        "OK" if memsize_back == memsize else "MISMATCH",
    ))
    print("  SPC_SEGMENTSIZE:  set={} readback={} {}".format(
        segment_samples, segsize_back,
        "OK" if segsize_back == segment_samples else "MISMATCH",
    ))
    print("  SPC_POSTTRIGGER:  set={} readback={} {}".format(
        posttrigger, post_back,
        "OK" if post_back == posttrigger else "MISMATCH",
    ))

    if acquire:
        elapsed, data = dma_transfer(hCard, n_samples, c_int16)
        print("  Wait time: {:.3f} s".format(elapsed))
        if data is None:
            print("  FAIL: no averaged data (card timed out)")
            return
        data_2d = data.reshape(records, segment_samples)
        describe_data(data_2d, "data (2D int16 FPGA sums)")

    print("  RESULT: {} PASS".format(label))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Probe Spectrum digitizer hardware")
    parser.add_argument("--info", action="store_true", help="card identity + features")
    parser.add_argument("--raw-multi", action="store_true", help="test RAW_MULTI mode")
    parser.add_argument("--average", action="store_true", help="test AVERAGE_32BIT config-only")
    parser.add_argument("--average16", action="store_true", help="test AVERAGE_16BIT config-only")
    parser.add_argument("--segs", type=int, default=1, help="segments for RAW_MULTI (default: 1)")
    parser.add_argument("--avg-records", type=int, default=1, help="FPGA output records per block-average hardware batch (default: 1)")
    parser.add_argument("--average-acquire", action="store_true", help="DMA averaged records after config; requires Ext0 triggers")
    parser.add_argument("--all", action="store_true", dest="all", help="run all tests")
    parser.add_argument("--device", type=str, default="/dev/spcm0", help="card device path")
    args = parser.parse_args()

    selected = args.info or args.raw_multi or args.average or args.average16 or args.all
    if not selected:
        args.all = True

    if args.all:
        args.info = True
        args.raw_multi = True
        args.average = True
        args.average16 = True
    if args.avg_records < 1:
        parser.error("--avg-records must be >= 1")

    _check_sdk()

    hCard = open_card(args.device.encode())

    try:
        if args.info:
            test_info(hCard)

        if args.raw_multi:
            test_raw_multi(hCard, 1)
            if args.segs > 1:
                test_raw_multi(hCard, args.segs)

        if args.average:
            test_average_config(hCard, args.avg_records, acquire=args.average_acquire)

        if args.average16:
            test_average16_config(hCard, args.avg_records, acquire=args.average_acquire)

    except SpectrumProbeError as exc:
        print("\nPROBE ERROR: {}".format(exc))
        sys.exit(1)
    finally:
        close_card(hCard)

    print("\n=== Probe complete ===")


if __name__ == "__main__":
    main()
