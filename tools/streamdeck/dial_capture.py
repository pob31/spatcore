#!/usr/bin/env python3
"""Capture what an Elgato Stream Deck+ really reports when its dials turn.

Read-only: opens the deck for reading through Windows' own HID calls (ctypes on
SetupAPI and hid.dll, no packages to install) and never writes to it, so it
runs beside an app that has the deck open - WFS-DIY, or Elgato's own app - and
shows the raw clicks next to what the app did with them.

It waits for the first turn, records until the dials have been still for a
while, then prints each dial's RUNS (reports no more than 120 ms apart): the
signed clicks of every report, the total, and a flag on any run that holds both
directions.

What it showed on 2026-09-24 (freshly updated firmware): one report every 50 ms
while a dial turns, each holding the signed clicks of that window - 1 to 3 on a
slow or medium turn, up to 16 on a flick - and never a reversal inside a flick.
The manager's click counting and acceleration (StreamDeckDialAcceleration.h)
rest on that; run this again after a firmware update to check it still holds.

Usage (Windows):
    python dial_capture.py                 # runs per dial
    python dial_capture.py --reports       # plus every report, with its gap
    python dial_capture.py --idle 4 --wait 60
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import statistics
import sys
import threading
import time

VENDOR_ID, PRODUCT_ID_PLUS = 0x0FD9, 0x0084
RUN_GAP_S = 0.12      # a gap longer than this (two silent windows) ends a run


class GUID(ctypes.Structure):
    _fields_ = [('Data1', ctypes.c_uint32), ('Data2', ctypes.c_uint16),
                ('Data3', ctypes.c_uint16), ('Data4', ctypes.c_ubyte * 8)]


class SP_DEVICE_INTERFACE_DATA(ctypes.Structure):
    _fields_ = [('cbSize', ctypes.c_uint32), ('InterfaceClassGuid', GUID),
                ('Flags', ctypes.c_uint32), ('Reserved', ctypes.c_size_t)]


def open_deck():
    """Return (handle, report length) of the first Stream Deck+ found, or exit."""
    setupapi = ctypes.WinDLL('setupapi', use_last_error=True)
    hiddll = ctypes.WinDLL('hid', use_last_error=True)
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)

    hiddll.HidD_GetHidGuid.argtypes = [ctypes.POINTER(GUID)]
    setupapi.SetupDiGetClassDevsW.argtypes = [ctypes.POINTER(GUID), wt.LPCWSTR, wt.HWND, wt.DWORD]
    setupapi.SetupDiGetClassDevsW.restype = ctypes.c_void_p
    setupapi.SetupDiEnumDeviceInterfaces.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(GUID),
                                                     wt.DWORD, ctypes.POINTER(SP_DEVICE_INTERFACE_DATA)]
    setupapi.SetupDiEnumDeviceInterfaces.restype = wt.BOOL
    setupapi.SetupDiGetDeviceInterfaceDetailW.argtypes = [ctypes.c_void_p, ctypes.POINTER(SP_DEVICE_INTERFACE_DATA),
                                                          ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD),
                                                          ctypes.c_void_p]
    setupapi.SetupDiGetDeviceInterfaceDetailW.restype = wt.BOOL
    setupapi.SetupDiDestroyDeviceInfoList.argtypes = [ctypes.c_void_p]
    k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
    k32.CreateFileW.restype = ctypes.c_void_p

    guid = GUID()
    hiddll.HidD_GetHidGuid(ctypes.byref(guid))
    devices = setupapi.SetupDiGetClassDevsW(ctypes.byref(guid), None, None, 0x02 | 0x10)  # PRESENT | INTERFACE
    wanted = f'vid_{VENDOR_ID:04x}&pid_{PRODUCT_ID_PLUS:04x}'
    path, index = None, 0
    while path is None:
        interface = SP_DEVICE_INTERFACE_DATA()
        interface.cbSize = ctypes.sizeof(SP_DEVICE_INTERFACE_DATA)
        if not setupapi.SetupDiEnumDeviceInterfaces(devices, None, ctypes.byref(guid), index, ctypes.byref(interface)):
            break
        index += 1
        size = wt.DWORD(0)
        setupapi.SetupDiGetDeviceInterfaceDetailW(devices, ctypes.byref(interface), None, 0, ctypes.byref(size), None)
        detail = ctypes.create_string_buffer(size.value + 8)
        ctypes.c_uint32.from_buffer(detail).value = 8 if ctypes.sizeof(ctypes.c_void_p) == 8 else 6
        if setupapi.SetupDiGetDeviceInterfaceDetailW(devices, ctypes.byref(interface), detail, size.value + 8,
                                                     None, None):
            candidate = ctypes.wstring_at(ctypes.addressof(detail) + 4)
            if wanted in candidate.lower():
                path = candidate
    setupapi.SetupDiDestroyDeviceInfoList(devices)

    if path is None:
        sys.exit('No Stream Deck+ found.')

    GENERIC_READ, SHARE_READ_WRITE, OPEN_EXISTING = 0x80000000, 0x1 | 0x2, 3
    handle = k32.CreateFileW(path, GENERIC_READ, SHARE_READ_WRITE, None, OPEN_EXISTING, 0, None)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        sys.exit(f'Could not open the Stream Deck+ for reading (error {ctypes.get_last_error()}).')
    return handle, 512


def capture(handle, length, wait_s, idle_s, max_s):
    """Read reports until the dials have been still for idle_s (or max_s since
    the first turn). Returns [(seconds, bytes)]."""
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
    k32.ReadFile.restype = wt.BOOL
    k32.CancelIoEx.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]

    reports, lost = [], []

    def reader():
        buffer, got = ctypes.create_string_buffer(length), wt.DWORD()
        while True:
            if not k32.ReadFile(handle, buffer, length, ctypes.byref(got), None):
                lost.append(True)
                return
            reports.append((time.perf_counter(), buffer.raw[:got.value]))

    threading.Thread(target=reader, daemon=True).start()
    print(f'Turn a dial (waiting up to {wait_s:.0f} s)...', flush=True)

    started = time.perf_counter()
    while True:
        time.sleep(0.05)
        now = time.perf_counter()
        turns = [t for (t, r) in list(reports) if is_turn(r)]
        if lost:
            print('The deck went away.')
            break
        if not turns:
            if now - started > wait_s:
                print('No turn.')
                break
            continue
        if now - turns[-1] > idle_s or now - turns[0] > max_s:
            break

    k32.CancelIoEx(handle, None)
    k32.CloseHandle(handle)
    return list(reports)


def is_turn(report):
    # Report 0x01, event 0x03 (dials), action 0x01 (rotate); clicks at bytes 5-8.
    return len(report) > 8 and report[0] == 0x01 and report[1] == 0x03 and report[4] == 0x01


def signed(byte):
    return byte - 256 if byte > 127 else byte


def summarise(reports, show_reports):
    turns = [(t, r) for (t, r) in reports if is_turn(r)]
    if not turns:
        return
    origin = turns[0][0]
    print(f'{len(turns)} turn reports')

    for dial in range(4):
        clicks = [(t, signed(r[5 + dial])) for (t, r) in turns if r[5 + dial] != 0]
        if not clicks:
            continue

        runs, run, last = [], [], None
        for (t, v) in clicks:
            if last is not None and t - last > RUN_GAP_S:
                runs.append(run)
                run = []
            run.append((t, v))
            last = t
        runs.append(run)

        print(f'\ndial {dial + 1}: {len(runs)} runs')
        mixed = 0
        for k, run in enumerate(runs):
            values = [v for (_, v) in run]
            both = len({v > 0 for v in values}) > 1
            mixed += both
            span = (run[-1][0] - run[0][0]) * 1000 + 50
            print(f'  run {k + 1:3d} @ {(run[0][0] - origin) * 1000:8.0f} ms  {str(values):34s}'
                  f' total {sum(values):+4d}  clicks {sum(abs(v) for v in values):3d}  over {span:5.0f} ms'
                  + ('   <<< BOTH DIRECTIONS' if both else ''))
            if show_reports:
                previous = None
                for (t, v) in run:
                    gap = '' if previous is None else f'  (+{(t - previous) * 1000:.1f} ms)'
                    print(f'        {(t - origin) * 1000:9.1f} ms  {v:+4d}{gap}')
                    previous = t

        gaps = [(b[0] - a[0]) * 1000 for run in runs for a, b in zip(run, run[1:])]
        cadence = f', report gap within runs: median {statistics.median(gaps):.1f} ms' if gaps else ''
        print(f'  runs holding both directions: {mixed}; most clicks in one report:'
              f' {max(abs(v) for (_, v) in clicks)}{cadence}')


def main():
    if sys.platform != 'win32':
        sys.exit('Windows only (it reads the deck through SetupAPI and hid.dll).')

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--wait', type=float, default=120.0, help='seconds to wait for the first turn')
    parser.add_argument('--idle', type=float, default=6.0, help='seconds of stillness that end the capture')
    parser.add_argument('--max', type=float, default=120.0, help='longest capture after the first turn, seconds')
    parser.add_argument('--reports', action='store_true', help='also list every report of every run')
    args = parser.parse_args()

    handle, length = open_deck()
    summarise(capture(handle, length, args.wait, args.idle, args.max), args.reports)


if __name__ == '__main__':
    main()
