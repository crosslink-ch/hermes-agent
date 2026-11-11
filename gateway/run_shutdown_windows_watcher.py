"""Source of the detached watcher process that restarts the gateway on Windows."""

# Windows has no bash/setsid chain: a tiny detached Python watcher waits for the gateway PID to
# exit (bounded), then spawns ``hermes gateway restart``.
_WINDOWS_RESTART_WATCHER = """
import os, subprocess, sys, time
from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway
pid = int(sys.argv[1])
restart_after_s = float(sys.argv[2])
cmd = sys.argv[3:]
deadline = time.monotonic() + restart_after_s

def _alive(p):
    # On Windows, os.kill(pid, 0) is NOT a no-op — it maps to
    # GenerateConsoleCtrlEvent(0, pid) (bpo-14484). Use the
    # Win32 handle-based existence check instead.
    if os.name == 'nt':
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.WaitForSingleObject.restype = ctypes.c_uint
        k32.GetLastError.restype = ctypes.c_uint
        h = k32.OpenProcess(0x1000 | 0x100000, False, int(p))
        if not h:
            return k32.GetLastError() != 87
        try:
            return k32.WaitForSingleObject(h, 0) == 0x102
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(p), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False

while time.monotonic() < deadline:
    if not _alive(pid):
        break
    time.sleep(0.2)
subprocess.Popen(
    cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=windows_detach_flags_without_breakaway(),
)
""".strip()
