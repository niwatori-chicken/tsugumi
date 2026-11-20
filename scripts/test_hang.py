"""Hang detection: run a kernel that never finishes and see how the host wait ends.

  TT_METAL_OPERATION_TIMEOUT_SECONDS=10 python scripts/test_hang.py [spin=1]
Prints whether ttnn raised (and the message) or the faulthandler backstop fired (exit 1).
Reset the device afterwards (tt-smi -r 0, twice). The server-level test is scripts/test_server_hang.sh."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsumi`
import faulthandler, os, sys, time
import ttnn
from tsumi.hang_kernel import run_hang_kernel

spin = int(sys.argv[1]) if len(sys.argv) > 1 else 1
faulthandler.dump_traceback_later(int(os.environ.get("BACKSTOP", "120")), exit=True)
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
t0 = time.time()
try:
    run_hang_kernel(dev, spin=bool(spin))
    print(f"finished normally {time.time() - t0:.1f}s", flush=True)
except Exception as e:
    print(f"raised after {time.time() - t0:.1f}s: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
    os._exit(3)
ttnn.close_device(dev)
