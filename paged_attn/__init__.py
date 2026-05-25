"""paged_attn: a step-by-step learning implementation of PagedAttention.

Modules are intentionally written to be read top-to-bottom; each file is a
standalone runnable script and a learning checkpoint.
"""

# ---------------------------------------------------------------------------
# Triton needs `libcuda.so` on the *link* path when it JIT-compiles its CUDA
# helper.  On many systems (including this one) only `libcuda.so.1` lives in
# /usr/lib/x86_64-linux-gnu and the unversioned symlink only ships with the
# CUDA SDK as a stub under .../cuda/lib64/stubs/.  We prepend that to
# LIBRARY_PATH so `gcc -lcuda` succeeds; the real driver is still loaded at
# runtime from /usr/lib/x86_64-linux-gnu.
# ---------------------------------------------------------------------------

import os as _os
import pathlib as _pl

_STUB_CANDIDATES = (
    "/usr/local/cuda/lib64/stubs",
    "/usr/local/cuda-12.6/lib64/stubs",
    "/usr/local/cuda-12/lib64/stubs",
)
for _stub in _STUB_CANDIDATES:
    if (_pl.Path(_stub) / "libcuda.so").exists():
        _existing = _os.environ.get("LIBRARY_PATH", "")
        if _stub not in _existing.split(":"):
            _os.environ["LIBRARY_PATH"] = (
                f"{_stub}:{_existing}" if _existing else _stub
            )
        break
