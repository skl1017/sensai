"""Atomic file writes shared by the storage backends (wiki §7.2/§9.2).

Both `ConversationStore` and `ProfileStore` persist one file per resource and
must never leave a torn file behind, whether the process is killed mid-write
or the final rename fails. `atomic_write` writes to a temporary file created
in the *same directory* as the destination (so the closing `os.replace` stays
on one filesystem, hence atomic), flushes and `fsync`s it, then replaces the
destination in one step. On any failure the temp file is removed and the
destination is left untouched.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def atomic_write(path: Path, text: str) -> None:
    """Atomically write `text` to `path` (temp file in `path`'s directory + fsync + replace)."""
    tmp = NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8")
    try:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    except BaseException:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
