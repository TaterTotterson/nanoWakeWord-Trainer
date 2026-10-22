"""Atomic provisioning for NanoWakeWord's broad negative feature bank."""

from __future__ import annotations

import ast
import errno
import fcntl
import hashlib
import json
import os
import struct
import urllib.request
from pathlib import Path
from typing import Callable, Iterable


FILENAME = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
URL = f"https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/{FILENAME}"
SIZE = 17_280_000_128
SHA256 = "721a66d0682c65a1b5c1da0aa109409cede1d20e28b15235c344b000cbb7654f"
DTYPE = "<f2"
SHAPE = (5_625_000, 16, 96)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _header(path: Path) -> tuple[str, tuple[int, ...]]:
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError("not an NPY file")
        major, _minor = handle.read(2)
        size = struct.unpack("<H" if major == 1 else "<I", handle.read(2 if major == 1 else 4))[0]
        value = ast.literal_eval(handle.read(size).decode("latin1").strip())
    return str(value.get("descr") or ""), tuple(value.get("shape") or ())


def _valid(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size == SIZE and _header(path) == (DTYPE, SHAPE)
    except (OSError, ValueError, SyntaxError, struct.error):
        return False


def _marker_matches(path: Path, marker: Path) -> bool:
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        stat = path.stat()
        return (
            value.get("sha256") == SHA256
            and value.get("size") == stat.st_size
            and value.get("mtime_ns") == stat.st_mtime_ns
            and value.get("resolved_path") == str(path.resolve())
        )
    except (OSError, json.JSONDecodeError):
        return False


def _write_marker(path: Path, marker: Path) -> None:
    stat = path.stat()
    marker.write_text(json.dumps({
        "sha256": SHA256, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "resolved_path": str(path.resolve()), "shape": list(SHAPE), "dtype": DTYPE,
    }, indent=2) + "\n", encoding="utf-8")


def ensure_broad_negative_feature_bank(
    directory: Path,
    *,
    adoption_candidates: Iterable[Path] = (),
    log: Callable[[str], None] = print,
) -> Path:
    """Ensure NWW has the bank even when OWW was not selected or installed."""
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / FILENAME
    marker = root / f".{FILENAME}.verified.json"
    lock = root / f".{FILENAME}.lock"
    with lock.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if _valid(destination) and _marker_matches(destination, marker):
            return destination
        if _valid(destination) and _sha256(destination) == SHA256:
            _write_marker(destination, marker)
            return destination
        for candidate in map(Path, adoption_candidates):
            if _valid(candidate) and _sha256(candidate) == SHA256:
                destination.unlink(missing_ok=True)
                try:
                    os.replace(candidate, destination)
                    action = "Moved"
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
                    destination.symlink_to(candidate.resolve())
                    action = "Linked"
                _write_marker(destination, marker)
                log(f"{action} shared negative feature bank without copying: {candidate}")
                return destination

        destination.unlink(missing_ok=True)
        partial = root / f".{FILENAME}.part"
        partial.unlink(missing_ok=True)
        log("Downloading broad negative feature bank for NanoWakeWord (about 17 GB)")
        try:
            with urllib.request.urlopen(URL) as response, partial.open("wb") as handle:
                while block := response.read(8 * 1024 * 1024):
                    handle.write(block)
            if not _valid(partial) or _sha256(partial) != SHA256:
                raise RuntimeError("downloaded broad negative feature bank failed validation")
            os.replace(partial, destination)
            _write_marker(destination, marker)
            return destination
        except Exception:
            partial.unlink(missing_ok=True)
            raise
