#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("NWW_DATA_DIR", str(ROOT_DIR))).resolve()
STATIC_DIR = Path(os.environ.get("STATIC_DIR", str(ROOT_DIR / "static"))).resolve()
PERSONAL_DIR = Path(os.environ.get("NWW_PERSONAL_DIR", str(DATA_DIR / "personal_samples"))).resolve()
NEGATIVE_DIR = Path(os.environ.get("NWW_NEGATIVE_DIR", str(DATA_DIR / "negative_samples"))).resolve()
BACKGROUND_DIR = Path(os.environ.get("NWW_BACKGROUND_DIR", str(DATA_DIR / "background_samples"))).resolve()
RIR_DIR = Path(os.environ.get("NWW_RIR_DIR", str(DATA_DIR / "rir_samples"))).resolve()
CAPTURED_DIR = Path(os.environ.get("NWW_CAPTURED_DIR", str(DATA_DIR / "captured_audio"))).resolve()
TRAINED_DIR = Path(
    os.environ.get("NWW_TRAINED_DIR", os.environ.get("NWW_EXPORT_DIR", str(DATA_DIR / "trained_wake_words")))
).resolve()
LOG_DIR = Path(os.environ.get("NWW_LOG_DIR", str(DATA_DIR / "logs"))).resolve()
TRAIN_SCRIPT = Path(os.environ.get("TRAIN_SCRIPT", str(ROOT_DIR / "train_nanowakeword.sh"))).resolve()

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2
MAX_LOG_LINES = int(os.environ.get("NWW_MAX_LOG_LINES", "1200"))
MODEL_SUFFIXES = {".onnx", ".pt", ".pth"}

for directory in (STATIC_DIR, PERSONAL_DIR, NEGATIVE_DIR, BACKGROUND_DIR, RIR_DIR, CAPTURED_DIR, TRAINED_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NanoWakeWord Trainer")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

STATE_LOCK = threading.Lock()
TRAIN_PROC: subprocess.Popen[str] | None = None
STATE: dict[str, Any] = {
    "training": {
        "running": False,
        "exit_code": None,
        "log_lines": [],
        "log_path": None,
        "safe_word": None,
        "started_at": None,
        "finished_at": None,
    }
}


def safe_name(raw: str) -> str:
    text = (raw or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "wakeword"


def _audio_dir(kind: str) -> Path:
    if kind == "personal":
        return PERSONAL_DIR
    if kind == "negative":
        return NEGATIVE_DIR
    if kind == "background":
        return BACKGROUND_DIR
    if kind == "rir":
        return RIR_DIR
    if kind == "captured":
        return CAPTURED_DIR
    raise HTTPException(status_code=404, detail="Unknown audio collection")


def _resolve_child(directory: Path, name: str) -> Path:
    candidate = Path(name or "").name
    if not candidate or candidate != (name or ""):
        raise HTTPException(status_code=400, detail="Invalid file path")
    path = (directory / candidate).resolve()
    if path.parent != directory.resolve():
        raise HTTPException(status_code=400, detail="Invalid file path")
    return path


def _wav_item(path: Path, directory: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "url": f"/api/audio/{directory.name}/{path.name}",
    }


def _list_wavs(directory: Path) -> list[dict[str, Any]]:
    return [
        _wav_item(path, directory)
        for path in sorted(directory.glob("*.wav"), key=lambda item: item.stat().st_mtime, reverse=True)
    ]


def _list_artifacts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(TRAINED_DIR.glob("*")):
        if path.suffix.lower() not in MODEL_SUFFIXES | {".json"}:
            continue
        stat = path.stat()
        rows.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "url": f"/api/artifacts/{path.name}",
            }
        )
    return rows


def _is_target_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wav:
            return (
                wav.getframerate() == TARGET_SAMPLE_RATE
                and wav.getnchannels() == TARGET_CHANNELS
                and wav.getsampwidth() == TARGET_SAMPLE_WIDTH
            )
    except Exception:
        return False


def _ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _convert_audio(source: Path, dest: Path) -> None:
    if source.suffix.lower() == ".wav" and _is_target_wav(source):
        shutil.copy2(source, dest)
        return

    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        raise HTTPException(status_code=400, detail="ffmpeg is required for non-16k mono PCM WAV uploads")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(dest),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {result.stderr.strip()}")


def _write_raw_pcm_wav(blob: bytes, dest: Path, sample_rate: int) -> None:
    if sample_rate != TARGET_SAMPLE_RATE:
        ffmpeg = _ffmpeg_path()
        if not ffmpeg:
            raise HTTPException(status_code=400, detail="ffmpeg is required to resample raw captured audio")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".raw") as tmp:
            tmp.write(blob)
            tmp_path = Path(tmp.name)
        try:
            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-i",
                str(tmp_path),
                "-ac",
                "1",
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "-sample_fmt",
                "s16",
                str(dest),
            ]
            result = subprocess.run(cmd, text=True, capture_output=True)
            if result.returncode != 0:
                raise HTTPException(status_code=400, detail=f"Raw audio conversion failed: {result.stderr.strip()}")
        finally:
            tmp_path.unlink(missing_ok=True)
        return

    with wave.open(str(dest), "wb") as wav:
        wav.setnchannels(TARGET_CHANNELS)
        wav.setsampwidth(TARGET_SAMPLE_WIDTH)
        wav.setframerate(TARGET_SAMPLE_RATE)
        wav.writeframes(blob)


async def _save_upload(file: UploadFile, directory: Path, prefix: str) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.wav").suffix.lower() or ".wav"
    safe_prefix = safe_name(Path(file.filename or prefix).stem)[:50] or prefix
    out_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{safe_prefix}_{uuid.uuid4().hex[:8]}.wav"
    dest = directory / out_name

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        _convert_audio(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"name": dest.name, "url": f"/api/audio/{directory.name}/{dest.name}"}


def _append_log(line: str, log_path: Path | None = None) -> None:
    clean = line.rstrip("\n")
    with STATE_LOCK:
        logs = STATE["training"]["log_lines"]
        logs.append(clean)
        if len(logs) > MAX_LOG_LINES:
            del logs[: len(logs) - MAX_LOG_LINES]
    if log_path:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(clean + "\n")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>NanoWakeWord Trainer</h1><p>static/index.html is missing.</p>")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/status")
def status() -> JSONResponse:
    with STATE_LOCK:
        training = dict(STATE["training"])
    return JSONResponse(
        {
            "training": training,
            "counts": {
                "personal": len(list(PERSONAL_DIR.glob("*.wav"))),
                "negative": len(list(NEGATIVE_DIR.glob("*.wav"))),
                "background": len(list(BACKGROUND_DIR.glob("*.wav"))),
                "rir": len(list(RIR_DIR.glob("*.wav"))),
                "captured": len(list(CAPTURED_DIR.glob("*.wav"))),
                "artifacts": len(_list_artifacts()),
            },
            "artifacts": _list_artifacts(),
            "defaults": {
                "positive_samples": int(os.environ.get("NWW_DEFAULT_POSITIVE_SAMPLES", "2000")),
                "negative_samples": int(os.environ.get("NWW_DEFAULT_NEGATIVE_SAMPLES", "2000")),
                "validation_samples": int(os.environ.get("NWW_DEFAULT_VALIDATION_SAMPLES", "400")),
                "steps": int(os.environ.get("NWW_DEFAULT_STEPS", "20000")),
                "num_workers": int(os.environ.get("NWW_DEFAULT_NUM_WORKERS", "4")),
            },
        }
    )


@app.get("/api/samples/{kind}")
def list_samples(kind: str) -> JSONResponse:
    return JSONResponse({"items": _list_wavs(_audio_dir(kind))})


@app.post("/api/samples/{kind}/upload")
async def upload_samples(kind: str, files: list[UploadFile] = File(...)) -> JSONResponse:
    directory = _audio_dir(kind)
    saved = [await _save_upload(file, directory, kind) for file in files]
    return JSONResponse({"saved": saved})


@app.delete("/api/samples/{kind}/{name}")
def delete_sample(kind: str, name: str) -> JSONResponse:
    path = _resolve_child(_audio_dir(kind), name)
    path.unlink(missing_ok=True)
    return JSONResponse({"ok": True})


@app.get("/api/audio/{collection}/{name}")
def get_audio(collection: str, name: str) -> FileResponse:
    directory_by_name = {
        PERSONAL_DIR.name: PERSONAL_DIR,
        NEGATIVE_DIR.name: NEGATIVE_DIR,
        BACKGROUND_DIR.name: BACKGROUND_DIR,
        RIR_DIR.name: RIR_DIR,
        CAPTURED_DIR.name: CAPTURED_DIR,
    }
    directory = directory_by_name.get(collection)
    if directory is None:
        raise HTTPException(status_code=404, detail="Unknown audio collection")
    path = _resolve_child(directory, name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.post("/api/captured/{name}/approve")
def approve_captured(name: str) -> JSONResponse:
    src = _resolve_child(CAPTURED_DIR, name)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Captured clip not found")
    dest = PERSONAL_DIR / src.name
    shutil.move(str(src), str(dest))
    return JSONResponse({"ok": True, "name": dest.name})


@app.post("/api/captured/{name}/false_wake")
def false_wake_captured(name: str) -> JSONResponse:
    src = _resolve_child(CAPTURED_DIR, name)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Captured clip not found")
    dest = NEGATIVE_DIR / src.name
    shutil.move(str(src), str(dest))
    return JSONResponse({"ok": True, "name": dest.name})


@app.post("/api/upload_captured_audio_raw")
async def upload_captured_audio_raw(request: Request) -> JSONResponse:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="No audio body received")
    sample_rate = int(request.query_params.get("sample_rate", request.headers.get("x-sample-rate", TARGET_SAMPLE_RATE)))
    out_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_captured_{uuid.uuid4().hex[:8]}.wav"
    dest = CAPTURED_DIR / out_name
    _write_raw_pcm_wav(body, dest, sample_rate)
    return JSONResponse({"ok": True, "name": dest.name, "url": f"/api/audio/{CAPTURED_DIR.name}/{dest.name}"})


@app.post("/api/train")
async def start_training(request: Request) -> JSONResponse:
    global TRAIN_PROC
    payload = await request.json()
    phrase = str(payload.get("phrase") or "").strip()
    if not phrase:
        raise HTTPException(status_code=400, detail="Wake phrase is required")

    with STATE_LOCK:
        if STATE["training"]["running"]:
            raise HTTPException(status_code=409, detail="Training is already running")

    safe_word = safe_name(phrase)
    log_path = LOG_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{safe_word}.log"

    cmd = [
        str(TRAIN_SCRIPT),
        phrase,
        "--positive-samples",
        str(int(payload.get("positive_samples") or 2000)),
        "--negative-samples",
        str(int(payload.get("negative_samples") or 2000)),
        "--validation-samples",
        str(int(payload.get("validation_samples") or 400)),
        "--steps",
        str(int(payload.get("steps") or 20000)),
        "--num-workers",
        str(int(payload.get("num_workers") or 4)),
        "--model-type",
        str(payload.get("model_type") or "dnn"),
        "--layer-size",
        str(int(payload.get("layer_size") or 32)),
        "--target-fp-per-hour",
        str(float(payload.get("target_fp_per_hour") or 0.5)),
    ]
    if bool(payload.get("overwrite")):
        cmd.append("--overwrite")
    for item in payload.get("custom_negative_phrases") or []:
        phrase_item = str(item or "").strip()
        if phrase_item:
            cmd.extend(["--custom-negative-phrase", phrase_item])

    with STATE_LOCK:
        STATE["training"] = {
            "running": True,
            "exit_code": None,
            "log_lines": [],
            "log_path": str(log_path),
            "safe_word": safe_word,
            "started_at": time.time(),
            "finished_at": None,
        }

    def worker() -> None:
        global TRAIN_PROC
        _append_log(f"$ {' '.join(cmd)}", log_path)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            TRAIN_PROC = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert TRAIN_PROC.stdout is not None
            for line in TRAIN_PROC.stdout:
                _append_log(line, log_path)
            exit_code = TRAIN_PROC.wait()
        except Exception as exc:
            _append_log(f"ERROR: {exc}", log_path)
            exit_code = 1
        finally:
            TRAIN_PROC = None
            with STATE_LOCK:
                STATE["training"]["running"] = False
                STATE["training"]["exit_code"] = exit_code
                STATE["training"]["finished_at"] = time.time()

    threading.Thread(target=worker, daemon=True).start()
    return JSONResponse({"ok": True, "log_path": str(log_path), "safe_word": safe_word})


@app.post("/api/train/stop")
def stop_training() -> JSONResponse:
    global TRAIN_PROC
    if TRAIN_PROC and TRAIN_PROC.poll() is None:
        TRAIN_PROC.terminate()
        return JSONResponse({"ok": True, "message": "Training process terminated"})
    return JSONResponse({"ok": True, "message": "No training process is running"})


@app.get("/api/train/log")
def training_log() -> JSONResponse:
    with STATE_LOCK:
        training = dict(STATE["training"])
        lines = list(training.get("log_lines") or [])
    return JSONResponse({"training": training, "lines": lines})


@app.get("/api/artifacts")
def artifacts() -> JSONResponse:
    return JSONResponse({"items": _list_artifacts()})


@app.get("/api/trained_wake_words/catalog")
def trained_wake_words_catalog() -> JSONResponse:
    return JSONResponse({"items": _list_artifacts()})


@app.get("/api/artifacts/{name}")
def get_artifact(name: str) -> FileResponse:
    path = _resolve_child(TRAINED_DIR, name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, filename=path.name)
