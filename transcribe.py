#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Whisper Transcription Pipeline
======================================

Portable CPU/GPU transcription for audio and video using faster-whisper.
Includes safe input discovery, streamed FFmpeg preparation, hardware-aware
defaults, decoding presets, bounded fallback, TXT/SRT output, progress and
lightweight quality sanity checks.

Author: Alberto Reyes Castro
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_INPUT = None
DEFAULT_OUTPUT = None
DEFAULT_WAV_OUTPUT = None
DEFAULT_SRT_OUTPUT = None

DEFAULT_DEVICE = "auto"
DEFAULT_COMPUTE_TYPE = "auto"
DEFAULT_MODEL = "auto"
DEFAULT_LANGUAGE = "auto"
DEFAULT_TASK = "transcribe"
DEFAULT_PRESET = "auto"
DEFAULT_THREADS = 8

DEFAULT_FFMPEG_PATH = None
DEFAULT_FFPROBE_PATH = None

# Increment when the semantics of the prepared WAV change.
PREPROCESSING_PROFILE_VERSION = "prep-v3"

SUPPORTED_INPUT_EXTENSIONS = [
    ".mp3", ".mp4", ".m4a", ".wav", ".mov", ".mkv",
    ".aac", ".flac", ".ogg", ".webm", ".wma",
]


@dataclass
class SegmentRecord:
    """Lightweight representation of a transcription segment."""

    start: float
    end: float
    text: str


@dataclass
class QualityReport:
    """Simple structural report used to decide whether an automatic fallback is needed."""

    passed: bool
    issues: List[str]


def get_desktop_candidates() -> List[Path]:
    """Return likely Desktop folders across English/Spanish Windows setups."""
    home = Path.home()
    candidates = [
        home / "OneDrive" / "Escritorio",
        home / "OneDrive" / "Desktop",
        home / "Escritorio",
        home / "Desktop",
    ]
    return [path for path in candidates if path.exists()]


def get_default_input_folders() -> List[Path]:
    """Return dedicated folders where automatic input detection is allowed."""
    return [desktop / "whisper_input" for desktop in get_desktop_candidates()]


def get_default_output_root() -> Path:
    """Return the default output folder."""
    desktops = get_desktop_candidates()
    if desktops:
        return desktops[0] / "whisper_output"
    return Path.cwd() / "whisper_output"


def is_supported_input_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS


def list_supported_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        [path for path in folder.iterdir() if is_supported_input_file(path)],
        key=lambda item: item.name.lower(),
    )


def format_path_list(paths: Sequence[Path]) -> str:
    if not paths:
        return "None"
    return "\n".join(f"  - {path}" for path in paths)


def find_single_file_in_dedicated_input_folder(search_folder: Optional[str] = None) -> Optional[Path]:
    folders = [Path(search_folder).expanduser()] if search_folder else get_default_input_folders()
    found_files: List[Path] = []
    for folder in folders:
        found_files.extend(list_supported_files(folder))

    if len(found_files) == 1:
        return found_files[0]
    if len(found_files) > 1:
        raise RuntimeError(
            "Multiple input files were found in the dedicated input folder(s).\n\n"
            "To avoid transcribing the wrong file, leave only one supported file there "
            "or pass the input explicitly with --input.\n\n"
            f"Detected files:\n{format_path_list(found_files)}"
        )
    return None


def find_default_input_file(search_folder: Optional[str] = None) -> Path:
    dedicated_file = find_single_file_in_dedicated_input_folder(search_folder=search_folder)
    if dedicated_file:
        print(f"✅ Auto-detected input file from dedicated folder: {dedicated_file}")
        return dedicated_file

    raise FileNotFoundError(
        "No input file was provided and no safe default input file was found.\n\n"
        "Recommended Spyder workflow:\n"
        "1) Create a folder called 'whisper_input' on your Desktop or OneDrive Desktop.\n"
        "2) Put exactly ONE supported audio/video file inside it.\n"
        "3) Run the script from Spyder.\n\n"
        "Alternative workflow:\n"
        'python transcribe.py --input "C:\\path\\to\\audio.m4a"\n\n'
        f"Supported extensions: {', '.join(SUPPORTED_INPUT_EXTENSIONS)}\n\n"
        f"Dedicated input folders checked:\n{format_path_list(get_default_input_folders())}"
    )


def input_fingerprint(input_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a short SHA-256 content fingerprint."""
    digest = hashlib.sha256()
    with input_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def preprocessing_identity(input_path: Path, normalize_audio: bool = True) -> Dict[str, Any]:
    """Return the identity required before a prepared WAV is safe to reuse."""
    return {
        "source_sha256_prefix": input_fingerprint(input_path),
        "profile": PREPROCESSING_PROFILE_VERSION,
        "normalize_audio": bool(normalize_audio),
        "sample_rate_hz": 16000,
        "channels": 1,
        "codec": "pcm_s16le",
    }


def prepared_wav_cache_filename(input_path: Path, normalize_audio: bool = True) -> str:
    """Return a cache filename whose identity includes preprocessing semantics."""
    identity = preprocessing_identity(input_path, normalize_audio=normalize_audio)
    normalization = "norm" if normalize_audio else "raw"
    return (
        f"{input_path.stem}_{identity['source_sha256_prefix']}_"
        f"{PREPROCESSING_PROFILE_VERSION}_{normalization}_prepared.wav"
    )


def prepared_wav_metadata_path(wav_path: Path) -> Path:
    """Return the sidecar path that records how a prepared WAV was produced."""
    return wav_path.with_name(f"{wav_path.name}.meta.json")


def _read_prepared_wav_metadata(wav_path: Path) -> Optional[Dict[str, Any]]:
    metadata_path = prepared_wav_metadata_path(wav_path)
    if not metadata_path.is_file():
        return None
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def can_reuse_prepared_wav(
    input_path: Path,
    wav_path: Path,
    normalize_audio: bool = True,
) -> bool:
    """Return True only when both the WAV and its preprocessing identity match."""
    if not wav_path.is_file() or wav_path.stat().st_size <= 0:
        return False
    metadata = _read_prepared_wav_metadata(wav_path)
    if metadata is None:
        return False
    return metadata == preprocessing_identity(input_path, normalize_audio=normalize_audio)


def default_txt_output_path(input_path: Path) -> Path:
    fingerprint = input_fingerprint(input_path)
    return get_default_output_root() / f"{input_path.stem}_{fingerprint}_transcription.txt"


def default_srt_output_path(input_path: Path) -> Path:
    fingerprint = input_fingerprint(input_path)
    return get_default_output_root() / f"{input_path.stem}_{fingerprint}_subtitles.srt"


def default_wav_output_path(input_path: Path, normalize_audio: bool = True) -> Path:
    return get_default_output_root() / "cache" / prepared_wav_cache_filename(
        input_path,
        normalize_audio=normalize_audio,
    )


def _canonical_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _validate_named_paths(named_paths: Sequence[Tuple[str, Path]]) -> None:
    canonical = [(name, _canonical_path(path)) for name, path in named_paths]
    for index, (left_name, left_path) in enumerate(canonical):
        for right_name, right_path in canonical[index + 1 :]:
            if left_path == right_path:
                raise ValueError(
                    f"Path collision: {left_name} and {right_name} resolve to the same path: "
                    f"{left_path}. Choose distinct paths."
                )


def validate_distinct_paths(
    input_path: Path,
    output_txt: Path,
    output_srt: Optional[Path],
    wav_path: Path,
) -> None:
    """Reject path collisions before any write can modify user input or another output."""
    named_paths: List[Tuple[str, Path]] = [
        ("input", input_path),
        ("TXT output", output_txt),
        ("prepared WAV", wav_path),
        ("prepared WAV metadata", prepared_wav_metadata_path(wav_path)),
    ]
    if output_srt is not None:
        named_paths.append(("SRT output", output_srt))
    _validate_named_paths(named_paths)


def validate_transcription_paths(
    wav_path: Path,
    output_txt: Path,
    output_srt: Optional[Path],
) -> None:
    """Reject collisions among the prepared audio and transcript outputs."""
    named_paths: List[Tuple[str, Path]] = [
        ("prepared WAV", wav_path),
        ("prepared WAV metadata", prepared_wav_metadata_path(wav_path)),
        ("TXT output", output_txt),
    ]
    if output_srt is not None:
        named_paths.append(("SRT output", output_srt))
    _validate_named_paths(named_paths)


def configure_threads(threads: int) -> None:
    """Limit CPU threads without enabling unsafe duplicate-runtime workarounds."""
    if threads <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)


def cuda_is_available() -> bool:
    """Return True if CTranslate2 can see at least one CUDA device."""
    try:
        import ctranslate2  # type: ignore

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if cuda_is_available() else "cpu"
    if requested_device == "cuda" and not cuda_is_available():
        print(
            "⚠️  CUDA was requested but no compatible CUDA device was detected. "
            "Falling back to CPU.",
            file=sys.stderr,
        )
        return "cpu"
    return requested_device


def resolve_compute_type(device: str, requested_compute_type: str, safe_cpu_fallback: bool = False) -> str:
    """Resolve a CTranslate2 compute type, keeping CUDA→CPU fallback CPU-safe."""
    if safe_cpu_fallback and device == "cpu":
        if requested_compute_type != "auto":
            print(
                "⚠️  Ignoring the requested compute type after CUDA fallback; "
                "using CPU-safe int8.",
                file=sys.stderr,
            )
        return "int8"
    if requested_compute_type != "auto":
        return requested_compute_type
    return "float16" if device == "cuda" else "int8"


def resolve_model(device: str, requested_model: str) -> str:
    if requested_model != "auto":
        return requested_model
    return "large-v3" if device == "cuda" else "medium"


def _resolve_executable(explicit_path: Optional[str], executable_name: str) -> str:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{executable_name} executable not found: {path}")
        return str(path)

    discovered = shutil.which(executable_name)
    if discovered:
        return discovered

    raise FileNotFoundError(
        f"{executable_name} was not found. Install FFmpeg or pass an explicit "
        f"--{executable_name}-path."
    )


def configure_ffmpeg(ffmpeg_path: Optional[str], ffprobe_path: Optional[str]) -> Tuple[str, str]:
    return (
        _resolve_executable(ffmpeg_path, "ffmpeg"),
        _resolve_executable(ffprobe_path, "ffprobe"),
    )


def ensure_ffmpeg_available(ffmpeg_path: Optional[str]) -> None:
    """Backwards-compatible availability check."""
    _resolve_executable(ffmpeg_path, "ffmpeg")


def probe_audio_duration(path: Path, ffprobe_binary: str) -> float:
    command = [
        ffprobe_binary,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not determine audio duration for: {path}") from exc
    if duration <= 0:
        raise RuntimeError(f"Invalid media duration reported for: {path}")
    return duration


def _parse_max_volume_db(ffmpeg_stderr: str) -> Optional[float]:
    match = re.search(r"max_volume:\s*(-?inf|[-+]?\d+(?:\.\d+)?)\s*dB", ffmpeg_stderr)
    if not match or match.group(1) == "-inf":
        return None
    return float(match.group(1))


def measure_peak_db(input_path: Path, ffmpeg_binary: str) -> Optional[float]:
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-nostats",
        "-i", str(input_path),
        "-vn",
        "-af", "volumedetect",
        "-f", "null",
        os.devnull,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return _parse_max_volume_db(result.stderr)


def _temporary_wav_path(wav_path: Path) -> Path:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{wav_path.stem}.",
        suffix=".tmp.wav",
        dir=wav_path.parent,
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        mode="w",
        encoding="utf-8",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare_wav(
    input_path: Path,
    wav_path: Path,
    normalize_audio: bool = True,
    reuse_existing_wav: bool = True,
    ffmpeg_binary: Optional[str] = None,
    ffprobe_binary: Optional[str] = None,
) -> float:
    """Atomically stream-convert media to 16 kHz mono PCM16 WAV using FFmpeg."""
    input_path = _canonical_path(input_path)
    wav_path = _canonical_path(wav_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path == wav_path:
        raise ValueError("Prepared WAV path must be different from the input path.")

    ffmpeg_binary = ffmpeg_binary or _resolve_executable(None, "ffmpeg")
    ffprobe_binary = ffprobe_binary or _resolve_executable(None, "ffprobe")
    metadata_path = prepared_wav_metadata_path(wav_path)

    if reuse_existing_wav and can_reuse_prepared_wav(
        input_path,
        wav_path,
        normalize_audio=normalize_audio,
    ):
        try:
            duration = probe_audio_duration(wav_path, ffprobe_binary)
        except (subprocess.CalledProcessError, RuntimeError, OSError):
            print(
                f"⚠️  Prepared WAV cache metadata matched but the audio could not be validated; "
                f"rebuilding: {wav_path}",
                file=sys.stderr,
            )
            wav_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
        else:
            print(f"✅ Reusing verified prepared WAV: {wav_path}")
            return duration

    print("🔄 Preparing audio as WAV 16 kHz mono PCM16...")
    temp_wav = _temporary_wav_path(wav_path)
    identity = preprocessing_identity(input_path, normalize_audio=normalize_audio)

    try:
        command = [
            ffmpeg_binary,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(input_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
        ]
        if normalize_audio:
            peak_db = measure_peak_db(input_path, ffmpeg_binary)
            if peak_db is not None:
                gain_db = -0.1 - peak_db
                command.extend(["-af", f"volume={gain_db:.3f}dB"])
        command.append(str(temp_wav))

        subprocess.run(command, check=True)
        duration = probe_audio_duration(temp_wav, ffprobe_binary)
        os.replace(temp_wav, wav_path)
        _write_json_atomic(metadata_path, identity)
        return duration
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg failed while preparing: {input_path}") from exc
    finally:
        temp_wav.unlink(missing_ok=True)


def get_preset_options(preset: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "task": "transcribe",
        "beam_size": 5,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }

    presets: Dict[str, Dict[str, Any]] = {
        "fast": {
            **base,
            "beam_size": 1,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 300, "speech_pad_ms": 150},
        },
        "balanced": {
            **base,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 200},
        },
        "robust": {
            **base,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 700, "speech_pad_ms": 300},
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.10,
        },
        "no-vad": {
            **base,
            "vad_filter": False,
        },
        "gpu-high-accuracy": {
            **base,
            "beam_size": 8,
            "patience": 1.2,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 300},
        },
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}")
    return presets[preset]


def preset_sequence(preset: str) -> List[str]:
    if preset == "auto":
        return ["balanced", "robust", "no-vad"]
    return [preset]


def filter_supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except Exception:
        return kwargs

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def load_model(model_source: str, device: str, compute_type: str, threads: int):
    from faster_whisper import WhisperModel

    model_path = Path(model_source).expanduser()
    source = str(model_path) if model_path.is_dir() else model_source
    init_kwargs: Dict[str, Any] = {"device": device, "compute_type": compute_type}
    if device == "cpu" and threads > 0:
        init_kwargs["cpu_threads"] = threads
    init_kwargs = filter_supported_kwargs(WhisperModel.__init__, init_kwargs)

    print(f"⏳ Loading model: {source}")
    print(f"🖥️  Device: {device} | compute_type: {compute_type}")
    return WhisperModel(source, **init_kwargs)


def clean_segment_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split())


def should_skip_segment(segment: Any, text: str, strict_filtering: bool) -> bool:
    if not text:
        return True
    if not strict_filtering:
        return False

    no_speech_prob = getattr(segment, "no_speech_prob", None)
    avg_logprob = getattr(segment, "avg_logprob", None)
    compression_ratio = getattr(segment, "compression_ratio", None)

    if no_speech_prob is not None and avg_logprob is not None:
        if no_speech_prob > 0.80 and avg_logprob < -1.0:
            return True
    if compression_ratio is not None and compression_ratio > 2.4:
        return True
    return False


def srt_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt_block(handle, index: int, start: float, end: float, text: str) -> None:
    handle.write(f"{index}\n")
    handle.write(f"{srt_timestamp(start)} --> {srt_timestamp(end)}\n")
    handle.write(f"{text}\n\n")


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_progress_bar(current_time: float, total_time: float, prefix: str = "Progress", width: int = 32) -> None:
    if total_time <= 0:
        return
    ratio = min(max(current_time / total_time, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "█" * filled + "░" * (width - filled)
    print(
        f"\r{prefix}: |{bar}| {ratio * 100:5.1f}%  "
        f"{format_duration(current_time)} / {format_duration(total_time)}",
        end="",
        flush=True,
    )


def transcribe_once(
    model: Any,
    wav_path: Path,
    output_txt: Path,
    output_srt: Optional[Path],
    preset: str,
    language: Optional[str],
    task: str,
    include_timestamps_in_txt: bool,
    audio_duration: float,
) -> List[SegmentRecord]:
    options = get_preset_options(preset)
    options["task"] = task
    if language is not None:
        options["language"] = language
    options = filter_supported_kwargs(model.transcribe, options)

    print(f"🎙️  Transcribing with preset: {preset}")
    segments_iterator, info = model.transcribe(str(wav_path), **options)

    if hasattr(info, "language"):
        probability = getattr(info, "language_probability", None)
        if probability is not None:
            print(f"🌍 Detected language: {info.language} ({probability:.2f})")
        else:
            print(f"🌍 Detected language: {info.language}")

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    if output_srt:
        output_srt.parent.mkdir(parents=True, exist_ok=True)

    strict_filtering = preset == "robust"
    records: List[SegmentRecord] = []
    last_text = ""
    repeat_streak = 0
    last_progress_time = 0.0
    srt_handle = open(output_srt, "w", encoding="utf-8") if output_srt else None

    try:
        with open(output_txt, "w", encoding="utf-8") as txt_handle:
            for raw_segment in segments_iterator:
                text = clean_segment_text(getattr(raw_segment, "text", ""))
                if should_skip_segment(raw_segment, text, strict_filtering):
                    continue

                if text == last_text:
                    repeat_streak += 1
                    if repeat_streak >= 2:
                        continue
                else:
                    repeat_streak = 0
                    last_text = text

                start = float(getattr(raw_segment, "start", 0.0))
                end = float(getattr(raw_segment, "end", start))
                record = SegmentRecord(start=start, end=end, text=text)
                records.append(record)

                if include_timestamps_in_txt:
                    txt_handle.write(f"[{srt_timestamp(start)} --> {srt_timestamp(end)}] {text}\n")
                else:
                    txt_handle.write(text + "\n")
                txt_handle.flush()

                if srt_handle:
                    write_srt_block(srt_handle, len(records), start, end, text)
                    srt_handle.flush()

                if end - last_progress_time >= 2.0 or len(records) == 1:
                    print_progress_bar(end, audio_duration, prefix=f"Transcribing ({preset})")
                    last_progress_time = end
    finally:
        if srt_handle:
            srt_handle.close()

    if records:
        last_end = max(record.end for record in records)
        print_progress_bar(last_end, audio_duration, prefix=f"Transcribing ({preset})")
    print()
    print(f"✅ Decoding finished; segments written: {len(records)}")
    return records


def lexical_diversity(texts: Sequence[str]) -> float:
    words: List[str] = []
    for text in texts:
        words.extend(word.lower() for word in text.split() if word.strip())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def assess_transcription(records: Sequence[SegmentRecord], audio_duration: float) -> QualityReport:
    """Detect common structural failures; this is not an accuracy metric."""
    issues: List[str] = []
    if not records:
        return QualityReport(False, ["no transcription segments were produced"])

    texts = [record.text for record in records]
    total_chars = sum(len(text) for text in texts)
    last_end = max(record.end for record in records)

    if audio_duration > 20 and total_chars < 80:
        issues.append("transcription is unexpectedly short")
    if audio_duration > 60 and last_end < audio_duration * 0.45:
        issues.append("transcription appears to stop too early")

    repeated_lines = sum(1 for previous, current in zip(texts, texts[1:]) if previous == current)
    if len(texts) >= 10 and repeated_lines / max(1, len(texts) - 1) > 0.20:
        issues.append("many consecutive repeated segments detected")

    diversity = lexical_diversity(texts)
    word_count = sum(len(text.split()) for text in texts)
    if word_count > 80 and diversity < 0.18:
        issues.append("very low lexical diversity; possible repetition loop")

    return QualityReport(passed=len(issues) == 0, issues=issues)


def _temporary_attempt_path(final_path: Optional[Path], attempt_index: int, preset: str) -> Optional[Path]:
    if final_path is None:
        return None
    final_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{final_path.name}.attempt-{attempt_index}-{preset}.",
        suffix=".tmp",
        dir=final_path.parent,
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _promote_attempt(temp_path: Optional[Path], final_path: Optional[Path]) -> None:
    if temp_path is None or final_path is None:
        return
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_path, final_path)


def _cleanup_attempt(path: Optional[Path]) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def transcribe_with_fallbacks(
    model: Any,
    wav_path: Path,
    output_txt: Path,
    output_srt: Optional[Path],
    requested_preset: str,
    language: Optional[str],
    task: str,
    include_timestamps_in_txt: bool,
    audio_duration: float,
) -> Tuple[str, QualityReport]:
    """Run retries with per-file atomic promotion of final outputs."""
    validate_transcription_paths(wav_path, output_txt, output_srt)
    attempts = preset_sequence(requested_preset)
    last_report = QualityReport(False, ["not attempted"])
    final_preset = attempts[-1]

    for attempt_index, preset in enumerate(attempts, start=1):
        if attempt_index > 1:
            print(f"\n🔁 Retrying with fallback preset: {preset}")

        temp_txt = _temporary_attempt_path(output_txt, attempt_index, preset)
        temp_srt = _temporary_attempt_path(output_srt, attempt_index, preset)
        assert temp_txt is not None

        try:
            records = transcribe_once(
                model=model,
                wav_path=wav_path,
                output_txt=temp_txt,
                output_srt=temp_srt,
                preset=preset,
                language=language,
                task=task,
                include_timestamps_in_txt=include_timestamps_in_txt,
                audio_duration=audio_duration,
            )
            report = assess_transcription(records, audio_duration)
            final_preset = preset
            last_report = report

            is_last_attempt = attempt_index == len(attempts)
            should_finish = report.passed or requested_preset != "auto" or is_last_attempt
            if should_finish:
                _promote_attempt(temp_txt, output_txt)
                _promote_attempt(temp_srt, output_srt)
                break

            print("⚠️  Possible quality issues detected:")
            for issue in report.issues:
                print(f"   - {issue}")
        finally:
            _cleanup_attempt(temp_txt)
            _cleanup_attempt(temp_srt)

    return final_preset, last_report


def _running_inside_spyder() -> bool:
    return any(name == "spyder_kernels" or name.startswith("spyder_kernels.") for name in sys.modules)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptive CPU/GPU Whisper transcription pipeline using faster-whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT,
                        help="Input audio/video file. If omitted, checks only the dedicated whisper_input folder.")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help="Output TXT file. If omitted, a source-fingerprinted default path is created.")
    parser.add_argument("--srt-output", default=DEFAULT_SRT_OUTPUT,
                        help="Optional SRT subtitle output path.")
    parser.add_argument("--wav-output", default=DEFAULT_WAV_OUTPUT,
                        help="Optional intermediate WAV path. Reuse is verified by a preprocessing metadata sidecar.")
    parser.add_argument("--input-folder", default=None,
                        help="Optional dedicated folder for safe input auto-detection.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_DEVICE,
                        help="Execution device.")
    parser.add_argument("--compute-type", default=DEFAULT_COMPUTE_TYPE,
                        help="CTranslate2 compute type, e.g. auto, int8, float16, float32 or int8_float16.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model name or local model directory. Use auto for CPU/GPU-aware defaults.")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        help="Language code such as en or es. Use auto for detection.")
    parser.add_argument("--task", choices=["transcribe", "translate"], default=DEFAULT_TASK,
                        help="Keep the original language or translate to English.")
    parser.add_argument("--preset",
                        choices=["auto", "fast", "balanced", "robust", "no-vad", "gpu-high-accuracy"],
                        default=DEFAULT_PRESET,
                        help="Behaviour preset. Auto tries balanced, robust, then no-vad if sanity checks fail.")
    parser.add_argument("--ffmpeg-path", default=DEFAULT_FFMPEG_PATH,
                        help="Optional explicit path to ffmpeg executable.")
    parser.add_argument("--ffprobe-path", default=DEFAULT_FFPROBE_PATH,
                        help="Optional explicit path to ffprobe executable.")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help="CPU threads for inference and OpenMP/MKL limits. Use 0 to leave defaults.")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable audio peak normalisation before transcription.")
    parser.add_argument("--no-reuse-wav", action="store_true",
                        help="Always recreate the intermediate WAV file.")
    parser.add_argument("--timestamps", action="store_true",
                        help="Include timestamps in the TXT output.")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse strictly in CLI mode; tolerate injected args only in Spyder."""
    parser = build_argument_parser()
    if argv is None and _running_inside_spyder():
        args, unknown_args = parser.parse_known_args()
        if unknown_args:
            print("⚠️  Ignoring arguments injected by Spyder:", unknown_args)
    else:
        args = parser.parse_args(argv)

    if not args.input:
        detected_input = find_default_input_file(search_folder=args.input_folder)
        args.input = str(detected_input)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_threads(args.threads)
    ffmpeg_binary, ffprobe_binary = configure_ffmpeg(args.ffmpeg_path, args.ffprobe_path)

    input_path = _canonical_path(Path(args.input))
    output_txt = (
        _canonical_path(Path(args.output))
        if args.output
        else _canonical_path(default_txt_output_path(input_path))
    )
    output_srt = _canonical_path(Path(args.srt_output)) if args.srt_output else None
    normalize_audio = not args.no_normalize
    wav_path = (
        _canonical_path(Path(args.wav_output))
        if args.wav_output
        else _canonical_path(default_wav_output_path(input_path, normalize_audio=normalize_audio))
    )

    validate_distinct_paths(input_path, output_txt, output_srt, wav_path)

    device = resolve_device(args.device)
    safe_cpu_fallback = args.device == "cuda" and device == "cpu"
    compute_type = resolve_compute_type(
        device,
        args.compute_type,
        safe_cpu_fallback=safe_cpu_fallback,
    )
    model_source = resolve_model(device, args.model)
    language = None if args.language.lower() == "auto" else args.language

    print("\n=== Adaptive Whisper Transcription Pipeline ===")
    print(f"Input:        {input_path}")
    print(f"Output TXT:   {output_txt}")
    print(f"WAV file:     {wav_path}")
    if output_srt:
        print(f"Output SRT:   {output_srt}")
    print(f"Device:       {device}")
    print(f"Compute type: {compute_type}")
    print(f"Model:        {model_source}")
    print(f"Preset:       {args.preset}")
    print(f"Language:     {args.language}")

    audio_duration = prepare_wav(
        input_path=input_path,
        wav_path=wav_path,
        normalize_audio=normalize_audio,
        reuse_existing_wav=not args.no_reuse_wav,
        ffmpeg_binary=ffmpeg_binary,
        ffprobe_binary=ffprobe_binary,
    )
    print(f"⏱️  Audio duration: {audio_duration:.1f}s")

    model = load_model(model_source, device, compute_type, args.threads)
    final_preset, report = transcribe_with_fallbacks(
        model=model,
        wav_path=wav_path,
        output_txt=output_txt,
        output_srt=output_srt,
        requested_preset=args.preset,
        language=language,
        task=args.task,
        include_timestamps_in_txt=args.timestamps,
        audio_duration=audio_duration,
    )

    print("\n=== Completed ===")
    print(f"Final preset: {final_preset}")
    print(f"TXT saved to: {output_txt}")
    if output_srt:
        print(f"SRT saved to: {output_srt}")
    if report.passed:
        print("Quality sanity check: passed")
    else:
        print("Quality sanity check: review recommended")
        for issue in report.issues:
            print(f"- {issue}")


if __name__ == "__main__":
    main()
