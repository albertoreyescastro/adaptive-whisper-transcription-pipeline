#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Whisper Transcription Pipeline
======================================

A portable CPU/GPU transcription workflow for audio and video files using
faster-whisper. It includes automatic hardware selection, audio normalisation,
VAD presets, anti-repetition safeguards, streaming text output, a visual progress
bar and optional SRT subtitle generation.

Author: Alberto Reyes Castro
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Spyder-friendly default configuration
# ---------------------------------------------------------------------------
# This script can be used in two safe ways:
#
# 1) Professional CLI mode:
#    python transcribe.py --input "C:\path\to\audio.m4a"
#
# 2) Spyder-friendly mode:
#    Create a folder called "whisper_input" on your Desktop or OneDrive Desktop.
#    Put exactly ONE supported audio/video file inside it.
#    Then run the script from Spyder.
#
# Recommended folder:
#    C:\Users\<your_user>\OneDrive\Escritorio\whisper_input
#
# Safety rule:
#    The script does NOT scan the whole Desktop, the current working directory,
#    or the Spyder folder for random audio/video files.

DEFAULT_INPUT = None
DEFAULT_OUTPUT = None
DEFAULT_WAV_OUTPUT = None
DEFAULT_SRT_OUTPUT = None

DEFAULT_DEVICE = "auto"          # Options: "auto", "cpu", "cuda"
DEFAULT_COMPUTE_TYPE = "auto"    # Options: "auto", "int8", "float16", "float32", etc.
DEFAULT_MODEL = "auto"           # Options: "auto", "medium", "large-v3", or local model path
DEFAULT_LANGUAGE = "auto"        # Use "auto", "en", "es", etc.
DEFAULT_TASK = "transcribe"      # Options: "transcribe", "translate"
DEFAULT_PRESET = "auto"          # Options: "auto", "fast", "balanced", "robust", "no-vad", "gpu-high-accuracy"
DEFAULT_THREADS = 8

# Optional FFmpeg paths.
# Leave them as None if FFmpeg already works from your environment/PATH.
DEFAULT_FFMPEG_PATH = None
DEFAULT_FFPROBE_PATH = None

SUPPORTED_INPUT_EXTENSIONS = [
    ".mp3",
    ".mp4",
    ".m4a",
    ".wav",
    ".mov",
    ".mkv",
    ".aac",
    ".flac",
    ".ogg",
    ".webm",
    ".wma",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SegmentRecord:
    """Lightweight representation of a transcription segment."""

    start: float
    end: float
    text: str


@dataclass
class QualityReport:
    """Simple quality report used to decide whether a fallback preset is needed."""

    passed: bool
    issues: List[str]


# ---------------------------------------------------------------------------
# Path discovery helpers
# ---------------------------------------------------------------------------

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
    """
    Return dedicated folders where the script is allowed to auto-detect files.

    The script only auto-detects input files inside these folders.
    It does not scan the full Desktop, Spyder folder or current directory.
    """
    folders: List[Path] = []

    for desktop in get_desktop_candidates():
        folders.append(desktop / "whisper_input")

    return folders


def get_default_output_root() -> Path:
    """Return the default output folder."""
    desktop_candidates = get_desktop_candidates()

    if desktop_candidates:
        return desktop_candidates[0] / "whisper_output"

    return Path.cwd() / "whisper_output"


def is_supported_input_file(path: Path) -> bool:
    """Return True if the file extension is supported."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS


def list_supported_files(folder: Path) -> List[Path]:
    """List supported audio/video files in a folder."""
    if not folder.is_dir():
        return []

    return sorted(
        [path for path in folder.iterdir() if is_supported_input_file(path)],
        key=lambda item: item.name.lower(),
    )


def format_path_list(paths: Sequence[Path]) -> str:
    """Format a list of paths for user-friendly console output."""
    if not paths:
        return "None"

    return "\n".join(f"  - {path}" for path in paths)


def find_single_file_in_dedicated_input_folder(search_folder: Optional[str] = None) -> Optional[Path]:
    """
    Find exactly one supported file inside a dedicated input folder.

    This is the only automatic detection mode. It prevents accidental
    transcription of files from Desktop, Spyder, Downloads or any other folder.
    """
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
    """
    Find a default input file safely.

    Priority:
    1) Exactly one supported file inside a dedicated 'whisper_input' folder.
    2) Otherwise, stop and ask the user to provide --input.

    This function does NOT scan the Desktop, Spyder folder or current directory.
    """
    dedicated_file = find_single_file_in_dedicated_input_folder(search_folder=search_folder)

    if dedicated_file:
        print(f"✅ Auto-detected input file from dedicated folder: {dedicated_file}")
        return dedicated_file

    default_input_folders = get_default_input_folders()

    raise FileNotFoundError(
        "No input file was provided and no safe default input file was found.\n\n"
        "Recommended Spyder workflow:\n"
        "1) Create a folder called 'whisper_input' on your Desktop or OneDrive Desktop.\n"
        "2) Put exactly ONE supported audio/video file inside it.\n"
        "3) Run the script from Spyder.\n\n"
        "Alternative workflow:\n"
        "Run the script from Anaconda Prompt with:\n"
        'python transcribe.py --input "C:\\path\\to\\audio.m4a"\n\n'
        "Supported extensions:\n"
        f"{', '.join(SUPPORTED_INPUT_EXTENSIONS)}\n\n"
        "Dedicated input folders checked:\n"
        f"{format_path_list(default_input_folders)}"
    )


def input_fingerprint(input_path: Path) -> str:
    """
    Create a short fingerprint from file path, size and modification time.

    This prevents accidentally reusing an old WAV cache when the input extension
    or content changes.
    """
    stat = input_path.stat()
    raw = f"{input_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")

    return hashlib.md5(raw).hexdigest()[:10]


def default_txt_output_path(input_path: Path) -> Path:
    """Create a default TXT output path based on the input file name."""
    output_root = get_default_output_root()
    return output_root / f"{input_path.stem}_transcription.txt"


def default_srt_output_path(input_path: Path) -> Path:
    """Create a default SRT output path based on the input file name."""
    output_root = get_default_output_root()
    return output_root / f"{input_path.stem}_subtitles.srt"


def default_wav_output_path(input_path: Path) -> Path:
    """Create a unique intermediate WAV path based on the input file fingerprint."""
    output_root = get_default_output_root()
    fingerprint = input_fingerprint(input_path)

    return output_root / "cache" / f"{input_path.stem}_{fingerprint}_prepared.wav"


# ---------------------------------------------------------------------------
# Environment and device handling
# ---------------------------------------------------------------------------

def configure_threads(threads: int) -> None:
    """Limit CPU thread usage for OpenMP/MKL-based libraries."""
    if threads <= 0:
        return

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)


def cuda_is_available() -> bool:
    """Return True if a CUDA-capable device appears to be available."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return True
    except Exception:
        pass

    try:
        import ctranslate2  # type: ignore

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_device(requested_device: str) -> str:
    """Resolve 'auto' into either 'cuda' or 'cpu'."""
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


def resolve_compute_type(device: str, requested_compute_type: str) -> str:
    """Choose a sensible compute type for the selected hardware."""
    if requested_compute_type != "auto":
        return requested_compute_type

    return "float16" if device == "cuda" else "int8"


def resolve_model(device: str, requested_model: str) -> str:
    """Choose a sensible default model for CPU or GPU execution."""
    if requested_model != "auto":
        return requested_model

    return "large-v3" if device == "cuda" else "medium"


# ---------------------------------------------------------------------------
# FFmpeg and audio preparation
# ---------------------------------------------------------------------------

def configure_ffmpeg(ffmpeg_path: Optional[str], ffprobe_path: Optional[str]) -> None:
    """Configure pydub to use explicit FFmpeg/FFprobe binaries when provided."""
    from pydub import AudioSegment

    if ffmpeg_path:
        path = Path(ffmpeg_path).expanduser()

        if not path.is_file():
            raise FileNotFoundError(f"FFmpeg executable not found: {path}")

        AudioSegment.converter = str(path)

    if ffprobe_path:
        path = Path(ffprobe_path).expanduser()

        if not path.is_file():
            raise FileNotFoundError(f"FFprobe executable not found: {path}")

        AudioSegment.ffprobe = str(path)


def ensure_ffmpeg_available(ffmpeg_path: Optional[str]) -> None:
    """Warn the user if FFmpeg is not visible from PATH and no explicit path is set."""
    if ffmpeg_path:
        return

    if shutil.which("ffmpeg") is None:
        print(
            "⚠️  FFmpeg was not found in PATH. If audio extraction fails, install FFmpeg "
            "or pass --ffmpeg-path and --ffprobe-path.",
            file=sys.stderr,
        )


def prepare_wav(
    input_path: Path,
    wav_path: Path,
    normalize_audio: bool = True,
    reuse_existing_wav: bool = True,
) -> float:
    """
    Convert any audio/video input into WAV 16 kHz mono PCM16.

    Returns
    -------
    float
        Audio duration in seconds.
    """
    from pydub import AudioSegment, effects

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if reuse_existing_wav and wav_path.is_file() and wav_path.stat().st_size > 0:
        print(f"✅ Reusing existing WAV: {wav_path}")
        audio = AudioSegment.from_file(str(wav_path))
        return len(audio) / 1000.0

    print("🔄 Preparing audio as WAV 16 kHz mono PCM16...")
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(str(input_path))
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

    if normalize_audio:
        audio = effects.normalize(audio)

    audio.export(
        str(wav_path),
        format="wav",
        parameters=["-acodec", "pcm_s16le"],
    )

    return len(audio) / 1000.0


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def get_preset_options(preset: str) -> Dict[str, Any]:
    """Return faster-whisper transcription options for a given preset."""
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
            "vad_parameters": {
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 150,
            },
        },
        "balanced": {
            **base,
            "vad_filter": True,
            "vad_parameters": {
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
        },
        "robust": {
            **base,
            "vad_filter": True,
            "vad_parameters": {
                "min_silence_duration_ms": 700,
                "speech_pad_ms": 300,
            },
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.10,
        },
        "no-vad": {
            **base,
            "vad_filter": False,
        },
        "gpu-high-accuracy": {
            **base,
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": {
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
        },
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}")

    return presets[preset]


def preset_sequence(preset: str) -> List[str]:
    """Return the preset order used by normal and automatic modes."""
    if preset == "auto":
        return ["balanced", "robust", "no-vad"]

    return [preset]


# ---------------------------------------------------------------------------
# Model loading and transcription
# ---------------------------------------------------------------------------

def filter_supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keyword arguments unsupported by a callable signature."""
    try:
        signature = inspect.signature(callable_obj)
    except Exception:
        return kwargs

    parameters = signature.parameters

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in parameters}


def load_model(model_source: str, device: str, compute_type: str, threads: int):
    """Load a faster-whisper model from a model name or local directory."""
    from faster_whisper import WhisperModel

    model_path = Path(model_source).expanduser()
    source = str(model_path) if model_path.is_dir() else model_source

    init_kwargs: Dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
    }

    if device == "cpu" and threads > 0:
        init_kwargs["cpu_threads"] = threads

    init_kwargs = filter_supported_kwargs(WhisperModel.__init__, init_kwargs)

    print(f"⏳ Loading model: {source}")
    print(f"🖥️  Device: {device} | compute_type: {compute_type}")

    return WhisperModel(source, **init_kwargs)


def clean_segment_text(text: Optional[str]) -> str:
    """Normalise whitespace in a transcription segment."""
    if not text:
        return ""

    return " ".join(text.strip().split())


def should_skip_segment(segment: Any, text: str, strict_filtering: bool) -> bool:
    """Apply optional filters for likely hallucinated or low-confidence segments."""
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
    """Format seconds as an SRT timestamp."""
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt_block(handle, index: int, start: float, end: float, text: str) -> None:
    """Write one SRT subtitle block."""
    handle.write(f"{index}\n")
    handle.write(f"{srt_timestamp(start)} --> {srt_timestamp(end)}\n")
    handle.write(f"{text}\n\n")


def format_duration(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def print_progress_bar(
    current_time: float,
    total_time: float,
    prefix: str = "Progress",
    width: int = 32,
) -> None:
    """
    Print a terminal progress bar based on the current audio timestamp.

    This is not a low-level model progress indicator. It estimates progress
    using the end time of the latest transcribed segment.
    """
    if total_time <= 0:
        return

    ratio = min(max(current_time / total_time, 0.0), 1.0)
    filled = int(width * ratio)
    empty = width - filled

    bar = "█" * filled + "░" * empty
    percent = ratio * 100

    current_label = format_duration(current_time)
    total_label = format_duration(total_time)

    print(
        f"\r{prefix}: |{bar}| {percent:5.1f}%  {current_label} / {total_label}",
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
    """Run one transcription attempt and stream the result to output files."""
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

                # Visual progress bar. The update is throttled slightly so the
                # console remains readable even for very short segments.
                if end - last_progress_time >= 2.0 or len(records) == 1:
                    print_progress_bar(
                        current_time=end,
                        total_time=audio_duration,
                        prefix=f"Transcribing ({preset})",
                    )
                    last_progress_time = end

    finally:
        if srt_handle:
            srt_handle.close()

    if records:
        print_progress_bar(
            current_time=max(record.end for record in records),
            total_time=audio_duration,
            prefix=f"Transcribing ({preset})",
        )

    print()
    print(f"✅ Segments written: {len(records)}")

    return records


# ---------------------------------------------------------------------------
# Quality checks and fallback logic
# ---------------------------------------------------------------------------

def lexical_diversity(texts: Sequence[str]) -> float:
    """Return a simple unique-word ratio for the transcription."""
    words: List[str] = []

    for text in texts:
        words.extend(word.lower() for word in text.split() if word.strip())

    if not words:
        return 0.0

    return len(set(words)) / len(words)


def assess_transcription(records: Sequence[SegmentRecord], audio_duration: float) -> QualityReport:
    """Detect common failure modes: empty output, early stop and repetition loops."""
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

    repeated_lines = sum(
        1 for previous, current in zip(texts, texts[1:])
        if previous == current
    )

    if len(texts) >= 10 and repeated_lines / max(1, len(texts) - 1) > 0.20:
        issues.append("many consecutive repeated segments detected")

    diversity = lexical_diversity(texts)
    word_count = sum(len(text.split()) for text in texts)

    if word_count > 80 and diversity < 0.18:
        issues.append("very low lexical diversity; possible repetition loop")

    return QualityReport(passed=len(issues) == 0, issues=issues)


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
    """Run the requested preset and retry with safer presets when needed."""
    attempts = preset_sequence(requested_preset)
    last_report = QualityReport(False, ["not attempted"])
    final_preset = attempts[-1]

    for attempt_index, preset in enumerate(attempts, start=1):
        if attempt_index > 1:
            print(f"\n🔁 Retrying with fallback preset: {preset}")

        records = transcribe_once(
            model=model,
            wav_path=wav_path,
            output_txt=output_txt,
            output_srt=output_srt,
            preset=preset,
            language=language,
            task=task,
            include_timestamps_in_txt=include_timestamps_in_txt,
            audio_duration=audio_duration,
        )

        report = assess_transcription(records, audio_duration)
        final_preset = preset
        last_report = report

        if report.passed or requested_preset != "auto":
            break

        print("⚠️  Possible quality issues detected:")

        for issue in report.issues:
            print(f"   - {issue}")

    return final_preset, last_report


# ---------------------------------------------------------------------------
# CLI / Spyder-friendly argument handling
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive CPU/GPU Whisper transcription pipeline using faster-whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        required=False,
        default=DEFAULT_INPUT,
        help="Input audio/video file. If omitted, the script checks only the dedicated whisper_input folder.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output TXT file. If omitted, a default output path is created.",
    )

    parser.add_argument(
        "--srt-output",
        default=DEFAULT_SRT_OUTPUT,
        help="Optional SRT subtitle output path.",
    )

    parser.add_argument(
        "--wav-output",
        default=DEFAULT_WAV_OUTPUT,
        help="Optional intermediate WAV path. If omitted, a unique cache path is created.",
    )

    parser.add_argument(
        "--input-folder",
        default=None,
        help="Optional dedicated folder used for safe input auto-detection.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=DEFAULT_DEVICE,
        help="Execution device.",
    )

    parser.add_argument(
        "--compute-type",
        default=DEFAULT_COMPUTE_TYPE,
        help="CTranslate2 compute type, for example auto, int8, float16, float32 or int8_float16.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name or local model directory. Use auto for CPU/GPU-aware defaults.",
    )

    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Language code such as en or es. Use auto for automatic detection.",
    )

    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default=DEFAULT_TASK,
        help="Use transcribe to keep the original language, or translate to English.",
    )

    parser.add_argument(
        "--preset",
        choices=["auto", "fast", "balanced", "robust", "no-vad", "gpu-high-accuracy"],
        default=DEFAULT_PRESET,
        help="Transcription behaviour preset. Auto tries balanced, then robust, then no-vad if needed.",
    )

    parser.add_argument(
        "--ffmpeg-path",
        default=DEFAULT_FFMPEG_PATH,
        help="Optional explicit path to ffmpeg.exe.",
    )

    parser.add_argument(
        "--ffprobe-path",
        default=DEFAULT_FFPROBE_PATH,
        help="Optional explicit path to ffprobe.exe.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="CPU threads for CPU inference and OpenMP/MKL limits. Use 0 to leave defaults.",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable audio volume normalisation before transcription.",
    )

    parser.add_argument(
        "--no-reuse-wav",
        action="store_true",
        help="Always recreate the intermediate WAV file.",
    )

    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in the TXT output.",
    )

    # parse_known_args prevents Spyder/IPython arguments from crashing the script.
    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        print("⚠️  Ignoring unknown arguments added by the IDE:", unknown_args)

    if not args.input:
        detected_input = find_default_input_file(search_folder=args.input_folder)
        args.input = str(detected_input)

    return args


def main() -> None:
    args = parse_args()

    configure_threads(args.threads)
    configure_ffmpeg(args.ffmpeg_path, args.ffprobe_path)
    ensure_ffmpeg_available(args.ffmpeg_path)

    input_path = Path(args.input).expanduser().resolve()

    output_txt = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_txt_output_path(input_path).resolve()
    )

    output_srt = (
        Path(args.srt_output).expanduser().resolve()
        if args.srt_output
        else None
    )

    wav_path = (
        Path(args.wav_output).expanduser().resolve()
        if args.wav_output
        else default_wav_output_path(input_path).resolve()
    )

    device = resolve_device(args.device)
    compute_type = resolve_compute_type(device, args.compute_type)
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
        normalize_audio=not args.no_normalize,
        reuse_existing_wav=not args.no_reuse_wav,
    )

    print(f"⏱️  Audio duration: {audio_duration:.1f}s")

    model = load_model(
        model_source=model_source,
        device=device,
        compute_type=compute_type,
        threads=args.threads,
    )

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
        print("Quality check: passed")
    else:
        print("Quality check: review recommended")

        for issue in report.issues:
            print(f"- {issue}")


if __name__ == "__main__":
    main()
