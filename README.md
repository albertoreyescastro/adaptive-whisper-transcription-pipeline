# Adaptive Whisper Transcription Pipeline

[![CI](https://github.com/albertoreyescastro/adaptive-whisper-transcription-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/albertoreyescastro/adaptive-whisper-transcription-pipeline/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A portable CPU/GPU transcription workflow for long-form-oriented audio and video processing using `faster-whisper` and CTranslate2.

The project keeps a deliberately simple shape: one reusable Python entry point, explicit FFmpeg preprocessing, hardware-aware defaults, bounded fallback presets, transcript/subtitle outputs and lightweight quality sanity checks. It is designed for a normal CLI while remaining friendly to a Windows/Spyder workflow.

> **Portfolio scope:** this is a reusable engineering project inspired by practical scientific/outreach transcription workflows. It is not a claim that this exact `faster-whisper` implementation was the production stack used during any specific internship or institution.

## Live demo

A lightweight Kaggle demonstration is available here:

**[Open the Kaggle demo](https://www.kaggle.com/code/albertoreyescastro20/adaptive-whisper-transcription-pipeline-demo)**

The Kaggle notebook intentionally uses a smaller CPU model and a short public sample so it can run quickly in a hosted notebook environment. The GitHub repository remains the canonical reusable implementation.

## What this project demonstrates

- Hardware-aware inference with CPU/CUDA selection.
- CTranslate2 compute-type selection (`int8` on CPU, `float16` on CUDA by default).
- Safe CUDA→CPU fallback: an explicit CUDA-oriented compute type is not carried into the CPU fallback.
- Long-form-oriented FFmpeg preprocessing without loading the entire media file into a Python audio object.
- Streaming peak normalisation that preserves the intent of the original `pydub.normalize()` step.
- Verified WAV caching using a SHA-256 source fingerprint, an explicit preprocessing profile and a metadata sidecar that also protects custom `--wav-output` paths.
- Atomic WAV preparation: FFmpeg writes to a temporary file and the final WAV is promoted only after FFprobe validation succeeds.
- Safe local input discovery restricted to a dedicated `whisper_input` folder.
- Preset-based decoding and bounded automatic fallback (`balanced → robust → no-vad`).
- TXT and optional SRT generation with streaming writes.
- Lightweight structural failure detection for empty output, early stopping and repetition loops.
- Per-file atomic fallback promotion so a failed retry does not overwrite the last completed transcript/subtitle file.
- Unit-tested deterministic behaviour and GitHub Actions CI across Python 3.10–3.12.
- CI dependency/import and FFmpeg/FFprobe preprocessing smoke checks without downloading Whisper model weights.

## Pipeline

```text
Audio / video input
        │
        ▼
Safe input resolution
        │
        ▼
SHA-256 source identity + preprocessing profile + sidecar verification
        │
        ▼
FFmpeg → peak normalisation → 16 kHz mono PCM16 WAV
        │
        ▼
Hardware + compute selection
        │
        ▼
faster-whisper / CTranslate2
        │
        ▼
Preset sanity checks + bounded fallback
        │
        ├── TXT transcript
        └── optional SRT subtitles
```

## Default hardware policy

| Hardware | Model | Compute type |
|---|---|---|
| CPU | `medium` | `int8` |
| CUDA GPU | `large-v3` | `float16` |

These are defaults, not hard requirements. You can override the model, device and compute type from the CLI.

If CUDA is explicitly requested but unavailable, the pipeline falls back to CPU and uses CPU-safe `int8` rather than retaining a potentially inappropriate explicit CUDA-oriented compute type.

## Presets

| Preset | Intended use |
|---|---|
| `fast` | Lighter decoding for quicker runs. |
| `balanced` | Default general-purpose decoding. |
| `robust` | More conservative settings for repetition/hallucination-prone audio. |
| `no-vad` | Useful when VAD removes speech after long silences. |
| `gpu-high-accuracy` | Wider-beam GPU-oriented decoding; more compute, not a guaranteed accuracy improvement. |
| `auto` | Starts at `balanced` and retries with `robust` then `no-vad` only when sanity checks fail. |

The quality gate is deliberately a **sanity check**, not an accuracy metric. It can catch obvious structural failures, but it does not replace WER/CER evaluation or human review. Some heuristics only apply above duration, segment-count or word-count thresholds, so a pass on a short sample is intentionally a weak structural signal.

## Installation

### Option A — pip

Install Python 3.10–3.12 and FFmpeg, then:

```powershell
pip install -r requirements.txt
```

FFmpeg and FFprobe must be available on `PATH`, or passed explicitly with `--ffmpeg-path` and `--ffprobe-path`.

### Option B — Conda

```powershell
conda env create -f environment.yml
conda activate adaptive-whisper-transcription
```

GPU runtime compatibility depends on the installed CTranslate2 release and your NVIDIA/CUDA/cuDNN environment. Check the current upstream faster-whisper/CTranslate2 installation guidance rather than relying on a hard-coded CUDA version.

## Basic usage

```powershell
python transcribe.py --input "C:\path\to\audio.m4a"
```

Force Spanish:

```powershell
python transcribe.py --input "C:\path\to\audio.m4a" --language es
```

Use robust mode:

```powershell
python transcribe.py --input "C:\path\to\audio.m4a" --preset robust
```

Generate SRT subtitles:

```powershell
python transcribe.py --input "C:\path\to\video.mp4" --srt-output "subtitles.srt"
```

Include timestamps in the TXT output:

```powershell
python transcribe.py --input "C:\path\to\audio.m4a" --timestamps
```

Explicit FFmpeg paths on Windows:

```powershell
python transcribe.py --input "C:\path\to\audio.m4a" --ffmpeg-path "C:\path\to\ffmpeg.exe" --ffprobe-path "C:\path\to\ffprobe.exe"
```

## Safe Spyder workflow

Create a folder named `whisper_input` on your Desktop or OneDrive Desktop and place exactly one supported audio/video file inside it.

Example:

```text
C:\Users\<your_user>\OneDrive\Escritorio\whisper_input
```

When no `--input` is provided, the script only checks those dedicated input folders. It deliberately does **not** scan the full Desktop, Downloads folder, current working directory or Spyder project folder.

CLI parsing is strict: misspelled options fail fast. Only arguments injected by a detected Spyder runtime are tolerated.

## Outputs and cache identity

By default:

```text
whisper_output/
  input_filename_<sha256-prefix>_transcription.txt
  cache/
    input_filename_<sha256-prefix>_prep-v3_norm_prepared.wav
    input_filename_<sha256-prefix>_prep-v3_norm_prepared.wav.meta.json
```

When `--no-normalize` is used, the default cache filename uses `raw` instead of `norm`. Every prepared WAV also receives a `.meta.json` sidecar containing the source fingerprint and preprocessing identity. Reuse is allowed only when that sidecar matches, including for an explicit `--wav-output` path.

Default transcript filenames include the source fingerprint, so two different inputs with the same stem do not silently overwrite each other. Explicit input/TXT/SRT/WAV paths and the WAV metadata sidecar are validated as distinct before any write occurs.

WAV preparation writes to a temporary file and promotes it only after FFmpeg and FFprobe succeed, so a failed conversion does not destroy the previous prepared WAV or the source input.

SRT output is optional and written only when requested. Fallback TXT/SRT attempts use temporary files and each final file is promoted atomically after an attempt completes. TXT and SRT promotion is per-file rather than a multi-file transaction.

## Supported media formats

The pipeline accepts common FFmpeg-readable formats, including:

```text
.mp3 .mp4 .m4a .wav .mov .mkv .aac .flac .ogg .webm .wma
```

## Testing

The test suite intentionally avoids downloading large Whisper models. It covers deterministic pipeline behaviour including:

- source fingerprinting and collision-resistant default output naming,
- preprocessing-cache identity and explicit-WAV sidecar validation,
- input/output path-collision rejection,
- atomic FFmpeg preparation failure handling,
- FFmpeg peak parsing,
- strict CLI validation,
- CPU/CUDA compute-type resolution,
- preset behaviour,
- timestamp formatting,
- all structural quality-gate branches,
- progress coverage semantics,
- atomic fallback output handling for TXT/SRT,
- a real FFmpeg/FFprobe preprocessing smoke test.

Run locally:

```powershell
python -m compileall -q transcribe.py tests
python -m unittest discover -s tests -v
```

GitHub Actions runs these checks on Python 3.10, 3.11 and 3.12. CI installs `requirements.txt`, verifies `faster_whisper`/`ctranslate2` imports, ensures FFmpeg/FFprobe are available (installing FFmpeg only when needed) and executes a synthetic-audio preprocessing smoke test. It deliberately avoids downloading Whisper model weights.

A real ASR transcription smoke test still requires a `faster-whisper` model download.

## Scientific outreach context

The engineering motivation came from practical transcription of audiovisual scientific/outreach material.

One relevant earlier context is the Instituto de Astrofísica de Canarias (IAC), where the internship transcription tooling was Python-based and used a different software stack, including `SpeechRecognition`, to process audiovisual material and support segmented multilingual subtitles. This repository is a later reusable `faster-whisper` re-engineering of that problem; it is not the original internship codebase.

- [IAC official website](https://www.iac.es/)
- [IAC news and outreach archive](https://www.iac.es/es/divulgacion/noticias)

No private institutional audio, video or real internal transcripts are included in this repository.

## Limitations

- The sanity checks do not measure word-level accuracy.
- On short samples, some threshold-based sanity heuristics do not apply.
- Domain names, acronyms, noisy speech and overlapping speakers may still require review.
- `large-v3` can require substantial GPU memory.
- The first run may download model weights from Hugging Face.
- Peak normalisation is pragmatic preprocessing, not a substitute for domain-specific audio conditioning.
- Streaming FFmpeg preprocessing does not imply end-to-end constant-memory inference.
- No diarisation is performed.
- Dependency compatibility is bounded by declared version ranges rather than a fully locked environment.
- Final TXT/SRT promotion is atomic per file, not a transactional two-file commit.

## Privacy

Do not commit private recordings, client/university media, generated cache files or real transcripts. The repository `.gitignore` excludes common audio/video formats, prepared-WAV metadata sidecars and generated transcription artifacts by default.

## Repository layout

```text
adaptive-whisper-transcription-pipeline/
├── transcribe.py
├── tests/
│   └── test_transcribe.py
├── docs/
│   └── TROUBLESHOOTING.md
├── examples/
│   └── README.md
├── .github/workflows/
│   └── ci.yml
├── requirements.txt
├── environment.yml
├── .gitignore
└── LICENSE
```

## License

MIT. See [LICENSE](LICENSE).

## Author

Alberto Reyes Castro
