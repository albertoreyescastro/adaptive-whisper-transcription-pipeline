# Adaptive Whisper Transcription Pipeline

[![CI](https://github.com/albertoreyescastro/adaptive-whisper-transcription-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/albertoreyescastro/adaptive-whisper-transcription-pipeline/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A portable CPU/GPU transcription workflow for long-form audio and video using `faster-whisper` and CTranslate2.

The project keeps a deliberately simple shape: one reusable Python entry point, explicit FFmpeg preprocessing, hardware-aware defaults, bounded fallback presets, transcript/subtitle outputs and lightweight quality sanity checks. It is designed to be useful from a normal CLI while remaining friendly to a Windows/Spyder workflow.

> **Portfolio scope:** this is a reusable engineering project inspired by practical scientific/outreach transcription workflows. It is not a claim that this exact `faster-whisper` implementation was the production stack used during any specific internship or institution.

## Live demo

A runnable Kaggle demonstration is available here:

**[Open the Kaggle demo](https://www.kaggle.com/code/albertoreyescastro20/adaptive-whisper-transcription-pipeline-demo)**

The Kaggle notebook is intentionally lighter than the local defaults so that it can run quickly in a hosted notebook environment. The GitHub repository remains the canonical reusable implementation.

## What this project demonstrates

- Hardware-aware inference with CPU/CUDA selection.
- CTranslate2 compute-type selection (`int8` on CPU, `float16` on CUDA by default).
- Long-form-friendly FFmpeg preprocessing without loading the entire media file into Python memory.
- Content-addressed WAV caching using a SHA-256 input fingerprint.
- Safe local input discovery restricted to a dedicated `whisper_input` folder.
- Preset-based decoding and bounded automatic fallback (`balanced → robust → no-vad`).
- TXT and optional SRT generation with streaming writes.
- Lightweight failure detection for empty output, early stopping and repetition loops.
- Atomic fallback outputs so a failed retry does not overwrite the last completed transcript.
- Unit-tested core behaviour and GitHub Actions CI across Python 3.10–3.12.

## Pipeline

```text
Audio / video input
        │
        ▼
Safe input resolution
        │
        ▼
SHA-256 cache identity
        │
        ▼
FFmpeg → 16 kHz mono PCM16 WAV
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

## Presets

| Preset | Intended use |
|---|---|
| `fast` | Lighter decoding for quicker runs. |
| `balanced` | Default general-purpose decoding. |
| `robust` | More conservative settings for repetition/hallucination-prone audio. |
| `no-vad` | Useful when VAD removes speech after long silences. |
| `gpu-high-accuracy` | Wider-beam GPU-oriented decoding; more compute, not a guaranteed accuracy improvement. |
| `auto` | Starts at `balanced` and retries with `robust` then `no-vad` only when sanity checks fail. |

The quality gate is intentionally described as a **sanity check**, not an accuracy metric. It can catch obvious failure modes but it does not replace WER/CER evaluation or human review.

## Installation

### Option A — pip

Install Python 3.10–3.12, FFmpeg, then:

```bash
pip install -r requirements.txt
```

FFmpeg and FFprobe must be available on `PATH`, or passed explicitly with `--ffmpeg-path` and `--ffprobe-path`.

### Option B — Conda

```bash
conda env create -f environment.yml
conda activate adaptive-whisper-transcription
```

## Basic usage

```bash
python transcribe.py --input "C:\path\to\audio.m4a"
```

Force Spanish:

```bash
python transcribe.py --input "C:\path\to\audio.m4a" --language es
```

Use robust mode:

```bash
python transcribe.py --input "C:\path\to\audio.m4a" --preset robust
```

Generate SRT subtitles:

```bash
python transcribe.py \
  --input "C:\path\to\video.mp4" \
  --srt-output "subtitles.srt"
```

Include timestamps in the TXT output:

```bash
python transcribe.py --input "C:\path\to\audio.m4a" --timestamps
```

Explicit FFmpeg paths on Windows:

```bash
python transcribe.py \
  --input "C:\path\to\audio.m4a" \
  --ffmpeg-path "C:\path\to\ffmpeg.exe" \
  --ffprobe-path "C:\path\to\ffprobe.exe"
```

## Safe Spyder workflow

Create a folder named `whisper_input` on your Desktop or OneDrive Desktop and place exactly one supported audio/video file inside it.

Example:

```text
C:\Users\<your_user>\OneDrive\Escritorio\whisper_input
```

When no `--input` is provided, the script only checks those dedicated input folders. It deliberately does **not** scan the full Desktop, Downloads folder, current working directory or Spyder project folder.

CLI parsing is strict: misspelled options fail fast. Only arguments injected by a detected Spyder runtime are tolerated.

## Outputs

By default:

```text
whisper_output/
  input_filename_transcription.txt
  cache/
    input_filename_<sha256-prefix>_prepared.wav
```

SRT output is optional and written only when requested.

Fallback attempts use temporary files and are promoted atomically only after an attempt completes. This prevents a failed retry from replacing a previously completed transcript.

## Supported media formats

The pipeline accepts common FFmpeg-readable formats, including:

```text
.mp3 .mp4 .m4a .wav .mov .mkv .aac .flac .ogg .webm .wma
```

## Testing

The CI suite intentionally avoids downloading large Whisper models. It checks deterministic pipeline logic such as cache identity, CLI validation, preset behaviour, formatting, quality gates and atomic fallback handling.

Run locally:

```bash
python -m compileall -q transcribe.py tests
python -m unittest discover -s tests -v
```

A real transcription smoke test still requires FFmpeg plus a `faster-whisper` model download.

## Scientific outreach context

The original motivation was practical long-form transcription for lectures, interviews, talks, seminars and scientific/outreach material. One relevant context is the Instituto de Astrofísica de Canarias (IAC), where earlier transcription work used a different software stack. This repository represents a later, reusable `faster-whisper` engineering implementation rather than a claim of byte-for-byte continuity with that earlier workflow.

- [IAC official website](https://www.iac.es/)
- [IAC news and outreach archive](https://www.iac.es/es/divulgacion/noticias)

No private institutional audio, video or real internal transcripts are included in this repository.

## Limitations

- The sanity checks do not measure word-level accuracy.
- Domain names, acronyms, noisy speech and overlapping speakers may still require review.
- `large-v3` can require substantial GPU memory.
- The first run may download model weights from Hugging Face.
- One-pass FFmpeg loudness normalisation is pragmatic preprocessing, not a substitute for domain-specific audio conditioning.
- No diarisation is performed.

## Privacy

Do not commit private recordings, client/university media, generated cache files or real transcripts. The repository `.gitignore` excludes common audio/video formats and generated transcription artifacts by default.

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
