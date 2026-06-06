# Adaptive Whisper Transcription Pipeline

A portable CPU/GPU transcription workflow for audio and video files using `faster-whisper`.

This project provides a practical transcription pipeline with automatic hardware selection, safe input detection, audio normalisation, VAD presets, anti-repetition safeguards, streaming text output, a visual progress bar and optional SRT subtitle generation.

## Demo

A runnable Kaggle demo notebook is available here:

[Open the Kaggle demo](https://www.kaggle.com/code/albertoreyescastro20/adaptive-whisper-transcription-pipeline-demo)

The demo shows the pipeline workflow in a notebook environment, while this repository contains the reusable script version, configuration files and project documentation.

## Features

- CPU/GPU-aware execution.
- Automatic model selection:
  - CPU → `medium` with `int8`
  - CUDA GPU → `large-v3` with `float16`
- Safe Spyder-friendly input workflow using a dedicated `whisper_input` folder.
- No accidental scanning of the full Desktop, Spyder folder or current working directory.
- Audio/video input support through FFmpeg and `pydub`.
- Automatic conversion to WAV 16 kHz mono PCM16.
- Unique WAV cache files using a fingerprint of the input file.
- Presets for different transcription behaviours:
  - `fast`
  - `balanced`
  - `robust`
  - `no-vad`
  - `gpu-high-accuracy`
  - `auto`
- Optional timestamped TXT output.
- Optional SRT subtitle output.
- Basic quality checks for empty output, early stopping and repetition loops.

## Project motivation

This tool was designed as a practical transcription workflow for long-form academic, scientific and outreach audio/video material. It is intended to be usable both as a command-line tool and as a Spyder-friendly script on Windows.

## Installation

Create and activate a Python environment, then install the requirements:

```bash
pip install -r requirements.txt
```

You also need FFmpeg installed and available from your system PATH.

On Windows, if FFmpeg is not available globally, you can pass explicit paths:

```bash
python transcribe.py --input "C:\path\to\audio.m4a" --ffmpeg-path "C:\path\to\ffmpeg.exe" --ffprobe-path "C:\path\to\ffprobe.exe"
```

## Basic usage

```bash
python transcribe.py --input "C:\path\to\audio.m4a"
```

Force Spanish:

```bash
python transcribe.py --input "C:\path\to\audio.m4a" --language es
```

Force English:

```bash
python transcribe.py --input "C:\path\to\audio.m4a" --language en
```

Use robust mode:

```bash
python transcribe.py --input "C:\path\to\audio.m4a" --preset robust
```

Generate an SRT subtitle file:

```bash
python transcribe.py --input "C:\path\to\video.mp4" --srt-output "subtitles.srt"
```

Include timestamps in the TXT output:

```bash
python transcribe.py --input "C:\path\to\audio.m4a" --timestamps
```

## Spyder-friendly workflow

Create a folder called `whisper_input` on your Desktop or OneDrive Desktop.

Example:

```text
C:\Users\<your_user>\OneDrive\Escritorio\whisper_input
```

Put exactly one supported audio/video file inside that folder.

Then run `transcribe.py` from Spyder. The script will only auto-detect a file from that dedicated folder.

This prevents accidental transcription of random files from the Desktop, Downloads folder, Spyder folder or current working directory.

## Output structure

By default, the script creates an output folder on the Desktop:

```text
whisper_output/
  input_filename_transcription.txt
  cache/
    input_filename_fingerprint_prepared.wav
```

If `--srt-output` is provided, the SRT file is saved to the path you choose.

## Supported input formats

The script supports common audio/video formats readable by FFmpeg, including:

```text
.mp3, .mp4, .m4a, .wav, .mov, .mkv, .aac, .flac, .ogg, .webm, .wma
```

## Presets

| Preset | Purpose |
|---|---|
| `fast` | Faster transcription with lighter settings. |
| `balanced` | Default balanced mode. |
| `robust` | More conservative mode for difficult audio, repetition loops or hallucinations. |
| `no-vad` | Useful when VAD cuts off speech after long silences. |
| `gpu-high-accuracy` | Intended for CUDA GPU usage. |
| `auto` | Runs `balanced`, then retries with `robust` and `no-vad` if quality checks fail. |

## Important notes

The visual progress bar is based on the timestamp of the latest transcribed segment. It is an estimate of audio coverage, not a low-level progress indicator from the Whisper model itself.

Do not commit private audio, video, cache files or real transcripts to GitHub.

## Recommended repository structure

```text
adaptive-whisper-transcription-pipeline/
  transcribe.py
  README.md
  requirements.txt
  environment.yml
  .gitignore
  docs/
    TROUBLESHOOTING.md
  examples/
    README.md
```

## Author

Alberto Reyes Castro

