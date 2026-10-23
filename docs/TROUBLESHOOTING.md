# Troubleshooting

## FFmpeg or FFprobe was not found

Install FFmpeg and make sure both `ffmpeg` and `ffprobe` are available on your system `PATH`.

On Windows, you can also pass explicit paths in a single PowerShell/CMD-safe command:

```powershell
python transcribe.py --input "C:\path\to\audio.m4a" --ffmpeg-path "C:\path\to\ffmpeg.exe" --ffprobe-path "C:\path\to\ffprobe.exe"
```

## The script cannot find an input file in Spyder

Create a folder called `whisper_input` on your Desktop or OneDrive Desktop and place exactly one supported audio/video file inside it.

```text
C:\Users\<your_user>\OneDrive\Escritorio\whisper_input
```

The script deliberately does not scan your full Desktop, Downloads, current working directory or Spyder folder.

## Multiple input files were found

Leave only one supported file inside `whisper_input`, or pass the file explicitly:

```powershell
python transcribe.py --input "C:\path\to\audio.m4a"
```

## An option is rejected as unknown

Normal CLI execution is intentionally strict so spelling mistakes are not ignored. Check the available options:

```powershell
python transcribe.py --help
```

Spyder-injected runtime arguments are handled separately when Spyder is detected.

## CUDA is not detected

This project uses CTranslate2 as the hardware check; PyTorch is not required just to detect CUDA.

```powershell
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

If `--device auto` finds no CUDA device, the pipeline uses CPU with `medium` and `int8` by default.

If you explicitly request `--device cuda` but CUDA is unavailable, the pipeline falls back to CPU and resets the compute type to CPU-safe `int8`, even if a CUDA-oriented compute type such as `float16` was supplied.

## CTranslate2 reports a CUDA/cuDNN runtime error

GPU runtime requirements can change across CTranslate2 releases. Check the current faster-whisper and CTranslate2 installation/compatibility notes for the version you installed rather than relying on a hard-coded CUDA/cuDNN combination.

As a safe fallback:

```powershell
python transcribe.py --input "C:\path\to\audio.m4a" --device cpu --compute-type int8
```

## OpenMP or MKL duplicate-runtime error

The pipeline intentionally does **not** set `KMP_DUPLICATE_LIB_OK=TRUE` automatically because that can hide environment conflicts. Prefer fixing the duplicate runtime in the Python/Conda environment. If you use a temporary workaround for local debugging, treat it as an environment-specific measure rather than repository configuration.

## The transcription stops too early

Try disabling VAD:

```powershell
python transcribe.py --input "C:\path\to\audio.m4a" --preset no-vad
```

Or allow bounded automatic retry:

```powershell
python transcribe.py --input "C:\path\to\audio.m4a" --preset auto
```

## The transcription has repeated phrases

Try robust mode:

```powershell
python transcribe.py --input "C:\path\to\audio.m4a" --preset robust
```

## The detected language is wrong

Force a language code:

```powershell
python transcribe.py --input "C:\path\to\audio.m4a" --language es
```

or:

```powershell
python transcribe.py --input "C:\path\to\audio.m4a" --language en
```

## A cached WAV seems wrong

Prepared WAV cache filenames include:

- a SHA-256 fingerprint of the source media,
- a preprocessing profile version,
- whether peak normalisation was enabled.

This prevents a normalised run from silently reusing an unnormalised cached WAV, or vice versa.

To force regeneration anyway:

```powershell
python transcribe.py --input "C:\path\to\audio.m4a" --no-reuse-wav
```

## The quality sanity check passes but words are still wrong

That is expected: the built-in check detects obvious structural failures such as empty output, early stopping and repetition loops. It does **not** measure WER/CER and should not be interpreted as a transcription-accuracy guarantee.

The exact checks are threshold-dependent; on a very short sample, some long-duration or high-segment-count heuristics may not apply.
