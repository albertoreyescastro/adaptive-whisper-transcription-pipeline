# Troubleshooting

## FFmpeg was not found

Install FFmpeg and make sure it is available from your system PATH.

You can also pass explicit paths:

```bash
python transcribe.py --input "C:\path\to\audio.m4a" --ffmpeg-path "C:\path\to\ffmpeg.exe" --ffprobe-path "C:\path\to\ffprobe.exe"
```

## The script cannot find an input file in Spyder

Create a folder called `whisper_input` on your Desktop or OneDrive Desktop and place exactly one supported audio/video file inside it.

Example:

```text
C:\Users\<your_user>\OneDrive\Escritorio\whisper_input
```

The script deliberately does not scan your full Desktop or Spyder folder.

## Multiple input files were found

Leave only one supported file inside `whisper_input`, or pass the file explicitly:

```bash
python transcribe.py --input "C:\path\to\audio.m4a"
```

## CUDA is not detected

Check whether PyTorch can see your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU detected')"
```

Check whether CTranslate2 can see CUDA:

```bash
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

If CUDA is not available, the script falls back to CPU mode using `medium` and `int8`.

## The transcription stops too early

Try disabling VAD:

```bash
python transcribe.py --input "C:\path\to\audio.m4a" --preset no-vad
```

## The transcription has repeated phrases

Try robust mode:

```bash
python transcribe.py --input "C:\path\to\audio.m4a" --preset robust
```

## The detected language is wrong

Force the language:

```bash
python transcribe.py --input "C:\path\to\audio.m4a" --language es
```

or:

```bash
python transcribe.py --input "C:\path\to\audio.m4a" --language en
```
