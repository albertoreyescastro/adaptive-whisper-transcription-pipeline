# Examples

This folder is reserved for safe, non-private examples.

Do not upload private audio, university recordings, client files or real internal transcripts.

Suggested safe examples:

- A short public-domain or clearly licensed audio sample.
- A synthetic test audio file generated locally.
- A redacted example transcript.
- A screenshot of the terminal progress bar without private filenames.

Example command using your own safe sample:

```powershell
python transcribe.py --input "C:\path\to\public_sample.wav" --language auto --preset auto
```

The repository intentionally does not ship `sample_audio.wav`; audio/video files are ignored by default to avoid accidentally publishing private media. For a runnable hosted example, use the Kaggle demo linked from the main README.
