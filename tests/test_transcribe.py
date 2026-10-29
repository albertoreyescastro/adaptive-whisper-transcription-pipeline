import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import transcribe


class FingerprintAndCacheTests(unittest.TestCase):
    def test_fingerprint_tracks_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            path.write_bytes(b"alpha")
            first = transcribe.input_fingerprint(path)
            path.write_bytes(b"bravo")
            second = transcribe.input_fingerprint(path)
            self.assertNotEqual(first, second)
            self.assertEqual(len(first), 12)

    def test_prepared_wav_cache_identity_tracks_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.wav"
            path.write_bytes(b"same-input")
            normalized = transcribe.prepared_wav_cache_filename(path, normalize_audio=True)
            raw = transcribe.prepared_wav_cache_filename(path, normalize_audio=False)
            self.assertNotEqual(normalized, raw)
            self.assertIn(transcribe.PREPROCESSING_PROFILE_VERSION, normalized)
            self.assertIn("_norm_", normalized)
            self.assertIn("_raw_", raw)

    def test_explicit_wav_reuse_requires_matching_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "input.wav"
            prepared = tmp / "custom.wav"
            source.write_bytes(b"source")
            prepared.write_bytes(b"prepared")

            self.assertFalse(
                transcribe.can_reuse_prepared_wav(source, prepared, normalize_audio=True)
            )

            meta = transcribe.prepared_wav_metadata_path(prepared)
            meta.write_text(
                json.dumps(transcribe.preprocessing_identity(source, normalize_audio=False)),
                encoding="utf-8",
            )
            self.assertFalse(
                transcribe.can_reuse_prepared_wav(source, prepared, normalize_audio=True)
            )
            self.assertTrue(
                transcribe.can_reuse_prepared_wav(source, prepared, normalize_audio=False)
            )

    def test_default_output_identity_avoids_same_stem_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = tmp / "a" / "interview.wav"
            b = tmp / "b" / "interview.wav"
            a.parent.mkdir()
            b.parent.mkdir()
            a.write_bytes(b"first")
            b.write_bytes(b"second")
            self.assertNotEqual(
                transcribe.default_txt_output_path(a).name,
                transcribe.default_txt_output_path(b).name,
            )


class PathSafetyTests(unittest.TestCase):
    def test_input_cannot_equal_txt_output(self):
        path = Path("same.wav")
        with self.assertRaises(ValueError):
            transcribe.validate_distinct_paths(path, path, None, Path("prepared.wav"))

    def test_input_cannot_equal_wav_output(self):
        path = Path("same.wav")
        with self.assertRaises(ValueError):
            transcribe.validate_distinct_paths(path, Path("out.txt"), None, path)

    def test_txt_cannot_equal_srt_output(self):
        output = Path("same.txt")
        with self.assertRaises(ValueError):
            transcribe.validate_distinct_paths(
                Path("input.wav"), output, output, Path("prepared.wav")
            )

    def test_output_cannot_collide_with_wav_metadata_sidecar(self):
        wav = Path("prepared.wav")
        metadata = transcribe.prepared_wav_metadata_path(wav)
        with self.assertRaises(ValueError):
            transcribe.validate_distinct_paths(Path("input.wav"), metadata, None, wav)

    def test_transcription_helper_rejects_wav_txt_collision(self):
        wav = Path("prepared.wav")
        with self.assertRaises(ValueError):
            transcribe.validate_transcription_paths(wav, wav, None)


class AudioUtilityTests(unittest.TestCase):
    def test_parse_max_volume(self):
        stderr = "[Parsed_volumedetect_0] max_volume: -3.4 dB\n"
        self.assertEqual(transcribe._parse_max_volume_db(stderr), -3.4)

    def test_parse_silent_volume(self):
        stderr = "[Parsed_volumedetect_0] max_volume: -inf dB\n"
        self.assertIsNone(transcribe._parse_max_volume_db(stderr))

    def test_prepare_wav_rejects_input_output_collision_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.wav"
            path.write_bytes(b"input")
            with patch("transcribe.subprocess.run") as run:
                with self.assertRaises(ValueError):
                    transcribe.prepare_wav(path, path)
                run.assert_not_called()
            self.assertTrue(path.exists())

    def test_ffmpeg_failure_preserves_existing_final_wav_and_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "source.wav"
            final_wav = tmp / "prepared.wav"
            source.write_bytes(b"source-data")
            final_wav.write_bytes(b"previous-good-wav")

            with patch.object(transcribe, "measure_peak_db", return_value=None), patch(
                "transcribe.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["ffmpeg"]),
            ):
                with self.assertRaises(RuntimeError):
                    transcribe.prepare_wav(
                        source,
                        final_wav,
                        reuse_existing_wav=False,
                        ffmpeg_binary="ffmpeg",
                        ffprobe_binary="ffprobe",
                    )

            self.assertEqual(source.read_bytes(), b"source-data")
            self.assertEqual(final_wav.read_bytes(), b"previous-good-wav")
            self.assertFalse(any(tmp.glob("*.tmp.wav")))

    def test_volumedetect_failure_cleans_temporary_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "source.wav"
            final_wav = tmp / "prepared.wav"
            temp_wav = tmp / ".prepared.test.tmp.wav"
            source.write_bytes(b"source-data")
            temp_wav.write_bytes(b"temporary")

            with patch.object(
                transcribe, "_temporary_wav_path", return_value=temp_wav
            ), patch.object(
                transcribe,
                "measure_peak_db",
                side_effect=subprocess.CalledProcessError(1, ["ffmpeg", "volumedetect"]),
            ):
                with self.assertRaises(RuntimeError):
                    transcribe.prepare_wav(
                        source,
                        final_wav,
                        reuse_existing_wav=False,
                        ffmpeg_binary="ffmpeg",
                        ffprobe_binary="ffprobe",
                    )

            self.assertTrue(source.exists())
            self.assertFalse(final_wav.exists())
            self.assertFalse(temp_wav.exists())

    def test_invalid_cached_wav_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "source.wav"
            prepared = tmp / "prepared.wav"
            source.write_bytes(b"source-data")
            prepared.write_bytes(b"not-a-real-wav")
            metadata = transcribe.prepared_wav_metadata_path(prepared)
            metadata.write_text(
                json.dumps(transcribe.preprocessing_identity(source, normalize_audio=True)),
                encoding="utf-8",
            )

            probe_failure = subprocess.CalledProcessError(1, ["ffprobe"])

            def fake_ffmpeg_run(command, **_kwargs):
                Path(command[-1]).write_bytes(b"rebuilt-wav")
                return SimpleNamespace(returncode=0)

            with patch.object(
                transcribe,
                "probe_audio_duration",
                side_effect=[probe_failure, 12.5],
            ), patch.object(
                transcribe, "measure_peak_db", return_value=None
            ), patch(
                "transcribe.subprocess.run", side_effect=fake_ffmpeg_run
            ):
                duration = transcribe.prepare_wav(
                    source,
                    prepared,
                    reuse_existing_wav=True,
                    ffmpeg_binary="ffmpeg",
                    ffprobe_binary="ffprobe",
                )

            self.assertEqual(duration, 12.5)
            self.assertTrue(prepared.exists())
            self.assertEqual(prepared.read_bytes(), b"rebuilt-wav")
            self.assertTrue(metadata.exists())
            self.assertTrue(
                transcribe.can_reuse_prepared_wav(source, prepared, normalize_audio=True)
            )


class RuntimeResolutionTests(unittest.TestCase):
    def test_normal_cpu_auto_compute_is_int8(self):
        self.assertEqual(transcribe.resolve_compute_type("cpu", "auto"), "int8")

    def test_cuda_fallback_resets_explicit_compute_type_to_int8(self):
        self.assertEqual(
            transcribe.resolve_compute_type(
                "cpu", "float16", safe_cpu_fallback=True
            ),
            "int8",
        )

    def test_explicit_cpu_compute_type_is_preserved_without_cuda_fallback(self):
        self.assertEqual(
            transcribe.resolve_compute_type(
                "cpu", "float32", safe_cpu_fallback=False
            ),
            "float32",
        )


class FormattingTests(unittest.TestCase):
    def test_srt_timestamp(self):
        self.assertEqual(transcribe.srt_timestamp(61.234), "00:01:01,234")

    def test_duration_format(self):
        self.assertEqual(transcribe.format_duration(65), "01:05")
        self.assertEqual(transcribe.format_duration(3661), "01:01:01")


class PresetTests(unittest.TestCase):
    def test_auto_sequence_preserves_fallback_order(self):
        self.assertEqual(
            transcribe.preset_sequence("auto"),
            ["balanced", "robust", "no-vad"],
        )

    def test_gpu_high_accuracy_is_distinct(self):
        balanced = transcribe.get_preset_options("balanced")
        thorough = transcribe.get_preset_options("gpu-high-accuracy")
        self.assertGreater(thorough["beam_size"], balanced["beam_size"])


class QualityTests(unittest.TestCase):
    def test_empty_transcription_fails(self):
        report = transcribe.assess_transcription([], 120.0)
        self.assertFalse(report.passed)

    def test_normal_short_sample_passes(self):
        records = [
            transcribe.SegmentRecord(0.0, 5.0, "This is a normal sample transcription."),
            transcribe.SegmentRecord(5.0, 10.0, "It contains several different words."),
        ]
        self.assertTrue(transcribe.assess_transcription(records, 10.0).passed)

    def test_unexpectedly_short_output_is_flagged(self):
        records = [transcribe.SegmentRecord(0.0, 25.0, "Too short")]
        report = transcribe.assess_transcription(records, 30.0)
        self.assertIn("transcription is unexpectedly short", report.issues)

    def test_early_stop_is_flagged(self):
        text = "This is a sufficiently long sentence to avoid the short output gate. " * 3
        records = [transcribe.SegmentRecord(0.0, 20.0, text)]
        report = transcribe.assess_transcription(records, 120.0)
        self.assertIn("transcription appears to stop too early", report.issues)

    def test_repeated_segments_are_flagged(self):
        records = [
            transcribe.SegmentRecord(float(i), float(i + 1), "repeated line")
            for i in range(10)
        ]
        report = transcribe.assess_transcription(records, 10.0)
        self.assertIn("many consecutive repeated segments detected", report.issues)

    def test_low_lexical_diversity_is_flagged(self):
        repeated = "word " * 90
        records = [transcribe.SegmentRecord(0.0, 10.0, repeated)]
        report = transcribe.assess_transcription(records, 10.0)
        self.assertIn("very low lexical diversity; possible repetition loop", report.issues)


class CliTests(unittest.TestCase):
    def test_explicit_cli_arguments_are_strict(self):
        with self.assertRaises(SystemExit):
            transcribe.parse_args(["--input", "example.wav", "--langauge", "en"])

    def test_explicit_input_parses_without_file_access(self):
        args = transcribe.parse_args(["--input", "example.wav", "--language", "en"])
        self.assertEqual(args.language, "en")


class AtomicFallbackTests(unittest.TestCase):
    def test_failed_first_attempt_does_not_replace_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            final_txt = tmp / "transcript.txt"
            final_srt = tmp / "subtitles.srt"
            final_txt.write_text("previous-good-txt", encoding="utf-8")
            final_srt.write_text("previous-good-srt", encoding="utf-8")
            wav = tmp / "prepared.wav"
            wav.write_bytes(b"placeholder")
            calls = []

            def fake_transcribe_once(**kwargs):
                calls.append(kwargs["preset"])
                Path(kwargs["output_txt"]).write_text(kwargs["preset"], encoding="utf-8")
                Path(kwargs["output_srt"]).write_text(
                    f"srt-{kwargs['preset']}", encoding="utf-8"
                )
                if kwargs["preset"] == "balanced":
                    return []
                return [
                    transcribe.SegmentRecord(
                        0, 10, "a sufficiently normal transcription sample"
                    )
                ]

            with patch.object(
                transcribe, "transcribe_once", side_effect=fake_transcribe_once
            ):
                preset, report = transcribe.transcribe_with_fallbacks(
                    model=object(),
                    wav_path=wav,
                    output_txt=final_txt,
                    output_srt=final_srt,
                    requested_preset="auto",
                    language="en",
                    task="transcribe",
                    include_timestamps_in_txt=False,
                    audio_duration=10.0,
                )

            self.assertEqual(calls, ["balanced", "robust"])
            self.assertEqual(preset, "robust")
            self.assertTrue(report.passed)
            self.assertEqual(final_txt.read_text(encoding="utf-8"), "robust")
            self.assertEqual(final_srt.read_text(encoding="utf-8"), "srt-robust")
            self.assertFalse(any(tmp.glob("*.tmp")))

    def test_transcription_rejects_txt_srt_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wav = tmp / "prepared.wav"
            wav.write_bytes(b"placeholder")
            output = tmp / "same.txt"
            with self.assertRaises(ValueError):
                transcribe.transcribe_with_fallbacks(
                    model=object(),
                    wav_path=wav,
                    output_txt=output,
                    output_srt=output,
                    requested_preset="fast",
                    language="en",
                    task="transcribe",
                    include_timestamps_in_txt=False,
                    audio_duration=10.0,
                )

    def test_attempt_temp_name_cannot_reuse_legacy_predictable_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            final_txt = tmp / "result.txt"
            legacy_collision = tmp / ".result.txt.attempt-1-fast.tmp"
            legacy_collision.write_text("original-input", encoding="utf-8")

            temp_path = transcribe._temporary_attempt_path(final_txt, 1, "fast")
            self.assertIsNotNone(temp_path)
            try:
                self.assertNotEqual(temp_path, legacy_collision)
                self.assertEqual(
                    legacy_collision.read_text(encoding="utf-8"),
                    "original-input",
                )
            finally:
                transcribe._cleanup_attempt(temp_path)

            self.assertTrue(legacy_collision.exists())


class ProgressTests(unittest.TestCase):
    def test_final_progress_uses_last_segment_not_forced_audio_duration(self):
        class FakeModel:
            def transcribe(self, _path, **_kwargs):
                segments = [SimpleNamespace(start=0.0, end=10.0, text="hello world")]
                return iter(segments), SimpleNamespace(language="en", language_probability=1.0)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            calls = []
            with patch.object(
                transcribe,
                "print_progress_bar",
                side_effect=lambda current, total, **kwargs: calls.append((current, total)),
            ):
                transcribe.transcribe_once(
                    model=FakeModel(),
                    wav_path=tmp / "prepared.wav",
                    output_txt=tmp / "out.txt",
                    output_srt=None,
                    preset="fast",
                    language="en",
                    task="transcribe",
                    include_timestamps_in_txt=False,
                    audio_duration=100.0,
                )
            self.assertTrue(calls)
            self.assertEqual(calls[-1], (10.0, 100.0))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required")
class FFmpegIntegrationTests(unittest.TestCase):
    def test_real_ffmpeg_preprocessing_is_mono_16khz_pcm16_and_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "source.wav"
            prepared = tmp / "prepared.wav"
            subprocess.run(
                [
                    shutil.which("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-ar", "44100", "-ac", "2", str(source),
                ],
                check=True,
            )

            duration = transcribe.prepare_wav(
                source,
                prepared,
                normalize_audio=True,
                ffmpeg_binary=shutil.which("ffmpeg"),
                ffprobe_binary=shutil.which("ffprobe"),
            )
            self.assertGreater(duration, 0.9)
            self.assertTrue(prepared.is_file())
            self.assertTrue(transcribe.prepared_wav_metadata_path(prepared).is_file())
            self.assertTrue(
                transcribe.can_reuse_prepared_wav(source, prepared, normalize_audio=True)
            )
            self.assertFalse(
                transcribe.can_reuse_prepared_wav(source, prepared, normalize_audio=False)
            )

            result = subprocess.run(
                [
                    shutil.which("ffprobe"), "-v", "error", "-select_streams", "a:0",
                    "-show_entries", "stream=sample_rate,channels,codec_name",
                    "-of", "json", str(prepared),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            stream = json.loads(result.stdout)["streams"][0]
            self.assertEqual(stream["sample_rate"], "16000")
            self.assertEqual(stream["channels"], 1)
            self.assertEqual(stream["codec_name"], "pcm_s16le")


if __name__ == "__main__":
    unittest.main()
