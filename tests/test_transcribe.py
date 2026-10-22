import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import transcribe


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_tracks_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            path.write_bytes(b"alpha")
            first = transcribe.input_fingerprint(path)
            path.write_bytes(b"bravo")
            second = transcribe.input_fingerprint(path)
            self.assertNotEqual(first, second)
            self.assertEqual(len(first), 12)


class FormattingTests(unittest.TestCase):
    def test_srt_timestamp(self):
        self.assertEqual(transcribe.srt_timestamp(61.234), "00:01:01,234")

    def test_duration_format(self):
        self.assertEqual(transcribe.format_duration(65), "01:05")
        self.assertEqual(transcribe.format_duration(3661), "01:01:01")


class PresetTests(unittest.TestCase):
    def test_auto_sequence_preserves_original_fallback_order(self):
        self.assertEqual(transcribe.preset_sequence("auto"), ["balanced", "robust", "no-vad"])

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


class CliTests(unittest.TestCase):
    def test_explicit_cli_arguments_are_strict(self):
        with self.assertRaises(SystemExit):
            transcribe.parse_args(["--input", "example.wav", "--langauge", "en"])

    def test_explicit_input_parses_without_file_access(self):
        args = transcribe.parse_args(["--input", "example.wav", "--language", "en"])
        self.assertEqual(args.language, "en")


class AtomicFallbackTests(unittest.TestCase):
    def test_failed_first_attempt_does_not_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            final_txt = tmp / "transcript.txt"
            final_txt.write_text("previous-good-output", encoding="utf-8")
            wav = tmp / "prepared.wav"
            wav.write_bytes(b"placeholder")
            calls = []

            def fake_transcribe_once(**kwargs):
                calls.append(kwargs["preset"])
                Path(kwargs["output_txt"]).write_text(kwargs["preset"], encoding="utf-8")
                if kwargs["preset"] == "balanced":
                    return []
                return [transcribe.SegmentRecord(0, 10, "a sufficiently normal transcription sample")]

            with patch.object(transcribe, "transcribe_once", side_effect=fake_transcribe_once):
                preset, report = transcribe.transcribe_with_fallbacks(
                    model=object(),
                    wav_path=wav,
                    output_txt=final_txt,
                    output_srt=None,
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
            self.assertFalse(any(tmp.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
