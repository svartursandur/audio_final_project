"""Convert audio to 32 kHz mono FLAC at a fixed loudness using ffmpeg."""
import json
import re
import subprocess

SAMPLE_RATE = 32000
TARGET_LUFS = -23.0
PEAK_LIMIT = 0.89  # about -1 dBFS
MIN_MEASURABLE_LUFS = -70.0
TIMEOUT_SECONDS = 1200
BASE = ["ffmpeg", "-nostdin", "-hide_banner", "-protocol_whitelist", "file"]
RESAMPLE = f"aformat=channel_layouts=mono,aresample={SAMPLE_RATE}"


class AudioError(RuntimeError):
    pass


def probe_duration(path):
    """Seconds of audio in the file; raises AudioError if it has no audio stream."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise AudioError(f"ffprobe failed: {result.stderr.strip()[:200]}")
    info = json.loads(result.stdout)
    if not info.get("streams"):
        raise AudioError("no audio stream")
    return float(info.get("format", {}).get("duration", 0) or 0)


def measure_loudness(path):
    result = subprocess.run(
        BASE + ["-i", str(path), "-af", f"{RESAMPLE},loudnorm=I={TARGET_LUFS}:TP=-1:LRA=20:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise AudioError(f"loudness scan failed: {result.stderr.strip()[-200:]}")
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr)
    if not match:
        raise AudioError("loudness scan gave no result")
    value = json.loads(match.group(0))["input_i"]
    try:
        loudness = float(value)
    except ValueError as error:
        raise AudioError(f"unmeasurable loudness: {value}") from error
    if loudness < MIN_MEASURABLE_LUFS or loudness != loudness or loudness in (float("inf"), float("-inf")):
        raise AudioError(f"silent audio ({value} LUFS)")
    return loudness


def to_normalized_flac(source, destination):
    """Write a 32 kHz mono 16-bit FLAC; returns (input_lufs, duration_seconds)."""
    loudness = measure_loudness(source)
    gain = TARGET_LUFS - loudness
    chain = f"{RESAMPLE},volume={gain:.2f}dB,alimiter=limit={PEAK_LIMIT}:level=0"
    partial = destination.with_name(destination.name + ".part.flac")
    result = subprocess.run(
        BASE + ["-v", "error", "-y", "-i", str(source), "-af", chain, "-sample_fmt", "s16", "-c:a", "flac", str(partial)],
        capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise AudioError(f"encode failed: {result.stderr.strip()[-200:]}")
    partial.replace(destination)
    return loudness, probe_duration(destination)
