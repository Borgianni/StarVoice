from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusItem:
    utterance_id: str
    speaker_id: str
    chapter_id: str
    source_flac: str
    transcript: str
    source_sha256: str
    duration_s: float


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _duration_s(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required; install ffmpeg")
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(proc.stdout.strip())


def _read_transcripts(root: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for path in sorted(root.rglob("*.trans.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt, text = line.split(" ", 1)
            transcripts[utt] = text.strip()
    return transcripts


def scan_librispeech(root: Path) -> list[CorpusItem]:
    transcripts = _read_transcripts(root)
    items: list[CorpusItem] = []
    for flac in sorted(root.rglob("*.flac")):
        utt = flac.stem
        parts = utt.split("-")
        if len(parts) < 3:
            raise ValueError(f"unexpected LibriSpeech utterance id: {utt}")
        speaker, chapter = parts[0], parts[1]
        if utt not in transcripts:
            raise ValueError(f"missing transcript for {utt}")
        items.append(
            CorpusItem(
                utterance_id=utt,
                speaker_id=speaker,
                chapter_id=chapter,
                source_flac=str(flac),
                transcript=transcripts[utt],
                source_sha256=_sha256(flac),
                duration_s=_duration_s(flac),
            )
        )
    if not items:
        raise ValueError(f"no FLAC files found under {root}")
    return items


def select_balanced_subset(
    items: list[CorpusItem],
    target_minutes: float,
    seed: int,
    max_per_speaker: int = 20,
) -> list[CorpusItem]:
    if target_minutes <= 0:
        raise ValueError("target_minutes must be positive")
    rng = random.Random(seed)
    by_speaker: dict[str, list[CorpusItem]] = {}
    for item in items:
        by_speaker.setdefault(item.speaker_id, []).append(item)
    for group in by_speaker.values():
        rng.shuffle(group)

    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    chosen: list[CorpusItem] = []
    counts = {s: 0 for s in speakers}
    total = 0.0
    target_s = target_minutes * 60.0

    while total < target_s:
        progress = False
        for speaker in speakers:
            if total >= target_s:
                break
            if counts[speaker] >= max_per_speaker or not by_speaker[speaker]:
                continue
            item = by_speaker[speaker].pop()
            chosen.append(item)
            counts[speaker] += 1
            total += item.duration_s
            progress = True
        if not progress:
            break

    if total < target_s:
        raise ValueError(
            f"could only select {total / 60:.2f} min, below target {target_minutes:.2f} min"
        )
    return chosen


def _convert(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required; install ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        check=True,
    )


def prepare_librispeech(
    root: Path,
    output: Path,
    target_minutes: float = 30.0,
    seed: int = 2027,
    max_per_speaker: int = 20,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    wav_dir = output / "wav"
    items = scan_librispeech(root)
    selected = select_balanced_subset(items, target_minutes, seed, max_per_speaker)

    manifest_rows: list[dict] = []
    for idx, item in enumerate(selected):
        wav = wav_dir / f"{item.utterance_id}.wav"
        _convert(Path(item.source_flac), wav)
        row = asdict(item)
        row.update(
            {
                "selection_index": idx,
                "wav_48k_mono_pcm16": str(wav),
                "wav_sha256": _sha256(wav),
            }
        )
        manifest_rows.append(row)

    jsonl = output / "manifest.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    csv_path = output / "manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "schema_version": 1,
        "source_root": str(root),
        "seed": seed,
        "target_minutes": target_minutes,
        "selected_utterances": len(selected),
        "selected_speakers": len({x.speaker_id for x in selected}),
        "selected_duration_s": sum(x.duration_s for x in selected),
        "max_per_speaker": max_per_speaker,
        "audio_format": "48000 Hz mono PCM16 WAV",
        "manifest_jsonl": str(jsonl),
        "manifest_csv": str(csv_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
