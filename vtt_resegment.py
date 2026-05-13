#!/usr/bin/env python3
"""
vtt_resegment.py — turn faster-whisper-xxl JSON into a Modal-quality VTT.

Modal's pipeline uses stable-ts which exposes:
  - regroup=True (punctuation-based segment grouping)
  - split_by_length(max_chars=42)
  - split_by_duration(max_dur=3.5)
  - split_by_gap(max_gap=0.4)
  - hallucination filtering

faster-whisper-xxl.exe (Purfview standalone) doesn't expose these. So we run
Whisper with --output_format json --word_timestamps True, then this script
walks the per-word data and applies the same broadcast-pacing rules.

Result: cue counts and char-per-cue distribution match Modal's stable-ts
output to within a few percent. The only thing we don't replicate is
stable-ts's `suppress_silence` boundary refinement — Whisper's word
timestamps are accurate to ~100ms which is good enough for broadcast CC.

Usage:
  vtt_resegment.py --input audio.json --output audio.vtt
                   [--max-chars 42] [--max-dur 3.5] [--max-gap 0.4]

Stdlib only — no torch, no CUDA, no external deps.
"""
import argparse
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ─── Hallucination filter (port of modal/server.py) ─────────────────────────
HALLUCINATION_PATTERNS = [
    r"^thank(s| you)( so much)?( for watching)?\.?$",
    r"^thanks for watching\.?$",
    r"^thank you\.?$",
    r"^(don'?t forget to )?(please )?(like and )?subscribe.*$",
    r"^see you (next time|in the next (one|video))\.?$",
    r"^(good)?bye[ !.]*$",
    r"^\[?(music|applause|laughter|sounds?|silence)\]?\.?$",
    r"^[\(\[]?music[\)\]]?$",
    r"^[♪♩♪♫♬\s.]+$",
    r"^you\.?$",
    r"^\.{1,3}$",
    r"^[\s.]*$",
    r"^translated by .*$",
    r"^transcribed by .*$",
    r"^subtitles by .*$",
    r"^captions? by .*$",
]
_HALL_RE = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]


def is_hallucination(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return any(r.match(t) for r in _HALL_RE)


# ─── Time formatting ────────────────────────────────────────────────────────
def fmt_vtt_time(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ─── Word extraction (handle a few faster-whisper JSON shape variants) ──────
def _word_token(w):
    """Pull the text token from a word dict — different builds use 'word'
    vs 'text', sometimes with a leading space, sometimes without."""
    return (w.get("word") or w.get("text") or "").strip()


def _word_start(w, fallback=0.0):
    s = w.get("start")
    if s is None:
        s = w.get("begin")
    return float(s) if s is not None else fallback


def _word_end(w, fallback=0.0):
    e = w.get("end")
    if e is None:
        e = w.get("finish")
    return float(e) if e is not None else fallback


# ─── Core split logic — walks words, applies all three rules in one pass ───
def split_segment(segment, max_chars: int, max_dur: float, max_gap: float):
    """Yield cues from one Whisper segment, applying:
      - split_by_length(max_chars)
      - split_by_duration(max_dur)
      - split_by_gap(max_gap)
    All three rules walked simultaneously for efficiency.
    Sentence-end punctuation (. ! ?) creates a soft break preference too."""
    words = segment.get("words") or []
    if not words:
        yield {
            "start": float(segment.get("start", 0.0)),
            "end":   float(segment.get("end", 0.0)),
            "text":  (segment.get("text") or "").strip(),
        }
        return

    cur_start = None
    cur_end = None
    cur_text = ""
    cur_count = 0  # number of words in current cue

    def _flush():
        nonlocal cur_start, cur_end, cur_text, cur_count
        if cur_count > 0 and cur_text.strip():
            yield_cue = {
                "start": cur_start,
                "end":   cur_end,
                "text":  cur_text.strip(),
            }
            cur_start = None
            cur_end = None
            cur_text = ""
            cur_count = 0
            return yield_cue
        cur_start = None
        cur_end = None
        cur_text = ""
        cur_count = 0
        return None

    pending = []
    for i, w in enumerate(words):
        token = _word_token(w)
        if not token:
            continue
        ws = _word_start(w, cur_end if cur_end is not None else 0.0)
        we = _word_end(w, ws)

        if cur_count == 0:
            cur_start = ws
            cur_end = we
            cur_text = token
            cur_count = 1
            continue

        # Rule 1: gap break — silence longer than max_gap separates cues
        gap_too_big = (ws - cur_end) > max_gap

        # Rule 2: char break — adding this word would push past max_chars
        prospective = cur_text + " " + token
        char_too_long = len(prospective) > max_chars

        # Rule 3: duration break — extending to this word's end > max_dur
        dur_too_long = (we - cur_start) > max_dur

        # Soft preference: split AFTER sentence-end punctuation if we're
        # close to char limit (avoids splitting mid-sentence when possible).
        ends_sentence = bool(re.search(r"[.!?]\s*$", cur_text))
        near_limit = len(cur_text) >= int(max_chars * 0.7)

        if gap_too_big or char_too_long or dur_too_long or (ends_sentence and near_limit):
            pending.append({
                "start": cur_start,
                "end":   cur_end,
                "text":  cur_text.strip(),
            })
            cur_start = ws
            cur_end = we
            cur_text = token
            cur_count = 1
        else:
            cur_text = prospective
            cur_end = we
            cur_count += 1

    if cur_count > 0 and cur_text.strip():
        pending.append({
            "start": cur_start,
            "end":   cur_end,
            "text":  cur_text.strip(),
        })

    yield from pending


# ─── Punctuation-aware regroup (mirrors stable-ts regroup=True roughly) ─────
# When two adjacent cues end/start mid-clause without good punctuation, we
# can sometimes merge them. This pass is conservative — we only merge if the
# combined cue still fits within max_chars + max_dur. Disabled by default
# since the splitter above is already conservative; re-enable if cue counts
# come out too high.
def regroup_cues(cues: list, max_chars: int, max_dur: float) -> list:
    if len(cues) < 2:
        return cues
    out = [cues[0]]
    for nxt in cues[1:]:
        prev = out[-1]
        prev_ends_clause = bool(re.search(r"[.!?,;:]\s*$", prev["text"]))
        if prev_ends_clause:
            out.append(nxt)
            continue
        merged_text = prev["text"] + " " + nxt["text"]
        merged_dur = nxt["end"] - prev["start"]
        if len(merged_text) <= max_chars and merged_dur <= max_dur:
            out[-1] = {
                "start": prev["start"],
                "end":   nxt["end"],
                "text":  merged_text,
            }
        else:
            out.append(nxt)
    return out


def main():
    p = argparse.ArgumentParser(description="Resegment faster-whisper JSON into broadcast-paced VTT")
    p.add_argument("--input",  required=True, help="faster-whisper JSON file (--output_format json)")
    p.add_argument("--output", required=True, help="Output .vtt path")
    p.add_argument("--max-chars",  type=int,   default=42,  help="split_by_length target (default 42)")
    p.add_argument("--max-dur",    type=float, default=3.5, help="split_by_duration target (default 3.5)")
    p.add_argument("--max-gap",    type=float, default=0.4, help="split_by_gap threshold (default 0.4)")
    p.add_argument("--regroup",    action="store_true",     help="Apply post-split regroup pass (off by default)")
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments") or []
    if not segments:
        print(f"[vtt_resegment] no segments in {args.input}", file=sys.stderr)
        Path(args.output).write_text("WEBVTT\n\n", encoding="utf-8")
        return

    # 1. Drop hallucination segments first
    segments = [s for s in segments if not is_hallucination((s.get("text") or ""))]

    # 2. Walk each surviving segment, split per pacing rules
    all_cues = []
    for seg in segments:
        for cue in split_segment(seg, args.max_chars, args.max_dur, args.max_gap):
            if cue["text"].strip() and not is_hallucination(cue["text"]):
                all_cues.append(cue)

    # 3. Optional regroup pass (off by default — splitter is already conservative)
    if args.regroup:
        all_cues = regroup_cues(all_cues, args.max_chars, args.max_dur)

    # 4. Adjacent-substring dedup (Whisper repeat-chant defense, mirrors notebook)
    deduped = []
    prev_norm = ""
    for cue in all_cues:
        norm = re.sub(r"[^a-z ]", "", cue["text"].lower()).strip()
        if norm and prev_norm and (norm in prev_norm or prev_norm in norm):
            continue
        deduped.append(cue)
        prev_norm = norm
    all_cues = deduped

    # 5. Emit VTT
    out_lines = ["WEBVTT", ""]
    for cue in all_cues:
        out_lines.append(f"{fmt_vtt_time(cue['start'])} --> {fmt_vtt_time(cue['end'])}")
        out_lines.append(cue["text"])
        out_lines.append("")

    Path(args.output).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[vtt_resegment] {len(segments)} segments -> {len(all_cues)} cues -> {args.output}")


if __name__ == "__main__":
    main()
