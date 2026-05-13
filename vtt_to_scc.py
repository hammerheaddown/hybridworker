#!/usr/bin/env python3
"""
VTT → SCC Conversion Pipeline with QC
======================================
Uses ttconv for proper CEA-608/SCC encoding via its IMSC canonical model.
Runs pre-conversion QC + normalization, then post-conversion validation.

Architecture:
  VTT input
    ↓
  Parse into ttconv canonical model
    ↓
  Extract caption data for QC
    ↓
  Pre-conversion QC + auto-fix (normalize)
    ↓
  Rebuild cleaned canonical model
    ↓
  ttconv SCC writer → .scc output
    ↓
  Post-conversion validation
    ↓
  QC report (JSON + console)

Usage:
  python vtt_to_scc.py input.vtt output.scc [--report report.json] [--no-fix] [--frame-rate 29.97df]
"""

import argparse
import io
import json
import os
import re
import sys
import textwrap
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Force UTF-8 on stdout/stderr so Unicode glyphs in help text and prints
# (arrows, em-dashes) don't crash on Windows consoles defaulted to CP1252.
# Modern Windows Terminal renders them; legacy cmd shows replacement chars
# but the script keeps running.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from ttconv.vtt import reader as vtt_reader
from ttconv.scc import writer as scc_writer
from ttconv.tt import SccWriterConfiguration
from ttconv import model


# ─────────────────────────────────────────────
# CEA-608 Constants
# ─────────────────────────────────────────────
CEA608_MAX_CHARS_PER_LINE = 32
CEA608_MAX_LINES = 2  # safe pop-on limit (4 for roll-up)
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 7.0
MAX_CPS = 20.0  # characters per second
MIN_CPS = 3.0   # suspiciously slow
MIN_GAP_SEC = 0.267  # ~8 frames at 29.97 — enough for SCC erase + pop-on control codes
# Minimum time from 00:00:00;00 before the first caption can display.
# Was 2.5s historically — a conservative buffer for SCC byte-pair priming.
# Reduced to 0.5s because:
#   - hybridCC-vod.c emits a marker SEI on the first video NAL, so decoder
#     state is primed before the first real caption regardless.
#   - 0.5s = ~30 byte-pairs at 60Hz, more than enough for typical caption
#     prologue (RCL + channel + ~30 chars of text + EDM + EOC).
#   - 2.5s creates a 2.5s lag between speech and displayed captions, plus
#     pushes the last caption past the end of short videos (visible loss
#     of trailing captions when the video ends before the cue does).
SCC_MIN_LEAD_SEC = 0.5
# Mirrors the Modal pipeline's stable-ts `split_by_length(42)` — Whisper
# segments longer than this get split into multiple cues at natural breaks
# (punctuation > conjunctions > spaces) before the line-wrap pass. Without
# this, raw Whisper segments (60-130 chars) overflow CEA-608's 2x32 layout
# and get truncated. Modal doesn't hit this because stable-ts pre-splits.
SPLIT_TARGET_CHARS = 42

# Characters that CEA-608 cannot encode (basic ASCII + some specials only)
# ttconv handles the actual encoding, but we strip things that will be lost
CEA608_SAFE_PATTERN = re.compile(
    r'[^\x20-\x7E'  # basic printable ASCII
    r'\u00e1\u00e9\u00ed\u00f3\u00fa'  # á é í ó ú
    r'\u00c1\u00c9\u00cd\u00d3\u00da'  # Á É Í Ó Ú
    r'\u00e7\u00c7'  # ç Ç
    r'\u00f1\u00d1'  # ñ Ñ
    r'\u00bf\u00a1'  # ¿ ¡
    r'\u00ae\u00a9\u00b0'  # ® © °
    r'\u00bd\u00bc\u00be'  # ½ ¼ ¾
    r'\u00a3\u00a2\u00a5'  # £ ¢ ¥
    r'\u266a\u2588'  # ♪ █
    r'\u00e0\u00e8\u00ec\u00f2\u00f9'  # à è ì ò ù
    r'\u00c0\u00c8\u00cc\u00d2\u00d9'  # À È Ì Ò Ù
    r'\u00e2\u00ea\u00ee\u00f4\u00fb'  # â ê î ô û
    r'\u00c2\u00ca\u00ce\u00d4\u00db'  # Â Ê Î Ô Û
    r'\u00e4\u00eb\u00ef\u00f6\u00fc'  # ä ë ï ö ü
    r'\u00c4\u00cb\u00cf\u00d6\u00dc'  # Ä Ë Ï Ö Ü
    r'\u00e5\u00c5'  # å Å
    r'\u00e6\u00c6'  # æ Æ
    r'\u00f8\u00d8'  # ø Ø
    r'\u00df'  # ß
    r'\n]'
)

# Smart quote / unicode normalization map
UNICODE_REPLACEMENTS = {
    '\u2018': "'",   # '
    '\u2019': "'",   # '
    '\u201C': '"',   # "
    '\u201D': '"',   # "
    '\u2013': '-',   # –
    '\u2014': '--',  # —
    '\u2026': '...',  # …
    '\u200B': '',    # zero-width space
    '\u00A0': ' ',   # non-breaking space
    '\uFEFF': '',    # BOM
    '\u200E': '',    # LTR mark
    '\u200F': '',    # RTL mark
}


# ─────────────────────────────────────────────
# Data classes for QC results
# ─────────────────────────────────────────────
@dataclass
class QCIssue:
    caption_index: int
    timestamp: str
    severity: str  # "error", "warning", "info"
    category: str
    message: str
    auto_fixed: bool = False

@dataclass
class CaptionData:
    index: int
    begin: float
    end: float
    text: str
    lines: list

@dataclass
class QCReport:
    input_file: str
    output_file: str
    total_captions: int
    issues: list = field(default_factory=list)
    pre_qc_issues: list = field(default_factory=list)
    post_qc_issues: list = field(default_factory=list)
    auto_fixes_applied: int = 0

    @property
    def error_count(self):
        """Total errors found (including auto-fixed)."""
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def unfixed_error_count(self):
        """Errors remaining after auto-fixes."""
        return sum(1 for i in self.issues if i.severity == "error" and not i.auto_fixed)

    @property
    def warning_count(self):
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self):
        d = asdict(self)
        d['error_count'] = self.error_count
        d['remaining_errors'] = len(getattr(self, 'remaining_errors', []))
        d['warning_count'] = self.warning_count
        d['pass'] = len(getattr(self, 'remaining_errors', [])) == 0 and len(self.post_qc_issues) == 0
        return d


# ─────────────────────────────────────────────
# Step 1: Parse VTT → extract caption data
# ─────────────────────────────────────────────
def parse_vtt_to_captions(vtt_path: str) -> tuple:
    """Read VTT via ttconv, return (ContentDocument, list[CaptionData])"""
    with open(vtt_path, 'r', encoding='utf-8') as f:
        doc = vtt_reader.to_model(f)

    captions = []
    body = doc.get_body()
    if body is None:
        return doc, captions

    idx = 0
    for div in body:
        for p in div:
            begin = p.get_begin()
            end = p.get_end()
            if begin is None or end is None:
                continue

            # Walk the tree to extract text
            text_parts = []
            for node in p.dfs_iterator():
                if isinstance(node, model.Text):
                    text_parts.append(node.get_text())

            text = ''.join(text_parts)
            lines = text.split('\n')

            captions.append(CaptionData(
                index=idx,
                begin=float(begin),
                end=float(end),
                text=text,
                lines=lines
            ))
            idx += 1

    return doc, captions


def format_tc(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for display"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ─────────────────────────────────────────────
# Step 2: Pre-conversion QC
# ─────────────────────────────────────────────
def run_pre_qc(captions: list, report: QCReport):
    """Validate captions BEFORE conversion. All issues are fixable."""
    prev = None
    for cap in captions:
        tc = format_tc(cap.begin)
        duration = cap.end - cap.begin

        # --- Line length ---
        for i, line in enumerate(cap.lines):
            if len(line) > CEA608_MAX_CHARS_PER_LINE:
                report.pre_qc_issues.append(QCIssue(
                    caption_index=cap.index,
                    timestamp=tc,
                    severity="error",
                    category="line_length",
                    message=f"Line {i+1} is {len(line)} chars (max {CEA608_MAX_CHARS_PER_LINE}): \"{line}\""
                ))

        # --- Too many lines ---
        if len(cap.lines) > CEA608_MAX_LINES:
            report.pre_qc_issues.append(QCIssue(
                caption_index=cap.index,
                timestamp=tc,
                severity="error",
                category="line_count",
                message=f"Caption has {len(cap.lines)} lines (max {CEA608_MAX_LINES} for pop-on)"
            ))

        # --- Duration ---
        if duration < MIN_DURATION_SEC:
            report.pre_qc_issues.append(QCIssue(
                caption_index=cap.index,
                timestamp=tc,
                severity="error",
                category="duration_short",
                message=f"Duration {duration:.2f}s is below {MIN_DURATION_SEC}s minimum"
            ))
        elif duration > MAX_DURATION_SEC:
            report.pre_qc_issues.append(QCIssue(
                caption_index=cap.index,
                timestamp=tc,
                severity="warning",
                category="duration_long",
                message=f"Duration {duration:.2f}s exceeds {MAX_DURATION_SEC}s"
            ))

        # --- Reading speed (CPS) ---
        clean_text = cap.text.replace('\n', ' ')
        char_count = len(clean_text.strip())
        if duration > 0 and char_count > 0:
            cps = char_count / duration
            if cps > MAX_CPS:
                report.pre_qc_issues.append(QCIssue(
                    caption_index=cap.index,
                    timestamp=tc,
                    severity="error",
                    category="reading_speed",
                    message=f"CPS={cps:.1f} exceeds {MAX_CPS} (text: \"{clean_text[:50]}...\")"
                ))
            elif cps < MIN_CPS and char_count > 3:
                report.pre_qc_issues.append(QCIssue(
                    caption_index=cap.index,
                    timestamp=tc,
                    severity="info",
                    category="reading_speed_slow",
                    message=f"CPS={cps:.1f} is very slow — caption may display too long"
                ))

        # --- Overlap with previous ---
        if prev is not None and cap.begin < prev.end:
            overlap = prev.end - cap.begin
            report.pre_qc_issues.append(QCIssue(
                caption_index=cap.index,
                timestamp=tc,
                severity="error",
                category="overlap",
                message=f"Overlaps previous caption by {overlap:.3f}s (prev ends {format_tc(prev.end)})"
            ))

        # --- Insufficient gap ---
        if prev is not None and cap.begin >= prev.end:
            gap = cap.begin - prev.end
            if 0 < gap < MIN_GAP_SEC:
                report.pre_qc_issues.append(QCIssue(
                    caption_index=cap.index,
                    timestamp=tc,
                    severity="warning",
                    category="gap_short",
                    message=f"Gap to previous caption is only {gap:.3f}s (min {MIN_GAP_SEC:.3f}s)"
                ))

        # --- Illegal / unsupported characters ---
        bad_chars = CEA608_SAFE_PATTERN.findall(cap.text)
        if bad_chars:
            unique = set(bad_chars)
            report.pre_qc_issues.append(QCIssue(
                caption_index=cap.index,
                timestamp=tc,
                severity="warning",
                category="illegal_chars",
                message=f"Unsupported characters will be stripped: {unique}"
            ))

        # --- Empty caption ---
        if not cap.text.strip():
            report.pre_qc_issues.append(QCIssue(
                caption_index=cap.index,
                timestamp=tc,
                severity="warning",
                category="empty",
                message="Empty caption"
            ))

        prev = cap

    report.issues.extend(report.pre_qc_issues)


# ─────────────────────────────────────────────
# Step 3: Auto-fix / normalize
# ─────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """Normalize Unicode and strip unsupported characters."""
    # Smart quotes and special chars
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)

    # NFC normalize first
    text = unicodedata.normalize('NFC', text)

    # Strip remaining unsupported chars
    text = CEA608_SAFE_PATTERN.sub('', text)

    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def smart_line_wrap(text: str, max_width: int = CEA608_MAX_CHARS_PER_LINE) -> str:
    """
    Wrap text to fit CEA-608 line limits.
    Tries to break at natural points (punctuation, conjunctions) and
    balance line lengths.
    """
    lines = text.split('\n')
    result_lines = []

    for line in lines:
        if len(line) <= max_width:
            result_lines.append(line)
            continue

        # Use textwrap but try to balance
        wrapped = textwrap.wrap(line, width=max_width, break_long_words=False, break_on_hyphens=True)
        result_lines.extend(wrapped)

    # If we end up with more than 2 lines, try to rebalance into 2 lines.
    # We NEVER truncate here — the split_long_cues pre-pass guarantees each
    # cue fits in 2x32. If somehow it doesn't (manual override, edge case),
    # we leave the extra lines in; ttconv's SCC writer will handle 4-row
    # roll-up display rather than silently dropping words.
    if len(result_lines) > CEA608_MAX_LINES:
        full_text = ' '.join(result_lines)
        if len(full_text) <= max_width * CEA608_MAX_LINES:
            mid = len(full_text) // 2
            best_break = mid
            for offset in range(min(15, mid)):
                if mid + offset < len(full_text) and full_text[mid + offset] == ' ':
                    best_break = mid + offset
                    break
                if mid - offset >= 0 and full_text[mid - offset] == ' ':
                    best_break = mid - offset
                    break
            line1 = full_text[:best_break].strip()
            line2 = full_text[best_break:].strip()
            if len(line1) <= max_width and len(line2) <= max_width:
                result_lines = [line1, line2]
        # else: leave result_lines as-is (3-4 rows). NO truncation.

    return '\n'.join(result_lines)


def _split_text_at_natural_break(text: str, target: int) -> list:
    """Split a long string into chunks of ~target chars at natural breaks.
    Preference order: sentence-end punctuation (. ! ?) > clause punctuation
    (, ; :) > conjunctions (and / but / or) > word boundaries.
    Returns a list of cleaned, non-empty chunks. Always returns >= 1 chunk."""
    text = text.strip()
    if len(text) <= target:
        return [text]

    chunks = []
    cursor = 0
    while cursor < len(text):
        remaining = text[cursor:]
        if len(remaining) <= target:
            chunks.append(remaining.strip())
            break

        # Search window: target ± window chars from cursor.
        window = max(8, target // 3)
        ideal = cursor + target

        # Tier 1: sentence-end punctuation
        best = -1
        for pos in range(ideal, max(cursor + 1, ideal - window), -1):
            if pos < len(text) and text[pos - 1] in ".!?":
                best = pos
                break
        # Tier 2: clause punctuation
        if best < 0:
            for pos in range(ideal, max(cursor + 1, ideal - window), -1):
                if pos < len(text) and text[pos - 1] in ",;:":
                    best = pos
                    break
        # Tier 3: conjunctions (split BEFORE the word, not after)
        if best < 0:
            for pos in range(ideal, max(cursor + 1, ideal - window), -1):
                if pos < len(text) and text[pos - 1] == " ":
                    nextword = text[pos:pos + 5].lower()
                    if nextword.startswith(("and ", "but ", "or ", "so ", "yet ")):
                        best = pos
                        break
        # Tier 4: any space
        if best < 0:
            for pos in range(ideal, max(cursor + 1, ideal - window), -1):
                if pos < len(text) and text[pos] == " ":
                    best = pos
                    break
        # Tier 5: hard cut at target (last resort, rare)
        if best < 0:
            best = ideal

        chunk = text[cursor:best].strip()
        if chunk:
            chunks.append(chunk)
        cursor = best
        # Skip leading whitespace into next chunk
        while cursor < len(text) and text[cursor] == " ":
            cursor += 1

    return [c for c in chunks if c]


def split_long_cues(captions: list, report: QCReport, target: int = SPLIT_TARGET_CHARS) -> list:
    """Pre-line-wrap pass: split cues whose text exceeds `target` chars into
    multiple cues at natural break points, dividing the original duration
    proportionally to chunk character counts. Mirrors stable-ts
    `split_by_length(42)` from the Modal pipeline."""
    out = []
    next_idx = 0
    for cap in captions:
        # Use total char count of joined text (line breaks not counted).
        flat = " ".join(cap.text.split())
        if len(flat) <= target:
            cap.index = next_idx
            out.append(cap)
            next_idx += 1
            continue

        chunks = _split_text_at_natural_break(flat, target)
        if len(chunks) <= 1:
            cap.index = next_idx
            out.append(cap)
            next_idx += 1
            continue

        total_chars = sum(len(c) for c in chunks) or 1
        total_dur = max(cap.end - cap.begin, 0.0)
        cur_t = cap.begin
        for chunk in chunks:
            share = len(chunk) / total_chars
            chunk_dur = total_dur * share
            new_cap = CaptionData(
                index=next_idx,
                begin=cur_t,
                end=cur_t + chunk_dur,
                text=chunk,
                lines=[chunk],
            )
            out.append(new_cap)
            cur_t += chunk_dur
            next_idx += 1

        report.issues.append(QCIssue(
            caption_index=cap.index,
            timestamp=format_tc(cap.begin),
            severity="info", category="cue_split",
            message=f"Split overlong cue ({len(flat)} chars) into {len(chunks)} cues",
            auto_fixed=True,
        ))

    return out


def auto_fix_captions(captions: list, report: QCReport, apply_fixes: bool = True) -> list:
    """Apply automatic fixes. Returns new list of CaptionData."""
    if not apply_fixes:
        return captions

    fixed = []
    fix_count = 0

    # ── Split long cues into multiple cues (pre-line-wrap) ─────────────────
    # Without this, the line_wrap pass below truncates anything that doesn't
    # fit in 2 rows × 32 chars. Modal avoids this via stable-ts pre-splitting.
    pre_split_count = len(captions)
    captions = split_long_cues(captions, report)
    if len(captions) > pre_split_count:
        fix_count += (len(captions) - pre_split_count)

    # ── Cascade SCC lead-in shift ──────────────────────────────────────────
    # If the first cue starts before SCC_MIN_LEAD_SEC, shift the ENTIRE
    # timeline forward by the delta. This preserves every cue's duration
    # and avoids the old per-cue shift that squeezed cue 0 via cascade
    # overlap-fix on tightly packed inputs.
    if captions and captions[0].begin < SCC_MIN_LEAD_SEC:
        shift_amt = SCC_MIN_LEAD_SEC - captions[0].begin
        fix_count += 1
        report.issues.append(QCIssue(
            caption_index=captions[0].index,
            timestamp=format_tc(captions[0].begin),
            severity="info", category="scc_lead_in",
            message=f"Cascade-shifted all {len(captions)} cues forward by {shift_amt:.3f}s for SCC byte-pair lead-in",
            auto_fixed=True
        ))
        captions = [
            CaptionData(
                index=c.index,
                begin=c.begin + shift_amt,
                end=c.end + shift_amt,
                text=c.text,
                lines=c.lines,
            )
            for c in captions
        ]

    for cap in captions:
        text = cap.text
        begin = cap.begin
        end = cap.end

        # --- Skip empty captions early ---
        normalized = normalize_text(text)
        if not normalized.strip():
            fix_count += 1
            report.issues.append(QCIssue(
                caption_index=cap.index, timestamp=format_tc(begin),
                severity="info", category="removed_empty",
                message="Removed empty caption", auto_fixed=True
            ))
            continue

        # --- Normalize unicode ---
        if normalized != text:
            fix_count += 1
            report.issues.append(QCIssue(
                caption_index=cap.index, timestamp=format_tc(begin),
                severity="info", category="normalize",
                message="Unicode normalized", auto_fixed=True
            ))
        text = normalized

        # --- Smart line wrap ---
        wrapped = smart_line_wrap(text)
        if wrapped != text:
            fix_count += 1
            report.issues.append(QCIssue(
                caption_index=cap.index, timestamp=format_tc(begin),
                severity="info", category="line_wrap",
                message=f"Re-wrapped to fit {CEA608_MAX_CHARS_PER_LINE} chars/line",
                auto_fixed=True
            ))
        text = wrapped

        # --- Fix short duration ---
        duration = end - begin
        if duration < MIN_DURATION_SEC and duration > 0:
            end = begin + MIN_DURATION_SEC
            fix_count += 1
            report.issues.append(QCIssue(
                caption_index=cap.index, timestamp=format_tc(begin),
                severity="info", category="duration_extend",
                message=f"Extended duration from {duration:.2f}s to {MIN_DURATION_SEC}s",
                auto_fixed=True
            ))

        new_cap = CaptionData(
            index=cap.index,
            begin=begin,
            end=end,
            text=text,
            lines=text.split('\n')
        )
        fixed.append(new_cap)

    # ── CPS-extend pass ────────────────────────────────────────────────────
    # For cues with CPS > MAX_CPS, extend the end time into available gap
    # before the next cue (or unlimited slack if it's the last cue). This
    # fixes "speech is too dense for the cue window" — a real reading-speed
    # problem that auto-fix couldn't resolve before this pass existed.
    for i, cap in enumerate(fixed):
        char_count = len(cap.text.replace('\n', ' ').strip())
        duration = cap.end - cap.begin
        if duration <= 0 or char_count == 0:
            continue
        cps = char_count / duration
        if cps <= MAX_CPS:
            continue

        # Target slightly below MAX_CPS so float-precision quirks don't
        # leave us at 20.0000001 which then trips the >20.0 check.
        target_duration = char_count / (MAX_CPS - 0.5)
        extension_needed = target_duration - duration

        if i + 1 < len(fixed):
            next_begin = fixed[i + 1].begin
            available = next_begin - cap.end - MIN_GAP_SEC
        else:
            # Last cue — unlimited slack (display past end-of-video is fine;
            # players just stop showing when video ends)
            available = extension_needed

        extension = min(extension_needed, max(0.0, available))
        if extension > 0:
            new_end = cap.end + extension
            new_cps = char_count / (new_end - cap.begin)
            fix_count += 1
            report.issues.append(QCIssue(
                caption_index=cap.index, timestamp=format_tc(cap.begin),
                severity="info", category="cps_extend",
                message=f"Extended end by {extension:.3f}s to lower CPS from {cps:.1f} to {new_cps:.1f}",
                auto_fixed=True
            ))
            cap.end = new_end

    # --- Second pass: fix overlaps and gaps sequentially ---
    for i in range(1, len(fixed)):
        prev = fixed[i - 1]
        curr = fixed[i]

        # Fix overlap: trim previous end to create minimum gap
        if curr.begin < prev.end:
            new_prev_end = curr.begin - MIN_GAP_SEC
            if new_prev_end > prev.begin + 0.5:  # keep at least 0.5s for prev
                fix_count += 1
                report.issues.append(QCIssue(
                    caption_index=curr.index, timestamp=format_tc(curr.begin),
                    severity="info", category="overlap_fix",
                    message=f"Trimmed previous caption end from {format_tc(prev.end)} to {format_tc(new_prev_end)}",
                    auto_fixed=True
                ))
                prev.end = new_prev_end
            else:
                # Can't trim prev enough — push current forward instead
                new_begin = prev.end + MIN_GAP_SEC
                shift = new_begin - curr.begin
                fix_count += 1
                report.issues.append(QCIssue(
                    caption_index=curr.index, timestamp=format_tc(curr.begin),
                    severity="info", category="overlap_fix",
                    message=f"Shifted caption forward by {shift:.3f}s to resolve overlap",
                    auto_fixed=True
                ))
                curr.end += shift
                curr.begin = new_begin

        # Fix insufficient gap
        elif curr.begin >= prev.end:
            gap = curr.begin - prev.end
            if 0 < gap < MIN_GAP_SEC:
                prev.end = curr.begin - MIN_GAP_SEC
                if prev.end < prev.begin + 0.5:
                    prev.end = prev.begin + 0.5
                fix_count += 1
                report.issues.append(QCIssue(
                    caption_index=curr.index, timestamp=format_tc(curr.begin),
                    severity="info", category="gap_fix",
                    message=f"Adjusted gap from {gap:.3f}s to {MIN_GAP_SEC:.3f}s",
                    auto_fixed=True
                ))

    report.auto_fixes_applied = fix_count
    return fixed


# ─────────────────────────────────────────────
# Step 4: Rebuild canonical model from cleaned captions
# ─────────────────────────────────────────────
SCC_BROADCAST_OFFSET_SEC = 3600.0  # 1-hour SMPTE TC convention for SCC files


def build_model_from_captions(captions: list) -> model.ContentDocument:
    """Build a fresh ttconv ContentDocument from cleaned caption data.

    Cue times are shifted by SCC_BROADCAST_OFFSET_SEC (1 hour) to follow the
    SMPTE timecode convention used in broadcast SCC files (program start at
    01:00:00;00 — leaves the first hour for color bars/slate/preroll). The
    1-hour buffer also gives ttconv unlimited preamble room for pop-on
    byte-pair encoding, which avoids the "stream would start earlier than
    start timecode" error on cues placed shortly after t=0."""
    doc = model.ContentDocument()

    # Create a default region (centered, bottom of screen)
    region = model.Region("r1", doc)
    doc.put_region(region)

    body = model.Body(doc)
    doc.set_body(body)

    div = model.Div(doc)
    body.push_child(div)

    for cap in captions:
        p = model.P(doc)
        p.set_begin(cap.begin + SCC_BROADCAST_OFFSET_SEC)
        p.set_end(cap.end + SCC_BROADCAST_OFFSET_SEC)
        p.set_region(region)

        span = model.Span(doc)
        p.push_child(span)

        text_node = model.Text(doc)
        text_node.set_text(cap.text)
        span.push_child(text_node)

        div.push_child(p)

    return doc


# ─────────────────────────────────────────────
# Step 5: Convert to SCC via ttconv
# ─────────────────────────────────────────────
def convert_to_scc(doc: model.ContentDocument, config: Optional[SccWriterConfiguration] = None) -> str:
    """Run ttconv SCC writer on the canonical model."""
    if config is None:
        config = SccWriterConfiguration()
    # 01:00:00;00 = SMPTE broadcast convention. Model already shifted by
    # SCC_BROADCAST_OFFSET_SEC in build_model_from_captions so this aligns.
    config.start_tc = "01:00:00;00"
    config.force_popon = True
    config.allow_reflow = True
    return scc_writer.from_model(doc, config)


# ─────────────────────────────────────────────
# Step 6: Post-conversion validation
# ─────────────────────────────────────────────
def validate_scc(scc_content: str, report: QCReport):
    """Validate the output SCC file structure."""
    lines = scc_content.strip().split('\n')

    # Check header
    if not lines or 'Scenarist_SCC' not in lines[0]:
        report.post_qc_issues.append(QCIssue(
            caption_index=-1, timestamp="00:00:00.000",
            severity="error", category="scc_header",
            message="Missing Scenarist_SCC V1.0 header"
        ))

    data_lines = [l for l in lines if l.strip() and 'Scenarist_SCC' not in l]
    tc_pattern = re.compile(r'^(\d{2}:\d{2}:\d{2};\d{2})\t(.+)$')

    prev_tc = None
    for line in data_lines:
        match = tc_pattern.match(line)
        if not match:
            continue

        tc_str = match.group(1)
        hex_data = match.group(2)

        # Validate hex pairs
        hex_words = hex_data.strip().split()
        for word in hex_words:
            if not re.match(r'^[0-9a-fA-F]{4}$', word):
                report.post_qc_issues.append(QCIssue(
                    caption_index=-1, timestamp=tc_str,
                    severity="error", category="scc_hex",
                    message=f"Invalid hex word: {word}"
                ))

        # Check timecode ordering
        if prev_tc is not None and tc_str <= prev_tc:
            report.post_qc_issues.append(QCIssue(
                caption_index=-1, timestamp=tc_str,
                severity="error", category="scc_tc_order",
                message=f"Timecode not ascending: {tc_str} <= {prev_tc}"
            ))
        prev_tc = tc_str

    if not data_lines:
        report.post_qc_issues.append(QCIssue(
            caption_index=-1, timestamp="00:00:00.000",
            severity="error", category="scc_empty",
            message="SCC file contains no caption data"
        ))

    report.issues.extend(report.post_qc_issues)


# ─────────────────────────────────────────────
# CLI + Main Pipeline
# ─────────────────────────────────────────────
def print_report(report: QCReport):
    """Pretty print the QC report to console."""
    print("\n" + "=" * 60)
    print("  VTT → SCC  QC REPORT")
    print("=" * 60)
    print(f"  Input:    {report.input_file}")
    print(f"  Output:   {report.output_file}")
    print(f"  Captions: {report.total_captions}")
    print(f"  Fixes:    {report.auto_fixes_applied}")
    print("-" * 60)

    if report.pre_qc_issues:
        print("\n  PRE-CONVERSION ISSUES:")
        for issue in report.pre_qc_issues:
            icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(issue.severity, "?")
            fixed = " [FIXED]" if issue.auto_fixed else ""
            print(f"    {icon} [{issue.timestamp}] {issue.category}: {issue.message}{fixed}")

    if report.post_qc_issues:
        print("\n  POST-CONVERSION ISSUES:")
        for issue in report.post_qc_issues:
            icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(issue.severity, "?")
            print(f"    {icon} [{issue.timestamp}] {issue.category}: {issue.message}")

    auto_fixed = [i for i in report.issues if i.auto_fixed]
    if auto_fixed:
        print(f"\n  AUTO-FIXES APPLIED: {len(auto_fixed)}")
        for issue in auto_fixed:
            print(f"    ✓ [{issue.timestamp}] {issue.category}: {issue.message}")

    print("\n" + "-" * 60)
    remaining_errs = getattr(report, 'remaining_errors', [])
    remaining_warns = getattr(report, 'remaining_warnings', [])
    has_post_errors = any(i.severity == "error" for i in report.post_qc_issues)
    is_pass = len(remaining_errs) == 0 and not has_post_errors
    result = "PASS" if is_pass else "FAIL"

    if remaining_errs:
        print("\n  REMAINING ERRORS (not auto-fixable):")
        for issue in remaining_errs:
            print(f"    ✗ [{issue.timestamp}] {issue.category}: {issue.message}")

    print(f"\n  RESULT: {result}")
    print(f"    {report.error_count} errors found in original")
    print(f"    {report.auto_fixes_applied} auto-fixes applied")
    print(f"    {len(remaining_errs)} errors remaining after fixes")
    print(f"    {report.warning_count} warnings")
    print("=" * 60 + "\n")


def write_cleaned_vtt(captions: list, path: str):
    """Write cleaned CaptionData list back as a WEBVTT file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('WEBVTT\n\n')
        for cap in captions:
            f.write(f'{format_tc(cap.begin)} --> {format_tc(cap.end)}\n')
            f.write(cap.text + '\n\n')


def write_cleaned_srt(captions: list, path: str):
    """Write cleaned CaptionData list as SRT (comma decimal separator, 1-based index)."""
    def srt_tc(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    with open(path, 'w', encoding='utf-8') as f:
        for i, cap in enumerate(captions, start=1):
            f.write(f'{i}\n')
            f.write(f'{srt_tc(cap.begin)} --> {srt_tc(cap.end)}\n')
            f.write(cap.text + '\n\n')


def run_pipeline(vtt_path: str, scc_path: str, report_path: Optional[str] = None,
                 apply_fixes: bool = True, frame_rate: str = "29.97df",
                 cleaned_vtt_path: Optional[str] = None,
                 cleaned_srt_path: Optional[str] = None):
    """Main pipeline: VTT → QC → normalize → SCC → validate → report"""

    report = QCReport(input_file=vtt_path, output_file=scc_path, total_captions=0)

    # Step 1: Parse VTT
    print(f"[1/6] Parsing VTT: {vtt_path}")
    doc, captions = parse_vtt_to_captions(vtt_path)
    report.total_captions = len(captions)
    print(f"      Found {len(captions)} captions")

    if not captions:
        print("      ERROR: No captions found in VTT file")
        report.issues.append(QCIssue(
            caption_index=-1, timestamp="00:00:00.000",
            severity="error", category="no_captions",
            message="No captions found in input file"
        ))
        print_report(report)
        return report

    # Step 2: Pre-conversion QC
    print(f"[2/6] Running pre-conversion QC...")
    run_pre_qc(captions, report)
    print(f"      {len(report.pre_qc_issues)} issues found")

    # Step 3: Auto-fix
    print(f"[3/6] Normalizing captions (auto-fix={'ON' if apply_fixes else 'OFF'})...")
    cleaned = auto_fix_captions(captions, report, apply_fixes)
    print(f"      {report.auto_fixes_applied} fixes applied, {len(cleaned)} captions remaining")

    # Step 3b: Re-validate cleaned captions to check what errors remain
    if apply_fixes:
        post_fix_report = QCReport(input_file=vtt_path, output_file=scc_path, total_captions=len(cleaned))
        run_pre_qc(cleaned, post_fix_report)
        remaining_errors = [i for i in post_fix_report.pre_qc_issues if i.severity == "error"]
        remaining_warnings = [i for i in post_fix_report.pre_qc_issues if i.severity == "warning"]
        report.remaining_errors = remaining_errors
        report.remaining_warnings = remaining_warnings
        if remaining_errors:
            print(f"      {len(remaining_errors)} errors remain after auto-fix")
            report.issues.extend(remaining_errors)
    else:
        report.remaining_errors = [i for i in report.pre_qc_issues if i.severity == "error"]
        report.remaining_warnings = [i for i in report.pre_qc_issues if i.severity == "warning"]

    # Step 3c: Write cleaned VTT/SRT sidecars if requested
    if cleaned_vtt_path:
        write_cleaned_vtt(cleaned, cleaned_vtt_path)
        print(f"      Cleaned VTT written to {cleaned_vtt_path}")
    if cleaned_srt_path:
        write_cleaned_srt(cleaned, cleaned_srt_path)
        print(f"      Cleaned SRT written to {cleaned_srt_path}")

    # Step 4: Rebuild model
    print(f"[4/6] Building canonical model...")
    clean_doc = build_model_from_captions(cleaned)

    # Step 5: Convert to SCC
    print(f"[5/6] Converting to SCC (frame rate: {frame_rate})...")
    scc_config = SccWriterConfiguration()
    try:
        scc_output = convert_to_scc(clean_doc, scc_config)

        # Write output
        with open(scc_path, 'w', encoding='utf-8') as f:
            f.write(scc_output)
        print(f"      Written to {scc_path}")

        # Step 6: Post-conversion validation
        print(f"[6/6] Validating SCC output...")
        validate_scc(scc_output, report)
        print(f"      {len(report.post_qc_issues)} post-conversion issues")

    except RuntimeError as e:
        print(f"      SCC conversion FAILED: {e}")
        report.post_qc_issues.append(QCIssue(
            caption_index=-1, timestamp="00:00:00.000",
            severity="error", category="scc_conversion_failed",
            message=f"ttconv SCC writer error: {e}"
        ))
        report.issues.append(report.post_qc_issues[-1])

    # Report
    print_report(report)

    if report_path:
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"Report saved to {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="VTT → SCC with QC pipeline (ttconv)")
    parser.add_argument("input", help="Input .vtt file")
    parser.add_argument("output", help="Output .scc file")
    parser.add_argument("--report", help="Save QC report as JSON", default=None)
    parser.add_argument("--no-fix", action="store_true", help="Disable auto-fixes (QC report only)")
    parser.add_argument("--frame-rate", default="29.97df", help="SCC frame rate (default: 29.97df)")
    parser.add_argument("--cleaned-vtt", help="Write cleaned (post-QC) VTT to this path", default=None)
    parser.add_argument("--cleaned-srt", help="Write cleaned (post-QC) SRT to this path", default=None)
    parser.add_argument(
        "--profile", choices=["broadcast", "shorts"], default="broadcast",
        help="QC profile. broadcast = FCC §79.1 (1.0s min, 20 CPS). "
             "shorts = relaxed timing for YouTube Shorts / Reels / TikTok-style content "
             "with continuous rapid speech and no audio breaks (0.5s min, 25 CPS)."
    )
    args = parser.parse_args()

    # Profile override: short-form content has continuous rapid speech with
    # no natural breaks. Strict broadcast pacing (1.0s min, 20 CPS) flags
    # almost every cue as too-short / too-fast even though the captions
    # themselves are correct. Loosen the rules so the verdict reflects
    # encoding quality, not source pacing.
    if args.profile == "shorts":
        global MIN_DURATION_SEC, MAX_CPS
        MIN_DURATION_SEC = 0.5
        MAX_CPS = 25.0
        print(f"[profile] shorts — relaxed timing: min duration {MIN_DURATION_SEC}s, max CPS {MAX_CPS}")

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    report = run_pipeline(
        vtt_path=args.input,
        scc_path=args.output,
        report_path=args.report,
        apply_fixes=not args.no_fix,
        frame_rate=args.frame_rate,
        cleaned_vtt_path=args.cleaned_vtt,
        cleaned_srt_path=args.cleaned_srt,
    )

    remaining = len(getattr(report, 'remaining_errors', []))
    sys.exit(0 if remaining == 0 else 1)


if __name__ == "__main__":
    main()
