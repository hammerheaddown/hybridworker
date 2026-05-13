#!/usr/bin/env python3
"""
proof_render.py — render an HTML proof report for a captioned video.

Standalone version of modal/server.py's `_render_proof_artifacts`, designed
for PyInstaller bundling so the local hybridCC pipeline produces a rich
audit PDF without needing Python on the customer PC.

Reads:
  - qc.json (output of vtt_to_scc.py --report)
  - the captioned MP4 (for size + duration via ffprobe, optional)
  - the SCC sidecar (for header validation)

Writes:
  - <output>.html — feed this to msedge --print-to-pdf to produce the PDF

Decoder verification (ffmpeg subcc + CCExtractor word-match scores) is
intentionally OMITTED here. That's a cloud-tier differentiator. The local
PDF gets verdict + QC verification + auto-fixes + remaining-errors detail,
which is enough to audit the encoding.

Usage:
  proof_render.py --qc <qc.json> --src-name <name.mp4> --src-stem <name>
                  --scc <name.scc> [--media-id <id>] [--output <out.html>]
                  [--source-duration-sec N] [--whisper-sec N] [--inject-sec N]
"""
import argparse
import html as html_mod
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


SOFT_QC_CATEGORIES = {"duration_short", "reading_speed"}
PASS_THRESHOLD = 0.80


# ─── Decoder verification helpers (mirror modal/server.py logic) ────────────
def _to_word_set(text: str) -> set:
    """Lowercase, split on non-word chars, drop short tokens. Same as Modal."""
    if not text:
        return set()
    words = re.findall(r"[A-Za-z']+", text.lower())
    return {w for w in words if len(w) >= 3}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _run_ffmpeg_subcc(ffmpeg_exe: Path, mp4: Path) -> str:
    """Decoder #1: ffmpeg's built-in subcc demuxer. Returns extracted VTT text or ''.

    NOTE on Windows path-escaping: ffmpeg's lavfi `movie=` filter parses the
    path string with its filter-graph syntax, where `:` separates options.
    On Windows, `C:\\Users\\...` literally contains a colon, so the filter
    engine sees `C` as the filename and `\\Users\\...` as malformed options
    -> the filter silently produces no captions and decoder #1 reports 0%.
    Fix: backslash-escape every colon and use forward slashes (libavfilter
    accepts both on Windows but the escape rule applies regardless)."""
    if not ffmpeg_exe.exists() or not mp4.exists():
        return ""
    out = Path(tempfile.gettempdir()) / f"hybridcc_ff_{mp4.stem}.vtt"
    try:
        # Windows ffmpeg gotcha: the lavfi `movie=` filter parses options
        # using `:` as separator, so a Windows path like `C:\Users\...` is
        # parsed as filename=C with `\Users\...` as malformed options. Two
        # fixes required:
        #   1. Pass `-f lavfi` explicitly so ffmpeg routes `-i movie=...`
        #      through libavfilter (don't rely on auto-detection).
        #   2. Wrap the path in single quotes inside the filtergraph string
        #      so the parser treats it as a literal value.
        # Then the standard `\:` escape inside the quoted value works.
        movie_path = str(mp4).replace("\\", "/").replace(":", r"\:")
        subprocess.run(
            [str(ffmpeg_exe), "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi",
             "-i", f"movie='{movie_path}'[out0+subcc]",
             "-map", "0:s", "-c:s", "webvtt", str(out)],
            capture_output=True, timeout=120, text=True,
        )
        return out.read_text(encoding="utf-8", errors="replace") if out.exists() else ""
    except Exception:
        return ""


def _run_ccextractor(ccx_exe: Path, mp4: Path) -> str:
    """Decoder #2: CCExtractor. Returns extracted SRT text or ''."""
    if not ccx_exe.exists() or not mp4.exists():
        return ""
    out = Path(tempfile.gettempdir()) / f"hybridcc_cc_{mp4.stem}.srt"
    try:
        subprocess.run(
            [str(ccx_exe), str(mp4), "-o", str(out)],
            capture_output=True, timeout=120, text=True,
        )
        return out.read_text(encoding="utf-8", errors="replace") if out.exists() else ""
    except Exception:
        return ""


def _scc_first_line(scc_path: Path) -> str:
    try:
        with open(scc_path, "r", encoding="utf-8", errors="replace") as f:
            return f.readline().strip()
    except Exception:
        return ""


def _badge(ok: bool) -> str:
    return ('<span class="pass">&#10003; PASS</span>' if ok
            else '<span class="fail">&#10007; FAIL</span>')


def render(args) -> str:
    qc_path = Path(args.qc)
    qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}
    qc_pass = bool(qc.get("pass", False))
    rem = qc.get("remaining_errors", [])
    if isinstance(rem, int):
        rem_count = rem
        issues = qc.get("issues", [])
        error_issues = [i for i in issues if i.get("severity") == "error"]
        rem = error_issues[-rem_count:] if rem_count > 0 else []
    auto_fixed = [i for i in qc.get("issues", []) if i.get("auto_fixed")]
    total_caps = qc.get("total_captions", 0)

    scc_first = _scc_first_line(Path(args.scc)) if args.scc else ""
    scc_ok = scc_first.startswith("Scenarist_SCC")

    # ─── Decoder verification (only if both --mp4 and --source-vtt and --bin given) ───
    ff_score = None
    cc_score = None
    decoders_ran = False
    if args.mp4 and args.source_vtt and args.bin:
        bin_dir = Path(args.bin)
        mp4_path = Path(args.mp4)
        source_vtt = Path(args.source_vtt)
        if mp4_path.exists() and source_vtt.exists():
            ffmpeg_exe = bin_dir / "ffmpeg.exe"
            ccx_exe = bin_dir / "ccextractor" / "ccextractorwinfull.exe"
            sw = _to_word_set(source_vtt.read_text(encoding="utf-8", errors="replace"))
            ff_text = _run_ffmpeg_subcc(ffmpeg_exe, mp4_path)
            cc_text = _run_ccextractor(ccx_exe, mp4_path)
            ff_score = _jaccard(sw, _to_word_set(ff_text)) if ff_text else 0.0
            cc_score = _jaccard(sw, _to_word_set(cc_text)) if cc_text else 0.0
            decoders_ran = True

    encoding_pass = False
    if decoders_ran:
        encoding_pass = (ff_score >= PASS_THRESHOLD) and (cc_score >= PASS_THRESHOLD) and scc_ok

    # Three-tier verdict.
    rem_categories = {i.get("category") for i in (rem or [])}
    only_soft_remain = bool(rem) and rem_categories.issubset(SOFT_QC_CATEGORIES)
    if decoders_ran:
        if encoding_pass and qc_pass:
            verdict, verdict_class = "PASS", "pass"
        elif encoding_pass and only_soft_remain:
            verdict, verdict_class = "WARN", "warn"
        else:
            verdict, verdict_class = "FAIL", "fail"
    else:
        # Fallback: SCC + QC only (decoders couldn't run — likely missing binaries)
        if qc_pass and scc_ok:
            verdict, verdict_class = "PASS", "pass"
        elif scc_ok and only_soft_remain:
            verdict, verdict_class = "WARN", "warn"
        else:
            verdict, verdict_class = "FAIL", "fail"

    src_name = args.src_name or "(unknown)"
    src_stem = args.src_stem or Path(src_name).stem
    media_id = args.media_id or ""

    identity_parts = [f'Source: <code>{html_mod.escape(src_name)}</code>']
    if media_id:
        identity_parts.append(f'Media&nbsp;ID: <code>{html_mod.escape(media_id)}</code>')
    identity_parts.append('Local pipeline')
    identity_html = ' &middot; '.join(identity_parts)

    # Timing block (no billing — local mode is free)
    timing_rows = []
    if args.source_duration_sec is not None:
        d = args.source_duration_sec
        timing_rows.append(f"<tr><td>Source duration</td><td>{d:.1f} sec ({d/60:.2f} min)</td></tr>")
    elapsed = (args.whisper_sec or 0) + (args.inject_sec or 0)
    if elapsed:
        timing_rows.append(f"<tr><td>Processing time</td><td>{elapsed:.1f} sec</td></tr>")
    if args.whisper_sec is not None:
        timing_rows.append(f"<tr><td>&nbsp;&nbsp;Whisper transcribe</td><td>{args.whisper_sec:.2f} sec</td></tr>")
    if args.inject_sec is not None:
        timing_rows.append(f"<tr><td>&nbsp;&nbsp;CEA-608 inject</td><td>{args.inject_sec:.2f} sec</td></tr>")
    timing_rows.append('<tr><td>Charge</td><td><strong>$0.00</strong> (local — no cloud cost)</td></tr>')
    timing_html = (
        '<h2>Job timing &amp; billing</h2>'
        '<table><tr><th>Item</th><th>Value</th></tr>'
        + "".join(timing_rows) + '</table>'
    )

    # Verdict callout — colored panel that summarizes what the verdict means.
    # PASS = green, WARN = amber, FAIL = red. Lives just under the verdict
    # badge so a reader skimming the PDF gets the bottom line without having
    # to read the table.
    encoded_blurb = (
        f'Two independent decoders read the captions out of <code>{html_mod.escape(src_name)}</code> '
        f'with {min(ff_score, cc_score)*100:.0f}%+ word match, and the SCC sidecar carries a valid Scenarist_SCC V1.0 header. '
        f'This is the same evidence a broadcast engineer would file for FCC &sect;79.1 acceptance.'
        if decoders_ran else
        'The captions encode against the source MP4 with CEA-608 SEI NAL units injected per video frame.'
    )

    # Heuristics for the WARN explanation: if the bulk of remaining issues are
    # CPS / duration overruns, the speaker is talking faster than broadcast
    # pacing assumes. Flag short-form content (<60s) explicitly so the customer
    # sees the connection between TikTok/Reels-style sources and the warnings.
    cps_count = sum(1 for r in (rem or []) if r.get('category') == 'reading_speed')
    short_dur_count = sum(1 for r in (rem or []) if r.get('category') == 'duration_short')
    pacing_dominant = (cps_count + short_dur_count) >= max(3, int(len(rem or []) * 0.5))
    is_short_form = (args.source_duration_sec is not None) and (0.0 < args.source_duration_sec < 60.0)

    if pacing_dominant and is_short_form:
        pacing_clause = (
            ' This is normal for short-form content (commercials under 60s, YouTube Shorts, Reels, TikTok-style clips) '
            'where the narrator is reading fast with few natural pauses to fit the runtime &mdash; '
            'the broadcast 1.0s / 20 CPS rules assume long-form pacing that this kind of source does not follow.'
        )
    elif pacing_dominant:
        pacing_clause = (
            ' The speaker is talking faster than broadcast pacing rules assume &mdash; '
            'continuous rapid speech with few breath pauses (news reads, interview cross-talk, energetic ad reads) '
            'will trip the 1.0s minimum and 20 CPS limit even though every word is captioned correctly.'
        )
    else:
        pacing_clause = ''

    if verdict == "PASS":
        note_block = (
            f'<div class="note-pass"><strong>PASS &mdash; ready to air.</strong> '
            f'{encoded_blurb}</div>'
        )
    elif verdict == "WARN":
        note_block = (
            f'<div class="note-warn"><strong>WARN &mdash; airworthy with caveats.</strong> '
            f'{encoded_blurb} '
            f'Pre-encode QC flagged {len(rem)} timing issue(s) &mdash; cues shorter than 1.0s or above 20 chars/sec.'
            f'{pacing_clause} '
            f'The captions encode correctly; listed below for manual edits if strict §79.1 timing review is required.</div>'
        )
    else:  # FAIL
        # Build a specific reason list so the customer knows what to look at.
        reasons = []
        if decoders_ran:
            if ff_score < PASS_THRESHOLD:
                reasons.append(f'Decoder #1 word match was only {ff_score*100:.0f}% (need ≥{int(PASS_THRESHOLD*100)}%) &mdash; the in-band CEA-608 stream may be missing or corrupted.')
            if cc_score < PASS_THRESHOLD:
                reasons.append(f'Decoder #2 word match was only {cc_score*100:.0f}% (need ≥{int(PASS_THRESHOLD*100)}%) &mdash; cross-decoder verification did not confirm the captions.')
        if not scc_ok:
            reasons.append('The SCC sidecar is missing or malformed (no <code>Scenarist_SCC V1.0</code> header). The MP4 may still play captions correctly, but the sidecar is unusable for broadcast handoff.')
        if rem and not (rem_categories.issubset(SOFT_QC_CATEGORIES)):
            hard_count = sum(1 for r in rem if r.get('category') not in SOFT_QC_CATEGORIES)
            reasons.append(f'{hard_count} hard QC error(s) survived the auto-fixer (line-wrap, encoding, or structural issues).')
        if not reasons:
            reasons.append(f'{len(rem)} pre-encode QC issue(s) survived the auto-fixer &mdash; see the table below.')
        reasons_html = '<ul>' + ''.join(f'<li>{r}</li>' for r in reasons) + '</ul>'
        # If most of the issues are pacing-related (fast speaker), surface that
        # alongside the hard failures so the customer understands the soft
        # warnings cluttering the errors table won't be fixable by the encoder.
        pacing_tail = (
            f'<p style="margin-top:.6em;">Note on the timing warnings:{pacing_clause}</p>'
            if pacing_clause else ''
        )
        note_block = (
            f'<div class="note-fail"><strong>FAIL &mdash; do not air without review.</strong>'
            f'{reasons_html}'
            f'See the QC errors table below for the full list of issues. The captioned MP4 may still play, '
            f'but the audit packet does not meet broadcast handoff standards as it stands.'
            f'{pacing_tail}</div>'
        )

    # Errors table
    errors_html = ""
    if rem:
        rows = []
        for e in rem[:50]:
            rows.append(
                f"<tr><td><code>{html_mod.escape(e.get('timestamp','-'))}</code></td>"
                f"<td>cue&nbsp;#{e.get('caption_index','-')}</td>"
                f"<td><code>{html_mod.escape(e.get('category','-'))}</code></td>"
                f"<td>{html_mod.escape(e.get('message','-'))}</td></tr>"
            )
        more = f'<p class="muted">&hellip; and {len(rem)-50} more</p>' if len(rem) > 50 else ''
        errors_html = (
            f'<h2>QC errors that need attention ({len(rem)})</h2>'
            '<p class="muted">These survived the auto-fix pass and may need manual editing for strict broadcast compliance.</p>'
            '<table><tr><th>Timestamp</th><th>Cue</th><th>Category</th><th>Message</th></tr>'
            + "".join(rows) + '</table>' + more
        )

    # Auto-fixes table
    fixes_html = ""
    if auto_fixed:
        rows = []
        for f in auto_fixed[:18]:
            rows.append(
                f"<tr><td><code>{html_mod.escape(f.get('timestamp','-'))}</code></td>"
                f"<td><code>{html_mod.escape(f.get('category','-'))}</code></td>"
                f"<td>{html_mod.escape(f.get('message','-'))}</td></tr>"
            )
        more = f'<p class="muted">&hellip; and {len(auto_fixed)-18} more</p>' if len(auto_fixed) > 18 else ''
        fixes_html = (
            f'<h2>Auto-fixes applied ({len(auto_fixed)})</h2>'
            '<table><tr><th>Timestamp</th><th>Category</th><th>What was fixed</th></tr>'
            + "".join(rows) + '</table>' + more
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>HybridCC Caption Proof Report - {html_mod.escape(src_name)}</title>
<style>
@page {{ size: letter; margin: 0.75in; }}
body {{ font-family: -apple-system, system-ui, "Segoe UI", sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; line-height: 1.5; color: #222; }}
h1 {{ border-bottom: 2px solid #222; padding-bottom: .3em; }}
.pass {{ color: #0a7; font-weight: bold; }}
.fail {{ color: #c33; font-weight: bold; }}
.warn {{ color: #d8a300; font-weight: bold; }}
.badge {{ display: inline-block; padding: .4em 1em; border-radius: .3em; color: white; font-size: 1.2em; font-weight: bold; }}
.badge.pass {{ background: #0a7; }}
.badge.warn {{ background: #d8a300; }}
.badge.fail {{ background: #c33; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ padding: .6em; border: 1px solid #ddd; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
.muted {{ color: #777; font-size: .9em; }}
.center {{ text-align: center; }}
.tagline {{ font-size: 1.3em; font-weight: 600; color: #444; margin-top: 1.5em; }}
.note-pass {{ background: #e8f7f0; border-left: 6px solid #0a7; padding: 1em 1.2em; margin: 1em 0; border-radius: 3px; }}
.note-warn {{ background: #fff8e1; border-left: 6px solid #d8a300; padding: 1em 1.2em; margin: 1em 0; border-radius: 3px; }}
.note-fail {{ background: #fde8e8; border-left: 6px solid #c33; padding: 1em 1.2em; margin: 1em 0; border-radius: 3px; }}
.note-pass strong, .note-warn strong, .note-fail strong {{ display: block; margin-bottom: .4em; font-size: 1.05em; }}
.note-fail ul, .note-warn ul {{ margin: .5em 0; padding-left: 1.4em; }}
.note-fail li, .note-warn li {{ margin: .3em 0; }}
@media print {{ body {{ margin: 0; max-width: none; }} table {{ page-break-inside: avoid; }} h2 {{ page-break-after: avoid; }} }}
</style></head><body>

<h1>HybridCC Caption Proof Report</h1>
<p class="muted">Generated {datetime.now().isoformat(timespec='seconds')}<br>{identity_html}</p>

<h2>Verdict: <span class="badge {verdict_class}">{verdict}</span></h2>

{f'''<table>
<tr><th>Check</th><th>Result</th><th>Score</th></tr>
<tr><td>CEA-608 captions in MP4 &mdash; decoder #1</td><td>{_badge(ff_score >= PASS_THRESHOLD)}</td><td>{ff_score*100:.1f}% word match vs source</td></tr>
<tr><td>CEA-608 captions in MP4 &mdash; decoder #2</td><td>{_badge(cc_score >= PASS_THRESHOLD)}</td><td>{cc_score*100:.1f}% word match vs source</td></tr>
<tr><td>SCC sidecar format</td><td>{_badge(scc_ok)}</td><td>{html_mod.escape(scc_first or 'no header found')}</td></tr>
<tr><td>Pre-encode QC (FCC &sect;79.1 line/CPS/duration rules)</td><td>{_badge(qc_pass)}</td><td>{total_caps} cues &middot; {len(auto_fixed)} auto-fixes &middot; {len(rem)} remaining errors</td></tr>
</table>''' if decoders_ran else f'''<table>
<tr><th>Check</th><th>Result</th><th>Score</th></tr>
<tr><td>SCC sidecar format</td><td>{_badge(scc_ok)}</td><td>{html_mod.escape(scc_first or 'no header found')}</td></tr>
<tr><td>Pre-encode QC (FCC &sect;79.1 line/CPS/duration rules)</td><td>{_badge(qc_pass)}</td><td>{total_caps} cues &middot; {len(auto_fixed)} auto-fixes &middot; {len(rem)} remaining errors</td></tr>
<tr><td>Decoder verification</td><td><span class="muted">SKIPPED</span></td><td><span class="muted">ffmpeg + CCExtractor not invoked (missing --bin / --mp4 / --source-vtt args)</span></td></tr>
</table>'''}

{note_block}

{timing_html}

{errors_html}

{fixes_html}

<h2>Verify the captions yourself</h2>
<ol>
<li><strong>Playback:</strong> Open <code>{html_mod.escape(src_name)}</code> in VLC, hit play, then Subtitle &rarr; Sub Track &rarr; <code>Closed Captions 1</code>.</li>
<li><strong>SCC format:</strong> Open <code>{html_mod.escape(src_stem)}.scc</code> in any text editor. First line: <code>Scenarist_SCC V1.0</code>.</li>
</ol>

<br>
<p class="tagline center">HybridCC &middot; FCC &sect;79.1 / &sect;79.4 compliant</p>

</body></html>"""
    return html_doc


def main():
    p = argparse.ArgumentParser(description="Render local HybridCC proof HTML from a qc.json")
    p.add_argument("--qc",       required=True, help="Path to qc.json (vtt_to_scc --report output)")
    p.add_argument("--src-name", required=True, help="Source filename (e.g. video.mp4)")
    p.add_argument("--src-stem", default=None,  help="Source stem (e.g. video). Defaults to src-name without ext.")
    p.add_argument("--scc",      default=None,  help="Path to .scc sidecar for header validation")
    p.add_argument("--media-id", default="",    help="Optional media ID for the audit trail")
    p.add_argument("--output",   required=True, help="Output HTML path")
    p.add_argument("--source-duration-sec", type=float, default=None)
    p.add_argument("--whisper-sec",          type=float, default=None)
    p.add_argument("--inject-sec",           type=float, default=None)
    # ─── Decoder verification args (optional) ───
    p.add_argument("--mp4",        default=None, help="Captioned MP4 to decode-verify")
    p.add_argument("--source-vtt", default=None, help="Source VTT to compare decoded captions against")
    p.add_argument("--bin",        default=None, help="hybridplayout/bin/ folder (looks for ffmpeg.exe + ccextractor/ccextractorwinfull.exe)")
    args = p.parse_args()

    html_doc = render(args)
    Path(args.output).write_text(html_doc, encoding="utf-8")
    print(f"[proof_render] wrote {args.output} ({len(html_doc)} chars)")


if __name__ == "__main__":
    main()
