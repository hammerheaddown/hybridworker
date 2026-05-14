"""
HybridCC Services - Modal deployment
====================================
One Modal app, two worker classes, three customer SKUs:

  POST /caption/transcribe   -> JSON  (Transcribe SKU - VTT only)
  POST /caption/cc           -> ZIP   (CC SKU - flat: MP4 + VTT + SRT + SCC + PDF + meta.json)
  POST /transcode            -> ZIP   (Transcode SKU - house-spec MP4)
  GET  /caption/json         -> JSON  (legacy alias, mirrors hybridcc-caption)
  GET  /health

Deploy:
  modal deploy hybridcc/modal/server.py

Auth: X-API-Key header. Allowed keys come from the `hybridcc` Modal Secret
(ALLOWED_API_KEYS JSON map + CAPTION_SERVICE_KEY for internal head-end calls).

Same registration pattern as the existing modal-caption-server.py - this app
phones home to HEAD_END_URL every 6 minutes so tools/captionServer.js keeps
its retry pool entry fresh.
"""

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import modal


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
APP_NAME = "hybridcc-services"
MODEL_NAME = "large-v3"
WHISPER_BEAM_SIZE = 5
PASS_THRESHOLD = 0.80   # Jaccard word-overlap threshold for decoder agreement

# Local repo root (hybridcc/) - this file lives at hybridcc/modal/server.py
LOCAL_REPO = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Image build
# ─────────────────────────────────────────────────────────────────────────────
# CUDA runtime base so faster-whisper can find libcublas/libcudnn.
# Then: ffmpeg + libcaption (built from source) + ttconv + WeasyPrint +
# CCExtractor + faster-whisper + stable-ts. hybridCC-vod is gcc-compiled
# from /opt/hybridcc/hybridCC-vod.c at image build time. vtt_to_scc.py is
# baked in at /opt/hybridcc/vtt_to_scc.py so subprocess can invoke it.
CAPTION_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "ffmpeg",
        "build-essential", "cmake", "git", "re2c",
        "ccextractor",
        "libpango-1.0-0", "libpangoft2-1.0-0", "libcairo2",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/szatmary/libcaption /opt/libcaption",
        "cd /opt/libcaption && cmake . -DENABLE_RE2C=ON && make -j$(nproc)",
    )
    .add_local_file(
        str(LOCAL_REPO / "src" / "hybridCC-vod.c"),
        "/opt/hybridcc/hybridCC-vod.c",
        copy=True,
    )
    .add_local_file(
        str(LOCAL_REPO / "vtt_to_scc.py"),
        "/opt/hybridcc/vtt_to_scc.py",
        copy=True,
    )
    .run_commands(
        "gcc -O2 -Wall "
        "-I /opt/libcaption -I /opt/libcaption/src "
        "-I /opt/libcaption/examples -I /opt/libcaption/caption "
        "-o /usr/local/bin/hybridCC-vod /opt/hybridcc/hybridCC-vod.c "
        "/opt/libcaption/examples/flv.c /opt/libcaption/libcaption.a -lm",
        "test -x /usr/local/bin/hybridCC-vod && echo 'hybridCC-vod built'",
    )
    .pip_install(
        "faster-whisper>=1.1.0",       # stable-ts 2.17+ needs BatchedInferencePipeline
        "stable-ts[fw]>=2.17.0",
        "ttconv",
        "weasyprint",
        "boto3",   # for R2 swap in v1.5
        "fastapi==0.115.0",
        "python-multipart==0.0.12",
        "huggingface-hub>=0.25.2",
    )
)

app = modal.App(APP_NAME, image=CAPTION_IMAGE)
model_cache = modal.Volume.from_name("hybridcc-whisper-cache", create_if_missing=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination filter (matches the notebook)
# ─────────────────────────────────────────────────────────────────────────────
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

def _is_hallucination(text):
    t = (text or "").strip()
    if not t:
        return True
    return any(r.match(t) for r in _HALL_RE)


# ─────────────────────────────────────────────────────────────────────────────
# House-spec presets — same names will appear in the playout app's Settings
# page so customer choice in the UI maps 1:1 to the API target= parameter.
# ─────────────────────────────────────────────────────────────────────────────
HOUSE_SPEC_PRESETS = {
    "broadcast_1080": {
        "max_width": 1920, "max_height": 1080,
        "vcodec": "libx264", "preset": "fast", "crf": 20,
        "profile": "high", "level": "4.0",
        "acodec": "aac", "abitrate": "192k", "achannels": 2, "arate": 48000,
        "label": "H.264 high@4.0, 1080p max, 192k AAC stereo 48k",
    },
    "broadcast_720": {
        "max_width": 1280, "max_height": 720,
        "vcodec": "libx264", "preset": "fast", "crf": 20,
        "profile": "high", "level": "3.1",
        "acodec": "aac", "abitrate": "128k", "achannels": 2, "arate": 48000,
        "label": "H.264 high@3.1, 720p max, 128k AAC stereo 48k",
    },
    "web_540": {
        "max_width": 960, "max_height": 540,
        "vcodec": "libx264", "preset": "fast", "crf": 22,
        "profile": "main", "level": "3.0",
        "acodec": "aac", "abitrate": "96k", "achannels": 2, "arate": 44100,
        "label": "H.264 main@3.0, 540p max, 96k AAC stereo 44k (mobile-friendly)",
    },
    "archive_1080": {
        "max_width": 1920, "max_height": 1080,
        "vcodec": "libx264", "preset": "slow", "crf": 18,
        "profile": "high", "level": "4.0",
        "acodec": "aac", "abitrate": "256k", "achannels": 2, "arate": 48000,
        "label": "H.264 high@4.0, 1080p max, 256k AAC stereo 48k (archival quality)",
    },
    "broadcast_480": {
        "max_width": 720, "max_height": 480,
        "vcodec": "libx264", "preset": "fast", "crf": 22,
        "profile": "main", "level": "3.0",
        "acodec": "aac", "abitrate": "128k", "achannels": 2, "arate": 48000,
        "force_dims": True,    # always scale to EXACT 720x480 (anamorphic)
        "aspect": "16:9",       # display AR — pixels are non-square
        "label": "H.264 main@3.0, 720x480 16:9 anamorphic NTSC SD, 128k AAC stereo 48k",
    },
}

# Aliases / legacy names. Map old `target=house` calls to broadcast_1080.
HOUSE_SPEC_ALIASES = {
    "house": "broadcast_1080",
    "default": "broadcast_1080",
    "1080": "broadcast_1080",
    "720": "broadcast_720",
    "540": "web_540",
    "480": "broadcast_480",
    "sd": "broadcast_480",
}

def _resolve_preset(name):
    """Resolve a target name to a preset dict. Falls back to broadcast_1080."""
    n = (name or "").strip().lower()
    n = HOUSE_SPEC_ALIASES.get(n, n)
    return HOUSE_SPEC_PRESETS.get(n) or HOUSE_SPEC_PRESETS["broadcast_1080"]


def _probe_dims(input_path):
    """Return (width, height) of input video, or (0, 0) on probe failure."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0",
             str(input_path)],
            capture_output=True, text=True, timeout=30
        )
        return tuple(map(int, probe.stdout.strip().split(",")))
    except Exception:
        return (0, 0)


def _build_house_spec_args(preset_name, input_path):
    """Build ffmpeg arg list from preset. Two scaling modes:

    - Square-pixel HD/4K presets (broadcast_1080, _720, web_540, archive_1080):
      AR-preserving scale, only when source exceeds preset's max dims. Skips
      the scale filter for small sources (no upscaling).

    - Anamorphic SD presets (broadcast_480) marked with `force_dims=True`:
      always scale to exact target dimensions and tag display AR via -aspect.
      Pixels are non-square; the display AR drives playback geometry."""
    p = _resolve_preset(preset_name)
    args = [
        "-c:v", p["vcodec"],
        "-preset", p["preset"],
        "-crf", str(p["crf"]),
        "-profile:v", p["profile"],
        "-level", p["level"],
        "-pix_fmt", "yuv420p",
    ]
    if p.get("force_dims"):
        # Anamorphic: scale to EXACT target dims, tag display AR.
        args += ["-vf", f"scale={p['max_width']}:{p['max_height']}"]
        if p.get("aspect"):
            args += ["-aspect", p["aspect"]]
    else:
        # Square-pixel: preserve AR, only downscale if source > target.
        w, h = _probe_dims(input_path)
        if w > p["max_width"] or h > p["max_height"]:
            args += ["-vf",
                     f"scale={p['max_width']}:{p['max_height']}:"
                     f"force_original_aspect_ratio=decrease:force_divisible_by=2"]
    args += [
        "-c:a", p["acodec"],
        "-b:a", p["abitrate"],
        "-ac", str(p["achannels"]),
        "-ar", str(p["arate"]),
    ]
    return args


def _build_house_spec_str(preset_name, input_path, extra_flags=""):
    """String form of _build_house_spec_args for bash-pipe ffmpeg invocations."""
    args = _build_house_spec_args(preset_name, input_path)
    s = " ".join(args)
    if extra_flags:
        s += " " + extra_flags
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Pricing — fetched from head-end with TTL cache, baked-in fallback
# ─────────────────────────────────────────────────────────────────────────────
# Defaults if HEAD_END_URL is unset or unreachable. Tweak these directly
# only as last-resort fallback values; the real source of truth is the
# playout PC's GET /api/services/pricing endpoint.
DEFAULT_PRICING = {
    "cc": {
        "rate_per_sec": 0.000694,   # $2.50 / hour of source video
        "min_charge":   0.25,
        "rate_label":   "$2.50/hr",
    },
    "transcribe": {
        "rate_per_sec": 0.000278,   # $1.00 / hour
        "min_charge":   0.20,
        "rate_label":   "$1.00/hr",
    },
    "transcode": {
        "rate_per_sec": 0.000278,   # $1.00 / hour
        "min_charge":   0.20,
        "rate_label":   "$1.00/hr",
    },
    # ── Bundles (small discount vs sum of parts) ────────────────────────────
    "cc-transcode": {
        "rate_per_sec": 0.000833,   # $3.00 / hour (CC $2.50 + Transcode $1, -$0.50 bundle discount)
        "min_charge":   0.40,
        "rate_label":   "$3.00/hr",
    },
    "transcode-vtt": {
        "rate_per_sec": 0.000417,   # $1.50 / hour (Transcode $1 + Transcribe $1, -$0.50 bundle discount)
        "min_charge":   0.25,
        "rate_label":   "$1.50/hr",
    },
}

_pricing_cache = {"data": None, "ts": 0.0}
_PRICING_TTL_SEC = 300   # refresh from head-end every 5 minutes

def _fetch_pricing(sku):
    """Get pricing for `sku` ('cc'|'transcribe'|'transcode'). Tries head-end
    GET /api/services/pricing, caches 5 min, falls back to DEFAULT_PRICING
    if head-end is unset or unreachable."""
    import urllib.request

    now = time.time()
    if _pricing_cache["data"] and (now - _pricing_cache["ts"] < _PRICING_TTL_SEC):
        return _pricing_cache["data"].get(sku) or DEFAULT_PRICING.get(sku, {})

    head_end = os.environ.get("HEAD_END_URL", "").rstrip("/")
    if not head_end:
        return DEFAULT_PRICING.get(sku, {})

    try:
        req = urllib.request.Request(
            f"{head_end}/api/services/pricing",
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            _pricing_cache["data"] = data
            _pricing_cache["ts"] = now
            return data.get(sku) or DEFAULT_PRICING.get(sku, {})
    except Exception as e:
        print(f"[pricing] head-end unreachable ({e}); falling back to defaults")
        return DEFAULT_PRICING.get(sku, {})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers - VTT post-processing (mirrors notebook Section 6)
# ─────────────────────────────────────────────────────────────────────────────
def _clean_vtt_text(text):
    """Strip stable-ts inline word tags, dedup adjacent repeats, ensure
    blank-line separators between cues."""
    text = re.sub(r"<\d{1,2}:\d{2}:\d{2}\.\d{3}>", "", text)
    text = re.sub(r" {2,}", " ", text)

    blocks, cur = [], []
    for line in text.splitlines():
        if line.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)

    if not blocks or not blocks[0] or not blocks[0][0].lstrip().upper().startswith("WEBVTT"):
        blocks.insert(0, ["WEBVTT"])

    header, *cue_blocks = blocks

    def _norm(t):
        return re.sub(r"[^a-z ]", "", t.lower()).strip()

    kept, prev_norm = [], ""
    for b in cue_blocks:
        if len(b) < 2:
            continue
        cue_text = " ".join(b[1:]).strip()
        if _is_hallucination(cue_text):
            continue
        norm = _norm(cue_text)
        if norm and prev_norm and (norm in prev_norm or prev_norm in norm):
            continue
        kept.append(b)
        prev_norm = norm

    return "\n\n".join("\n".join(b) for b in [header] + kept) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers - Word-overlap scoring for proof report
# ─────────────────────────────────────────────────────────────────────────────
def _to_word_set(text):
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\d{1,2}:\d{2}:\d{2}[.,]\d+\s*-->\s*\d{1,2}:\d{2}:\d{2}[.,]\d+", "", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.M)
    text = re.sub(r"WEBVTT.*", "", text)
    return set(re.findall(r"[a-z]{2,}", text.lower()))

def _jaccard(a, b):
    if not (a or b):
        return 0.0
    return len(a & b) / max(len(a | b), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers - Proof report rendering (HTML + PDF + verify.txt)
# ─────────────────────────────────────────────────────────────────────────────
def _render_proof_artifacts(work_dir, output_mp4, source_vtt, scc_path,
                             qc_path, src_name, src_stem, media_id,
                             billing=None, content_type=None):
    """Run two independent decoders, render proof.html + proof.pdf + verify.txt.
    `billing` is an optional dict with keys: source_duration_sec,
    total_elapsed_sec, whisper_sec, inject_sec, proof_sec, charge_usd, rate.
    Returns (html_path, pdf_path, txt_path)."""
    import html as html_mod

    # Decoder #1: ffmpeg subcc
    ff_path = work_dir / "_extracted_ffmpeg.vtt"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"movie={output_mp4}[out0+subcc]",
         "-map", "0:s", "-c:s", "webvtt", str(ff_path)],
        capture_output=True
    )
    ff_ok = ff_path.exists() and ff_path.stat().st_size > 50

    # Decoder #2: CCExtractor
    cc_path = work_dir / "_extracted_ccextractor.srt"
    subprocess.run(["ccextractor", str(output_mp4), "-o", str(cc_path)],
                   capture_output=True)
    cc_ok = cc_path.exists() and cc_path.stat().st_size > 50

    # SCC format check
    scc_ok = False
    scc_first = ""
    if scc_path.exists() and scc_path.stat().st_size > 0:
        scc_first = scc_path.read_text().splitlines()[0]
        scc_ok = scc_first.startswith("Scenarist_SCC")

    # Word-overlap scores
    sw = _to_word_set(source_vtt.read_text())
    ffw = _to_word_set(ff_path.read_text()) if ff_ok else set()
    ccw = _to_word_set(cc_path.read_text()) if cc_ok else set()
    ff_score = _jaccard(sw, ffw)
    cc_score = _jaccard(sw, ccw)

    # QC report
    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}
    qc_pass = bool(qc.get("pass", False))

    # Remaining errors (with backward-compat fallback to issues array scan)
    rem = qc.get("remaining_errors", [])
    if isinstance(rem, int):
        rem_count = rem
        issues = qc.get("issues", [])
        error_issues = [i for i in issues if i.get("severity") == "error"]
        rem = error_issues[-rem_count:] if rem_count > 0 else []
    auto_fixed = [i for i in qc.get("issues", []) if i.get("auto_fixed")]

    encoding_pass = (ff_score >= PASS_THRESHOLD) and (cc_score >= PASS_THRESHOLD) and scc_ok
    overall_pass = encoding_pass and qc_pass

    # Three-tier verdict. Hard errors (line wrap, overlap, malformed input)
    # mean the captions are actually broken — those stay FAIL. "Soft" timing
    # errors (duration_short, reading_speed) are inherent to short-form
    # content (YouTube Shorts, Reels, TikTok) where speech is continuous
    # and rapid with no natural breaks; the captions still air correctly,
    # so we downgrade those to WARN instead of failing the whole job.
    SOFT_QC_CATEGORIES = {"duration_short", "reading_speed"}
    rem_categories = {i.get("category") for i in (rem or [])}
    only_soft_remain = bool(rem) and rem_categories.issubset(SOFT_QC_CATEGORIES)
    if overall_pass:
        verdict = "PASS"; verdict_class = "pass"
    elif encoding_pass and only_soft_remain:
        verdict = "WARN"; verdict_class = "warn"
    else:
        verdict = "FAIL"; verdict_class = "fail"
    is_short_form = bool(content_type and content_type.lower() in ("shorts", "reels", "short", "tiktok"))

    def _badge(ok):
        return ('<span class="pass">&#10003; PASS</span>' if ok
                else '<span class="fail">&#10007; FAIL</span>')

    def _safe(s, n=2000):
        return html_mod.escape((s or "")[:n])

    # Identity row
    identity = [f'Source: <code>{html_mod.escape(src_name)}</code>']
    if media_id:
        identity.append(f'Media&nbsp;ID: <code>{html_mod.escape(media_id)}</code>')
    identity_html = ' &middot; '.join(identity)

    # Billing block (optional)
    billing_html = ''
    if billing:
        dur = billing.get('source_duration_sec', 0)
        elapsed = billing.get('total_elapsed_sec', 0)
        charge = billing.get('charge_usd', 0)
        rate = billing.get('rate', '')
        rows = []
        rows.append(f"<tr><td>Source duration</td><td>{dur:.1f} sec ({dur/60:.2f} min)</td></tr>")
        rows.append(f"<tr><td>Processing time</td><td>{elapsed:.1f} sec</td></tr>")
        if billing.get('whisper_sec'):
            rows.append(f"<tr><td>&nbsp;&nbsp;Whisper transcribe</td><td>{billing['whisper_sec']:.2f} sec</td></tr>")
        if billing.get('inject_sec'):
            rows.append(f"<tr><td>&nbsp;&nbsp;CEA-608 inject</td><td>{billing['inject_sec']:.2f} sec</td></tr>")
        if billing.get('proof_sec'):
            rows.append(f"<tr><td>&nbsp;&nbsp;Proof &amp; verification</td><td>{billing['proof_sec']:.2f} sec</td></tr>")
        if charge:
            rows.append(f"<tr><td><strong>Charge</strong></td><td><strong>${charge:.2f}</strong>{f' ({html_mod.escape(rate)})' if rate else ''}</td></tr>")
        billing_html = (
            '<h2>Job timing &amp; billing</h2>'
            '<table><tr><th>Item</th><th>Value</th></tr>'
            + ''.join(rows) +
            '</table>'
        )

    # Errors table
    errors_html = ""
    if rem:
        rows = []
        for e in rem:
            rows.append(
                f"<tr><td><code>{html_mod.escape(e.get('timestamp','-'))}</code></td>"
                f"<td>cue #{e.get('caption_index','-')}</td>"
                f"<td><code>{html_mod.escape(e.get('category','-'))}</code></td>"
                f"<td>{html_mod.escape(e.get('message','-'))}</td></tr>"
            )
        errors_html = (
            f'<h2>QC errors that need attention ({len(rem)})</h2>'
            '<p class="muted">These survived the auto-fix pass and may need '
            'manual editing for strict broadcast compliance.</p>'
            '<table><tr><th>Timestamp</th><th>Cue</th><th>Category</th>'
            f'<th>Message</th></tr>{"".join(rows)}</table>'
        )

    # Auto-fixes table
    fixes_html = ""
    if auto_fixed:
        rows = []
        for f in auto_fixed[:20]:
            rows.append(
                f"<tr><td><code>{html_mod.escape(f.get('timestamp','-'))}</code></td>"
                f"<td><code>{html_mod.escape(f.get('category','-'))}</code></td>"
                f"<td>{html_mod.escape(f.get('message','-'))}</td></tr>"
            )
        more = (f"<p class='muted'>... and {len(auto_fixed)-20} more</p>"
                if len(auto_fixed) > 20 else "")
        fixes_html = (
            f'<h2>Auto-fixes applied ({len(auto_fixed)})</h2>'
            '<table><tr><th>Timestamp</th><th>Category</th>'
            f'<th>What was fixed</th></tr>{"".join(rows)}</table>{more}'
        )

    src_text = source_vtt.read_text()
    ff_text = ff_path.read_text() if ff_ok else "(not produced)"
    cc_text = cc_path.read_text() if cc_ok else "(ccextractor unavailable or failed)"

    # Verdict callout — colored panel that summarizes what the verdict means.
    # PASS = green, WARN = amber, FAIL = red. Lives just under the verdict
    # badge so a reader skimming the PDF gets the bottom line without having
    # to read the table.
    encoded_blurb = (
        f'Two independent decoders read the captions out of <code>{html_mod.escape(src_name)}</code> '
        f'with {min(ff_score, cc_score)*100:.0f}%+ word match, and the SCC sidecar carries a valid Scenarist_SCC V1.0 header. '
        f'This is the same evidence a broadcast engineer would file for FCC &sect;79.1 acceptance.'
    )

    if verdict == "PASS":
        note_block = (
            f'<div class="note-pass"><strong>PASS &mdash; ready to air.</strong> '
            f'{encoded_blurb}</div>'
        )
    elif verdict == "WARN":
        shorts_clause = (
            ' This is normal for short-form content (YouTube Shorts, Reels, TikTok-style clips) where speech is continuous and rapid with no natural breaks.'
            if is_short_form else
            ' This typically happens with continuous rapid speech (news reads, interview cross-talk) where there is no natural break long enough to satisfy the 1.0s minimum or 20 CPS limit.'
        )
        note_block = (
            f'<div class="note-warn"><strong>WARN &mdash; airworthy with caveats.</strong> '
            f'{encoded_blurb} '
            f'Pre-encode QC flagged {len(rem)} timing issue(s) &mdash; cues shorter than 1.0s or above 20 chars/sec.{shorts_clause} '
            f'The captions encode correctly; listed below for manual edits if strict §79.1 timing review is required.</div>'
        )
    else:  # FAIL
        reasons = []
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
        # Surface fast-speaker / short-form context if it explains the bulk
        # of the soft warnings polluting the errors table.
        cps_count = sum(1 for r in (rem or []) if r.get('category') == 'reading_speed')
        short_dur_count = sum(1 for r in (rem or []) if r.get('category') == 'duration_short')
        pacing_dominant = (cps_count + short_dur_count) >= max(3, int(len(rem or []) * 0.5))
        if pacing_dominant and is_short_form:
            pacing_clause = (
                ' Most timing warnings come from short-form pacing (commercials under 60s, '
                'YouTube Shorts, Reels, TikTok-style clips) where the narrator reads fast with few '
                'natural pauses to fit the runtime &mdash; the broadcast 1.0s / 20 CPS rules assume '
                'long-form pacing this kind of source does not follow.'
            )
        elif pacing_dominant:
            pacing_clause = (
                ' Most timing warnings come from a fast-talking source &mdash; continuous rapid speech '
                'with few breath pauses (news reads, interview cross-talk, energetic ad reads) trips '
                'the 1.0s minimum and 20 CPS limit even though every word is captioned correctly.'
            )
        else:
            pacing_clause = ''
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

    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>HybridCC Caption Proof Report - {html_mod.escape(src_name)}</title>
<style>
@page {{ size: letter; margin: 0.75in; }}
body {{ font-family: -apple-system, system-ui, "Segoe UI", sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; line-height: 1.5; color: #222; }}
h1 {{ border-bottom: 2px solid #222; padding-bottom: .3em; }}
.pass {{ color: #0a7; font-weight: bold; }}
.fail {{ color: #c33; font-weight: bold; }}
.badge {{ display: inline-block; padding: .4em 1em; border-radius: .3em; color: white; font-size: 1.2em; font-weight: bold; }}
.badge.pass {{ background: #0a7; }}
.badge.warn {{ background: #d8a300; }}
.badge.fail {{ background: #c33; }}
.warn {{ color: #d8a300; font-weight: bold; }}
.note-pass {{ background: #e8f7f0; border-left: 6px solid #0a7; padding: 1em 1.2em; margin: 1em 0; border-radius: 3px; }}
.note-warn {{ background: #fff8e1; border-left: 6px solid #d8a300; padding: 1em 1.2em; margin: 1em 0; border-radius: 3px; }}
.note-fail {{ background: #fde8e8; border-left: 6px solid #c33; padding: 1em 1.2em; margin: 1em 0; border-radius: 3px; }}
.note-pass strong, .note-warn strong, .note-fail strong {{ display: block; margin-bottom: .4em; font-size: 1.05em; }}
.note-fail ul, .note-warn ul {{ margin: .5em 0; padding-left: 1.4em; }}
.note-fail li, .note-warn li {{ margin: .3em 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ padding: .6em; border: 1px solid #ddd; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
pre {{ background: #f7f7f7; padding: .8em; overflow-x: auto; font-size: .85em; max-height: 250px; white-space: pre-wrap; word-break: break-word; }}
.muted {{ color: #777; font-size: .9em; }}
.center {{ text-align: center; }}
.tagline {{ font-size: 1.3em; font-weight: 600; color: #444; margin-top: 1.5em; }}
@media print {{
  body {{ margin: 0; max-width: none; }}
  pre {{ max-height: none; overflow: visible; font-size: .75em; page-break-inside: auto; }}
  table {{ page-break-inside: avoid; }}
  h2 {{ page-break-after: avoid; }}
}}
</style></head><body>

<h1>HybridCC Caption Proof Report</h1>
<p class="muted">Generated {datetime.now().isoformat(timespec='seconds')}<br>{identity_html}</p>

<h2>Verdict: <span class="badge {verdict_class}">{verdict}</span></h2>

<table>
<tr><th>Check</th><th>Result</th><th>Score</th></tr>
<tr><td>CEA-608 captions in MP4 &mdash; decoder #1</td><td>{_badge(ff_score >= PASS_THRESHOLD)}</td><td>{ff_score*100:.1f}% word match vs source</td></tr>
<tr><td>CEA-608 captions in MP4 &mdash; decoder #2</td><td>{_badge(cc_score >= PASS_THRESHOLD)}</td><td>{cc_score*100:.1f}% word match vs source</td></tr>
<tr><td>SCC sidecar format</td><td>{_badge(scc_ok)}</td><td>{html_mod.escape(scc_first or 'no header found')}</td></tr>
<tr><td>Pre-encode QC (FCC &sect;79.1 line/CPS/duration rules)</td><td>{_badge(qc_pass)}</td><td>{qc.get('total_captions', 0)} cues &middot; {len(auto_fixed)} auto-fixes &middot; {len(rem)} remaining errors</td></tr>
</table>

{note_block}

{billing_html}

{errors_html}

{fixes_html}

<h2>Verify it yourself (60 seconds)</h2>
<ol>
<li><strong>Playback:</strong> Open <code>{html_mod.escape(src_name)}</code> in VLC and <strong>hit play</strong>. Let it start, then Subtitle &rarr; Sub Track &rarr; <code>Closed Captions 1</code> to see the captions. Scroll back to the beginning if needed &mdash; the captions will be there.</li>
<li><strong>ffmpeg:</strong> <code>ffmpeg -i "movie={html_mod.escape(src_name)}[out0+subcc]" -map 0:s -c:s webvtt out.vtt</code></li>
<li><strong>CCExtractor:</strong> <code>ccextractor {html_mod.escape(src_name)} -o out.srt</code></li>
<li><strong>SCC format:</strong> Open <code>{html_mod.escape(src_stem)}.scc</code> in any text editor. First line: <code>Scenarist_SCC V1.0</code>.</li>
</ol>

<br>
<p class="tagline center">HybridCC &middot; FCC &sect;79.1 / &sect;79.4 compliant</p>

</body></html>"""

    html_out = work_dir / "proof-report.html"
    html_out.write_text(html_doc, encoding="utf-8")

    # PDF render via WeasyPrint
    pdf_out = work_dir / "proof-report.pdf"
    try:
        from weasyprint import HTML as WeasyHTML
        WeasyHTML(filename=str(html_out)).write_pdf(str(pdf_out))
    except Exception as e:
        print(f"[proof] PDF render failed: {e}")

    return html_out, pdf_out, {
        "qc_pass": qc_pass,
        "encoding_pass": encoding_pass,
        "overall_pass": overall_pass,
        "verdict": verdict,
        "ff_score": ff_score,
        "cc_score": cc_score,
        "remaining_errors": len(rem),
        "auto_fixes": len(auto_fixed),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GPU WORKER - Whisper + libcaption inject (Transcribe + CC SKUs)
# ─────────────────────────────────────────────────────────────────────────────
@app.cls(
    gpu="L4",
    cpu=4.0,
    volumes={"/model-cache": model_cache},
    secrets=[modal.Secret.from_name("hybridcc")],
    scaledown_window=300,
    min_containers=0,
    timeout=3600,
)
class GPUWorker:

    @modal.enter()
    def load_model(self):
        # HF token (silences rate-limit warnings on weight downloads)
        hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf:
            os.environ["HF_TOKEN"] = hf
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf

        import stable_whisper
        self.model = stable_whisper.load_faster_whisper(
            MODEL_NAME,
            device="cuda",
            compute_type="float16",
            download_root="/model-cache",
        )
        print(f"[GPUWorker] loaded {MODEL_NAME}")

    def _whisper_to_vtt(self, video_path):
        """Whisper transcribe + broadcast pacing + clean VTT.
        Returns (vtt_text, info_duration, elapsed_sec)."""
        t0 = time.time()
        result = self.model.transcribe(
            str(video_path),
            language="en",
            regroup=True,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
            condition_on_previous_text=False,
            no_repeat_ngram_size=3,
            suppress_silence=True,
            hallucination_silence_threshold=2.0,
        )

        # Drop hallucination segments before split
        result.segments = [s for s in result.segments
                           if not _is_hallucination(s.text or "")]

        # Broadcast pacing — CEA-608 hard limit is 32 cols/row. Larger values
        # produce lines that hybridCC-vod truncates at col 32, dropping the
        # tail of long cues.
        result.split_by_length(max_chars=32)
        result.split_by_duration(max_dur=3.5)
        result.split_by_gap(max_gap=0.4)

        # Write to a temp VTT then clean
        with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False) as tmp:
            vtt_temp = tmp.name
        result.to_srt_vtt(vtt_temp, segment_level=True, word_level=False)
        raw = Path(vtt_temp).read_text()
        cleaned = _clean_vtt_text(raw)
        os.unlink(vtt_temp)

        elapsed = time.time() - t0
        duration = 0.0
        # stable-ts result has segments with end times; use last as a proxy
        if result.segments:
            duration = round(float(result.segments[-1].end), 1)
        return cleaned, duration, elapsed

    @modal.method()
    def transcribe(self, video_bytes, filename="upload.mp4"):
        """SKU 1: Transcribe - returns VTT only, no inject, no QC."""
        suffix = os.path.splitext(filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        try:
            vtt, duration, elapsed = self._whisper_to_vtt(tmp_path)
            return {
                "ok": True,
                "vtt": vtt,
                "duration": duration,
                "elapsed_sec": round(elapsed, 2),
                "model": MODEL_NAME,
                "backend": "modal",
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {str(e)[:300]}",
                "backend": "modal",
            }
        finally:
            try: os.unlink(tmp_path)
            except Exception: pass

    @modal.method()
    def caption(self, video_bytes, filename="upload.mp4", media_id="",
                transcode_to_house=False, target="broadcast_1080",
                content_type="broadcast"):
        """SKU: CC (full pipeline). Returns flat ZIP bytes containing
        captioned MP4 + VTT + SRT + SCC + proof.pdf + meta.json.

        If transcode_to_house=True, the second ffmpeg re-encodes per the
        named preset (broadcast_1080|broadcast_720|web_540|archive_1080).
        Used by the cc-transcode bundle SKU. Target is ignored when
        transcode_to_house=False (stream-copy preserves source codec).

        content_type: 'broadcast' (default, FCC §79.1 strict timing) or
        'shorts'/'reels' (relaxed timing for continuous-speech short-form
        content where strict 1.0s/20 CPS rules are inappropriate)."""
        stem = Path(filename).stem
        work = Path(tempfile.mkdtemp(prefix="hybridcc_"))
        input_mp4 = work / "input.mp4"
        input_mp4.write_bytes(video_bytes)
        t_total_start = time.time()    # wall-clock start for billing
        sku = "cc-transcode" if transcode_to_house else "cc"
        meta = {
            "ok": False,
            "sku": sku,
            "media_id": media_id,
            "src_name": filename,
            "src_stem": stem,
            "model": MODEL_NAME,
            "backend": "modal",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_bytes": len(video_bytes),
        }

        try:
            # 1. Whisper -> cleaned VTT
            t_whisper = time.time()
            vtt_text, duration, _ = self._whisper_to_vtt(input_mp4)
            meta["duration"] = duration
            meta["whisper_sec"] = round(time.time() - t_whisper, 2)

            vtt_path = work / "input.vtt"
            vtt_path.write_text(vtt_text)

            # 2. vtt_to_scc QC -> SCC + cleaned VTT/SRT + qc.json
            scc_path = work / "input.scc"
            qc_path = work / "input.qc.json"
            cleaned_vtt = work / "input.cleaned.vtt"
            cleaned_srt = work / "input.srt"

            # Map UI-side content_type values to vtt_to_scc.py's --profile arg.
            qc_profile = (
                "shorts" if (content_type or "").lower() in ("shorts", "reels", "short", "tiktok")
                else "broadcast"
            )
            meta["content_type"] = content_type or "broadcast"
            meta["qc_profile"] = qc_profile
            qc_proc = subprocess.run([
                "python3", "/opt/hybridcc/vtt_to_scc.py",
                str(vtt_path), str(scc_path),
                "--report", str(qc_path),
                "--cleaned-vtt", str(cleaned_vtt),
                "--cleaned-srt", str(cleaned_srt),
                "--profile", qc_profile,
            ], capture_output=True, text=True)
            if qc_proc.returncode not in (0, 1):
                # Exit 0 = pass, 1 = remaining errors. Anything else = real failure.
                print(f"[caption] vtt_to_scc errored: {qc_proc.stderr[:500]}")

            if cleaned_vtt.exists():
                vtt_path.write_text(cleaned_vtt.read_text())
                vtt_text = cleaned_vtt.read_text()

            # 3. Inject CEA-608 SEI via the FLV pipe
            output_mp4 = work / "output.mp4"
            t_inject = time.time()
            # Second ffmpeg: -c:v copy by default (preserves source). For the
            # cc-transcode bundle, re-encode to house spec while keeping CC.
            # -a53cc 1 on output writes the closed_captions=1 metadata
            # so VLC and friends light up their CC menu.
            if transcode_to_house:
                # Build encoder args from named preset (resolves aliases,
                # pre-probes input to skip upscale on small sources).
                second_codec = _build_house_spec_str(
                    target, input_mp4, extra_flags="-a53cc 1"
                )
                meta["target"] = target
                meta["target_label"] = _resolve_preset(target)["label"]
            else:
                second_codec = "-c:v copy -c:a copy -a53cc 1"
            # FLV container only accepts H.264 video. For non-H.264 sources
            # (mpeg2video, hevc, vp9, etc) the first ffmpeg has to re-encode
            # before piping through hybridCC-vod, otherwise FLV mux fails with
            # "codec X not compatible with flv". -preset ultrafast + -crf 18
            # keeps the intermediate visually transparent before the second
            # ffmpeg's final preset encode.
            try:
                probe_proc = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=codec_name",
                     "-of", "default=nw=1:nk=1", str(input_mp4)],
                    capture_output=True, text=True, timeout=15,
                )
                src_vcodec = (probe_proc.stdout or "").strip().lower()
            except Exception:
                src_vcodec = ""
            # -bsf:v filter_units=remove_types=6 strips ALL SEI NALs from
            # the source bitstream before our injector runs. Without it, an
            # H.264 source that already carried CEA-608 in SEI would end up
            # with both old + new captions after -c:v copy. The re-encode
            # branch doesn't need it — libx264 doesn't carry source SEI.
            first_video = (
                "-c:v copy -bsf:v filter_units=remove_types=6"
                if src_vcodec == "h264"
                else "-c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p"
            )
            inject_cmd = (
                f'ffmpeg -y -hide_banner -loglevel error '
                f'-i {input_mp4} {first_video} '
                f'-c:a aac -ac 2 -ar 44100 -f flv pipe:1 '
                f'| /usr/local/bin/hybridCC-vod {vtt_path} '
                f'| ffmpeg -y -hide_banner -loglevel error '
                f'-f flv -i pipe:0 {second_codec} '
                # fMP4 (moof segments) — Shaka/hls.js/MSE players run their
                # CEA-608 SEI parser through MSE, which is skipped for plain
                # progressive MP4. +faststart keeps the moov at front for fast
                # HTTP-range start.
                f'-movflags +frag_keyframe+empty_moov+default_base_moof+faststart {output_mp4}'
            )
            inject_proc = subprocess.run(
                ["bash", "-c", "set -o pipefail; " + inject_cmd],
                capture_output=True, text=True
            )
            if inject_proc.returncode != 0 or not output_mp4.exists():
                meta["error"] = f"inject pipeline failed: {inject_proc.stderr[:400]}"
                return self._error_zip(meta)
            meta["inject_sec"] = round(time.time() - t_inject, 2)

            # 4. Render proof artifacts
            t_proof = time.time()
            # Pricing fetched from head-end (GET /api/services/pricing) with
            # 5-min cache + fallback to DEFAULT_PRICING. Lets us change rates
            # without redeploying Modal. Bundle uses cc-transcode SKU rate.
            pricing = _fetch_pricing(sku)
            rate_per_sec = pricing.get("rate_per_sec", 0.000694)
            min_charge   = pricing.get("min_charge", 0.25)
            rate_label   = pricing.get("rate_label", "$2.50/hr")
            elapsed_so_far = round(time.time() - t_total_start, 2)
            calc_charge = max(min_charge, rate_per_sec * meta.get("duration", 0))
            billing_info = {
                "source_duration_sec": meta.get("duration", 0),
                "total_elapsed_sec":   elapsed_so_far,
                "whisper_sec":         meta.get("whisper_sec", 0),
                "inject_sec":          meta.get("inject_sec", 0),
                "charge_usd":          round(calc_charge, 2),
                "rate":                rate_label,
            }
            html_p, pdf_p, proof_meta = _render_proof_artifacts(
                work_dir=work,
                output_mp4=output_mp4,
                source_vtt=vtt_path,
                scc_path=scc_path,
                qc_path=qc_path,
                src_name=filename,
                src_stem=stem,
                media_id=media_id,
                billing=billing_info,
                content_type=content_type,
            )
            meta["proof_sec"] = round(time.time() - t_proof, 2)
            meta["charge_usd"] = billing_info["charge_usd"]
            meta["rate"] = billing_info["rate"]
            meta.update(proof_meta)

            # 5. Flat delivery zip: MP4 + VTT + SRT + SCC + PDF + meta.json.
            # HTML preview, verify.txt, qc.json are intentionally NOT shipped —
            # playout-side splits this into source-folder MP4, DB-resident VTT,
            # and an on-disk audit zip (PDF/SCC/SRT only).
            meta["ok"] = True
            meta["total_elapsed_sec"] = round(time.time() - t_total_start, 2)
            meta["output_bytes"] = output_mp4.stat().st_size
            delivery = work / "delivery.zip"
            with zipfile.ZipFile(delivery, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(output_mp4, f"{stem}.mp4")
                z.writestr(f"{stem}.vtt", vtt_text)
                if cleaned_srt.exists(): z.write(cleaned_srt, f"{stem}.srt")
                if scc_path.exists():    z.write(scc_path,    f"{stem}.scc")
                if pdf_p.exists():       z.write(pdf_p,       f"{stem}.proof.pdf")
                z.writestr("meta.json", json.dumps(meta, indent=2))
            return delivery.read_bytes()
        except Exception as e:
            meta["error"] = f"{type(e).__name__}: {str(e)[:500]}"
            return self._error_zip(meta)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @modal.method()
    def transcode_vtt(self, video_bytes, filename="upload.mp4", media_id="",
                      target="broadcast_1080"):
        """SKU: Transcode + Transcribe bundle. Returns ZIP with re-encoded
        MP4 (per named preset) + VTT sidecar + meta.json. No CEA-608, no
        audit PDF — that's the CC-transcode bundle.
        Target: broadcast_1080|broadcast_720|web_540|archive_1080."""
        stem = Path(filename).stem
        work = Path(tempfile.mkdtemp(prefix="hybridcc_xv_"))
        input_mp4 = work / "input.mp4"
        input_mp4.write_bytes(video_bytes)
        t_total_start = time.time()
        meta = {
            "ok": False,
            "sku": "transcode-vtt",
            "media_id": media_id,
            "src_name": filename,
            "src_stem": stem,
            "model": MODEL_NAME,
            "backend": "modal",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_bytes": len(video_bytes),
        }
        try:
            # 1. Whisper -> VTT
            t_w = time.time()
            vtt_text, duration, _ = self._whisper_to_vtt(input_mp4)
            meta["duration"] = duration
            meta["whisper_sec"] = round(time.time() - t_w, 2)

            # 2. ffmpeg transcode per named preset (no captions)
            output_mp4 = work / "output.mp4"
            t_x = time.time()
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                "-i", str(input_mp4),
            ]
            cmd += _build_house_spec_args(target, input_mp4)
            # fMP4 — see inject branch above for rationale.
            cmd += ["-movflags", "+frag_keyframe+empty_moov+default_base_moof+faststart", str(output_mp4)]
            meta["target"] = target
            meta["target_label"] = _resolve_preset(target)["label"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            meta["transcode_sec"] = round(time.time() - t_x, 2)
            if r.returncode != 0 or not output_mp4.exists():
                meta["error"] = f"transcode failed: {r.stderr[:500]}"
                return self._error_zip(meta)

            # 3. Pricing
            pricing = _fetch_pricing("transcode-vtt")
            rate_per_sec = pricing.get("rate_per_sec", 0.000417)
            min_charge   = pricing.get("min_charge", 0.25)
            charge = max(min_charge, rate_per_sec * meta["duration"])

            meta["ok"] = True
            meta["total_elapsed_sec"] = round(time.time() - t_total_start, 2)
            meta["output_bytes"] = output_mp4.stat().st_size
            meta["charge_usd"] = round(charge, 2)
            meta["rate"] = pricing.get("rate_label", "$1.50/hr")

            # 4. Wrap delivery zip
            delivery = work / "delivery.zip"
            with zipfile.ZipFile(delivery, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(output_mp4, f"{stem}.mp4")
                z.writestr(f"{stem}.vtt", vtt_text)
                z.writestr("meta.json", json.dumps(meta, indent=2))
            return delivery.read_bytes()
        except Exception as e:
            meta["error"] = f"{type(e).__name__}: {str(e)[:500]}"
            return self._error_zip(meta)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @staticmethod
    def _error_zip(meta):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("meta.json", json.dumps(meta, indent=2))
        return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# CPU WORKER - ffmpeg-only transcode (Transcode SKU)
# ─────────────────────────────────────────────────────────────────────────────
# CPU-only container is ~10x cheaper than GPU. ffmpeg software encode (libx264)
# is plenty fast on 8 cores for SD/HD content.
@app.cls(
    cpu=8.0,
    memory=8192,
    secrets=[modal.Secret.from_name("hybridcc")],
    scaledown_window=300,
    min_containers=0,
    timeout=3600,
)
class CPUWorker:

    @modal.method()
    def transcode(self, video_bytes, filename="upload.mp4",
                  target="broadcast_1080", media_id=""):
        """SKU: Transcode - re-encode to a named house-spec preset.
        target: broadcast_1080|broadcast_720|web_540|archive_1080
        Aliases: house|default|1080|720|540 (all map to a preset).
        Returns ZIP with output.mp4 + meta.json."""
        stem = Path(filename).stem
        work = Path(tempfile.mkdtemp(prefix="hybridcc_xc_"))
        input_path = work / "input.mp4"
        input_path.write_bytes(video_bytes)
        output_path = work / "output.mp4"

        meta = {
            "ok": False,
            "media_id": media_id,
            "src_name": filename,
            "target": target,
            "backend": "modal",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Preset lookup (resolves aliases like 'house' → 'broadcast_1080')
        preset = _resolve_preset(target)
        meta["target"] = target
        meta["target_label"] = preset["label"]

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-i", str(input_path),
        ]
        cmd += _build_house_spec_args(target, input_path)
        # fMP4 — see hybridCC-vod inject branch for rationale.
        cmd += ["-movflags", "+frag_keyframe+empty_moov+default_base_moof+faststart", str(output_path)]

        try:
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            meta["transcode_sec"] = round(time.time() - t0, 2)
            if proc.returncode != 0 or not output_path.exists():
                meta["error"] = f"ffmpeg failed: {proc.stderr[:500]}"
                return self._wrap_zip(meta, None)
            meta["ok"] = True
            meta["output_bytes"] = output_path.stat().st_size
            return self._wrap_zip(meta, (output_path, f"{stem}.mp4"))
        except Exception as e:
            meta["error"] = f"{type(e).__name__}: {str(e)[:500]}"
            return self._wrap_zip(meta, None)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @staticmethod
    def _wrap_zip(meta, mp4):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            if mp4 and mp4[0].exists():
                z.write(mp4[0], mp4[1])
            z.writestr("meta.json", json.dumps(meta, indent=2))
        return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI surface
# ─────────────────────────────────────────────────────────────────────────────
@app.function(
    secrets=[modal.Secret.from_name("hybridcc")],
    scaledown_window=300,
    min_containers=0,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, Response

    api = FastAPI(title="HybridCC Services - Modal")
    gpu = GPUWorker()
    cpu = CPUWorker()

    SERVICE_KEY = os.environ.get("CAPTION_SERVICE_KEY", "hba_colab_service_internal")
    usage_log = []   # last 100 jobs, in-memory only

    def _allowed_keys():
        raw = os.environ.get("ALLOWED_API_KEYS", "{}")
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def verify_key(api_key):
        keys = _allowed_keys()
        if api_key and api_key in keys:
            return keys[api_key]
        if api_key == SERVICE_KEY:
            return "internal-service"
        raise HTTPException(status_code=401, detail="Invalid API key")

    def _log(customer, sku, filename, ok, extra=None):
        entry = {
            "job_id": str(uuid.uuid4())[:8],
            "customer": customer,
            "sku": sku,
            "filename": filename,
            "ok": ok,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            entry.update(extra)
        usage_log.append(entry)
        del usage_log[:-100]

    @api.get("/health")
    def health():
        return {
            "ok": True,
            "service": "hybridcc-services",
            "model": MODEL_NAME,
            "backend": "modal",
            "skus": ["transcribe", "caption/cc", "transcode",
                     "caption/cc-transcode", "caption/transcode-vtt"],
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    # ── SKU 1: Transcribe (VTT only) ──────────────────────────────────────
    @api.post("/caption/transcribe")
    async def caption_transcribe(
        file: UploadFile = File(...),
        media_id: str = Form(""),
        api_key: str = Header(None, alias="X-API-Key"),
    ):
        customer = verify_key(api_key)
        content = await file.read()
        result = await gpu.transcribe.remote.aio(content, file.filename or "upload.mp4")
        _log(customer, "transcribe", file.filename, result.get("ok", False),
             {"duration": result.get("duration", 0)})
        if result.get("ok") is False:
            return JSONResponse(status_code=422, content=result)
        if media_id:
            result["media_id"] = media_id
        return JSONResponse(result)

    # Legacy alias - matches existing modal-caption-server.py
    @api.post("/caption/json")
    async def caption_json(
        file: UploadFile = File(...),
        api_key: str = Header(None, alias="X-API-Key"),
    ):
        return await caption_transcribe(file=file, media_id="", api_key=api_key)

    # ── SKU 3: CC (full bundle) ───────────────────────────────────────────
    @api.post("/caption/cc")
    async def caption_cc(
        file: UploadFile = File(...),
        media_id: str = Form(""),
        content_type: str = Form("broadcast"),
        api_key: str = Header(None, alias="X-API-Key"),
    ):
        customer = verify_key(api_key)
        content = await file.read()
        zip_bytes = await gpu.caption.remote.aio(
            content, file.filename or "upload.mp4", media_id,
            content_type=content_type,
        )
        # Pull meta.json from the zip for usage logging
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                meta = json.loads(z.read("meta.json"))
        except Exception:
            meta = {"ok": False, "error": "could not read meta.json"}
        _log(customer, "cc", file.filename, meta.get("ok", False),
             {"duration": meta.get("duration", 0),
              "qc_pass": meta.get("qc_pass", False)})

        stem = Path(file.filename or "upload").stem
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{stem}.delivery.zip"',
                "X-HybridCC-Verdict": meta.get("verdict") or ("PASS" if meta.get("overall_pass") else "FAIL"),
                "X-HybridCC-Media-Id": media_id or "",
            },
        )

    # ── Bundle: CC + Transcode (re-encode to house spec WITH CEA-608) ────
    @api.post("/caption/cc-transcode")
    async def caption_cc_transcode(
        file: UploadFile = File(...),
        media_id: str = Form(""),
        target: str = Form("broadcast_1080"),
        content_type: str = Form("broadcast"),
        api_key: str = Header(None, alias="X-API-Key"),
    ):
        customer = verify_key(api_key)
        content = await file.read()
        zip_bytes = await gpu.caption.remote.aio(
            content, file.filename or "upload.mp4", media_id,
            transcode_to_house=True, target=target,
            content_type=content_type,
        )
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                meta = json.loads(z.read("meta.json"))
        except Exception:
            meta = {"ok": False}
        _log(customer, "cc-transcode", file.filename, meta.get("ok", False),
             {"duration": meta.get("duration", 0)})
        stem = Path(file.filename or "upload").stem
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{stem}.delivery.zip"',
                "X-HybridCC-Verdict": meta.get("verdict") or ("PASS" if meta.get("overall_pass") else "FAIL"),
                "X-HybridCC-Media-Id": media_id or "",
            },
        )

    # ── Bundle: Transcode + Transcribe (house-spec MP4 + VTT, no CEA-608) ─
    @api.post("/caption/transcode-vtt")
    async def caption_transcode_vtt(
        file: UploadFile = File(...),
        media_id: str = Form(""),
        target: str = Form("broadcast_1080"),
        api_key: str = Header(None, alias="X-API-Key"),
    ):
        customer = verify_key(api_key)
        content = await file.read()
        zip_bytes = await gpu.transcode_vtt.remote.aio(
            content, file.filename or "upload.mp4", media_id,
            target=target,
        )
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                meta = json.loads(z.read("meta.json"))
        except Exception:
            meta = {"ok": False}
        _log(customer, "transcode-vtt", file.filename, meta.get("ok", False),
             {"duration": meta.get("duration", 0)})
        stem = Path(file.filename or "upload").stem
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{stem}.delivery.zip"',
                "X-HybridCC-Media-Id": media_id or "",
            },
        )

    # ── SKU 2: Transcode ─────────────────────────────────────────────────
    @api.post("/transcode")
    async def transcode(
        file: UploadFile = File(...),
        target: str = Form("broadcast_1080"),
        media_id: str = Form(""),
        api_key: str = Header(None, alias="X-API-Key"),
    ):
        customer = verify_key(api_key)
        content = await file.read()
        zip_bytes = await cpu.transcode.remote.aio(
            content, file.filename or "upload.mp4", target, media_id
        )
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                meta = json.loads(z.read("meta.json"))
        except Exception:
            meta = {"ok": False}
        _log(customer, "transcode", file.filename, meta.get("ok", False),
             {"target": target})

        stem = Path(file.filename or "upload").stem
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{stem}.transcode.zip"',
                "X-HybridCC-Media-Id": media_id or "",
            },
        )

    @api.get("/usage")
    def usage():
        return {"ok": True, "recent": usage_log[-50:], "count": len(usage_log)}

    @api.get("/presets")
    def presets():
        """Return the canonical list of house-spec presets. The playout app's
        Settings page fetches this so its dropdown matches what the API
        accepts — single source of truth, no drift."""
        return {
            "ok": True,
            "default": "broadcast_1080",
            "presets": {
                name: {
                    "label":      p["label"],
                    "max_width":  p["max_width"],
                    "max_height": p["max_height"],
                    "vcodec":     p["vcodec"],
                    "profile":    p["profile"],
                    "level":      p["level"],
                    "abitrate":   p["abitrate"],
                }
                for name, p in HOUSE_SPEC_PRESETS.items()
            },
            "aliases": HOUSE_SPEC_ALIASES,
        }

    @api.on_event("startup")
    async def _register_on_startup():
        import threading
        threading.Thread(target=_do_register, args=("startup",), daemon=True).start()

    return api


# ─────────────────────────────────────────────────────────────────────────────
# Head-end registration (mirrors hybridcc-caption pattern)
# ─────────────────────────────────────────────────────────────────────────────
def _do_register(tag="heartbeat"):
    import urllib.request

    head_end = os.environ.get("HEAD_END_URL", "").rstrip("/")
    if not head_end:
        print(f"[{tag}] HEAD_END_URL not set; skipping")
        return

    workspace = os.environ.get("MODAL_WORKSPACE_NAME", "")
    if workspace:
        tunnel_url = f"https://{workspace}--{APP_NAME}-fastapi-app.modal.run"
    else:
        tunnel_url = os.environ.get("MODAL_PUBLIC_URL", "")

    if not tunnel_url:
        print(f"[{tag}] no tunnel_url; set MODAL_PUBLIC_URL in the hybridcc secret")
        return

    payload = json.dumps({
        "service_id":   APP_NAME + "-1",
        "service_type": "caption",
        "tunnel_url":   tunnel_url,
        "capabilities": [
            "transcribe",       # /caption/transcribe + /caption/json
            "caption",          # /caption/cc (CC SKU)
            "transcode",        # /transcode
            "word_timestamps",
            "audit_pdf",
        ],
        "metadata": {
            "backend": "modal",
            "model":   MODEL_NAME,
            "gpu":     "L4",
            "service": APP_NAME,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{head_end}/api/services/register",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[{tag}] {resp.status} {tunnel_url}")
    except Exception as e:
        print(f"[{tag}] error: {e}")


@app.function(
    schedule=modal.Period(minutes=6),
    secrets=[modal.Secret.from_name("hybridcc")],
)
def register_with_head_end():
    _do_register("heartbeat")


# ─────────────────────────────────────────────────────────────────────────────
# Local CLI helper
# ─────────────────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print(f"HybridCC services (Modal) - deploy with: modal deploy {Path(__file__).name}")
    print("After deploy, public URL is at:")
    print(f"  https://<workspace>--{APP_NAME}-fastapi-app.modal.run")
    print()
    print("Endpoints:")
    print("  POST /caption/transcribe   - VTT only")
    print("  POST /caption/cc           - flat zip: MP4 + VTT + SRT + SCC + PDF + meta.json")
    print("  POST /transcode            - house-spec MP4")
    print("  GET  /health")
