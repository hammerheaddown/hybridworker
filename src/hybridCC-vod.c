/*
 * hybridCC-vod -- CEA-608 caption injector for VOD (timestamp-matched)
 *
 * Reads FLV from stdin, parses ALL cues from a VTT file at startup,
 * and injects each cue's text into H264 frames whose PTS falls within
 * the cue's [start, end) range. Writes captioned FLV to stdout.
 *
 * Usage: hybridCC-vod captions.vtt < input.flv > output.flv
 *
 *   ffmpeg -i input.mp4 -c:v copy -c:a copy -f flv pipe:1 | \
 *     hybridCC-vod captions.vtt | \
 *     ffmpeg -f flv -i pipe:0 -c:v copy -c:a copy -a53cc 1 output.mp4
 *
 * Forked from hybridCC-stdin.c on 2026-05-06. Differences vs the live tool:
 *   - VTT loaded ONCE at startup (no hot-reload, no mtime polling)
 *   - All cues parsed into a sorted array, not just the last cue
 *   - Each video frame's PTS is matched against cue ranges
 *   - Forward-cursor lookup (cues are time-sorted, frames stream in order)
 */

#include "caption/caption.h"
#include "caption/mpeg.h"   /* sei_t, sei_init, sei_free, sei_from_caption_frame */
#include "flv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#define MAX_VTT_SIZE  (4 * 1024 * 1024)   /* 4 MB hard cap */
#define MAX_CUES       20000               /* ~5h at 1 cue/sec */
#define MAX_CUE_TEXT   1024                /* single-cue text limit */

/* libcaption's caption_frame_from_text writes line N into row N starting at
 * row 0, which puts captions at the TOP — wrong for broadcast (standard is
 * bottom rows 14-15). Leading-newline padding fails because the inner loop
 * skips whitespace at line-start and only bumps the row counter for non-blank
 * lines.
 *
 * Instead, mirror flvtag_addcaption_text but call caption_frame_write_char
 * directly to drop the last 1-2 lines of input into rows 13-14, then take the
 * same sei_init → sei_from_caption_frame → flvtag_addsei → sei_free path the
 * library uses internally. */
#define CEA608_BOTTOM_ROW   14   /* 0-indexed; CEA-608 has 15 rows total */
#define CEA608_COL_WIDTH    32

static int flvtag_addcaption_text_bottom(flvtag_t* tag, const utf8_char_t* text)
{
    if (!text || !*text) return flvtag_addcaption_text(tag, text);

    /* Find the last 1-2 non-blank lines of input. We display 2 lines max,
     * which matches CEA-608 pop-on convention and how downstream decoders
     * (VLC, Shaka, hardware STBs) expect roll-up content to look. */
    const utf8_char_t* line_prev = NULL;
    const utf8_char_t* line_last = NULL;
    const utf8_char_t* p = text;
    while (*p) {
        while (*p == '\n' || *p == '\r' || *p == ' ' || *p == '\t') p++;
        if (!*p) break;
        line_prev = line_last;
        line_last = p;
        while (*p && *p != '\n' && *p != '\r') p++;
    }
    if (!line_last) return flvtag_addcaption_text(tag, text);  /* nothing to write */

    const utf8_char_t* sources[2];
    int rows[2];
    int n;
    if (line_prev) {
        sources[0] = line_prev;        rows[0] = CEA608_BOTTOM_ROW - 1;
        sources[1] = line_last;        rows[1] = CEA608_BOTTOM_ROW;
        n = 2;
    } else {
        sources[0] = line_last;        rows[0] = CEA608_BOTTOM_ROW;
        n = 1;
    }

    sei_t sei;
    sei_init(&sei, (double)flvtag_pts(tag) / 1000.0);

    caption_frame_t frame;
    caption_frame_init(&frame);
    /* caption_frame_from_text does this before writing — caption_frame_write_char
     * mutates whichever buffer ->write points at. */
    frame.write = &frame.back;

    for (int i = 0; i < n; i++) {
        const utf8_char_t* lp = sources[i];
        int col = 0;
        while (*lp && *lp != '\n' && *lp != '\r' && col < CEA608_COL_WIDTH) {
            size_t cl = utf8_char_length(lp);
            if (cl == 0) break;
            caption_frame_write_char(&frame, rows[i], col, eia608_style_white, 0, lp);
            lp += cl;
            col++;
        }
    }

    caption_frame_end(&frame);
    sei_from_caption_frame(&sei, &frame);
    int ret = flvtag_addsei(tag, &sei);
    sei_free(&sei);
    return ret;
}

typedef struct {
    double start;       /* seconds */
    double end;         /* seconds */
    char   text[MAX_CUE_TEXT];
} vtt_cue_t;

static vtt_cue_t g_cues[MAX_CUES];
static int g_cue_count = 0;
static int g_cursor    = 0;   /* hint for forward-only lookup */

/* Parse "HH:MM:SS.mmm" or "MM:SS.mmm" → seconds. -1 on failure. */
static double parse_ts(const char* s, int len)
{
    char buf[32];
    if (len <= 0 || len >= (int)sizeof(buf)) return -1.0;
    memcpy(buf, s, len);
    buf[len] = 0;

    int h = 0, m = 0, sec = 0, ms = 0;
    /* Try HH:MM:SS.mmm first */
    if (sscanf(buf, "%d:%d:%d.%d", &h, &m, &sec, &ms) == 4) {
        return h * 3600.0 + m * 60.0 + sec + ms / 1000.0;
    }
    /* Fall back to MM:SS.mmm (VTT allows both) */
    if (sscanf(buf, "%d:%d.%d", &m, &sec, &ms) == 3) {
        return m * 60.0 + sec + ms / 1000.0;
    }
    return -1.0;
}

/* Load + parse every cue from a VTT file. Returns cue count. */
static int load_vtt_cues(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[CC] Cannot open VTT: %s\n", path);
        return 0;
    }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz >= MAX_VTT_SIZE) {
        fprintf(stderr, "[CC] VTT empty or too large: %ld bytes\n", sz);
        fclose(f);
        return 0;
    }

    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return 0; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return 0;
    }
    buf[sz] = 0;
    fclose(f);

    char* p = buf;
    while (*p && g_cue_count < MAX_CUES) {
        /* Find next timestamp-arrow line */
        char* arrow = strstr(p, "-->");
        if (!arrow) break;

        /* Walk back to start of this line */
        char* line_start = arrow;
        while (line_start > buf && line_start[-1] != '\n') line_start--;

        /* Parse start timestamp (before "-->"), trim trailing whitespace */
        char* s_end = arrow;
        while (s_end > line_start && (s_end[-1] == ' ' || s_end[-1] == '\t')) s_end--;
        double start = parse_ts(line_start, (int)(s_end - line_start));

        /* Parse end timestamp (after "-->"), skipping whitespace */
        char* e_start = arrow + 3;
        while (*e_start == ' ' || *e_start == '\t') e_start++;
        char* e_end = e_start;
        while (*e_end && *e_end != ' ' && *e_end != '\r' && *e_end != '\n') e_end++;
        double end = parse_ts(e_start, (int)(e_end - e_start));

        if (start < 0 || end <= start) {
            /* Bad line, advance past arrow and try again */
            p = arrow + 3;
            continue;
        }

        /* Text starts on next line */
        char* text_start = strchr(arrow, '\n');
        if (!text_start) break;
        text_start++;

        /* Text ends at next blank line (preferred) OR at the start of the
         * next timestamp line (fallback for malformed VTT without blank-line
         * separators). Whichever comes first. */
        char* text_end = strstr(text_start, "\r\n\r\n");
        if (!text_end) text_end = strstr(text_start, "\n\n");

        char* next_arrow = strstr(text_start, "-->");
        if (next_arrow) {
            /* Walk back to the start of that timestamp's line */
            char* next_arrow_line = next_arrow;
            while (next_arrow_line > text_start && next_arrow_line[-1] != '\n')
                next_arrow_line--;
            if (!text_end || next_arrow_line < text_end)
                text_end = next_arrow_line;
        }

        if (!text_end) text_end = buf + sz;

        /* Copy text, collapsing newlines/CR into spaces, strip dup whitespace */
        vtt_cue_t* cue = &g_cues[g_cue_count];
        int outlen = 0;
        int prev_space = 1;   /* suppress leading whitespace */
        int span = (int)(text_end - text_start);
        for (int i = 0; i < span; i++) {
            char c = text_start[i];
            if (c == '\r' || c == '\n' || c == '\t') c = ' ';
            if (c == ' ' && prev_space) continue;
            if (outlen >= MAX_CUE_TEXT - 1) break;
            cue->text[outlen++] = c;
            prev_space = (c == ' ');
        }
        /* Trim trailing whitespace */
        while (outlen > 0 && cue->text[outlen-1] == ' ') outlen--;
        cue->text[outlen] = 0;

        if (outlen > 0) {
            cue->start = start;
            cue->end   = end;
            g_cue_count++;
        }

        p = text_end;
    }

    free(buf);
    fprintf(stderr, "[CC] Parsed %d cues from %s\n", g_cue_count, path);

    /* Sanity: VTT should already be sorted, but verify */
    for (int i = 1; i < g_cue_count; i++) {
        if (g_cues[i].start < g_cues[i-1].start) {
            fprintf(stderr, "[CC] WARN: cue %d out of order (%.3f < %.3f) -- VTT not sorted\n",
                    i, g_cues[i].start, g_cues[i-1].start);
            break;
        }
    }
    return g_cue_count;
}

/* Find the cue whose [start, end) contains pts_sec.
 * Uses g_cursor hint -- frames stream forward, so this is O(1) amortized. */
static const char* find_cue_at(double pts_sec)
{
    /* Advance cursor past cues that have ended */
    while (g_cursor < g_cue_count && g_cues[g_cursor].end <= pts_sec) {
        g_cursor++;
    }
    if (g_cursor >= g_cue_count) return NULL;
    if (pts_sec >= g_cues[g_cursor].start) return g_cues[g_cursor].text;
    return NULL;   /* in a gap before next cue */
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "hybridCC-vod -- CEA-608 VOD caption injector (FLV stdin/stdout)\n\n");
        fprintf(stderr, "Usage: %s captions.vtt < input.flv > output.flv\n\n", argv[0]);
        fprintf(stderr, "  ffmpeg -i in.mp4 -c copy -f flv pipe:1 | \\\n");
        fprintf(stderr, "    %s in.vtt | \\\n", argv[0]);
        fprintf(stderr, "    ffmpeg -f flv -i pipe:0 -c copy -a53cc 1 out.mp4\n");
        return 1;
    }

#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    load_vtt_cues(argv[1]);   /* OK if 0 cues -- pass-through */

    flvtag_t tag;
    flvtag_init(&tag);

    int has_audio = 0, has_video = 0;
    if (!flv_read_header(stdin, &has_audio, &has_video)) {
        fprintf(stderr, "[CC] Not a valid FLV on stdin\n");
        return 1;
    }
    flv_write_header(stdout, has_audio, has_video);
    fprintf(stderr, "[CC] Streaming (audio=%d video=%d)...\n", has_audio, has_video);

    long video_frames = 0, injected = 0;
    int first_nal = 1;   /* marker for first video NAL — see below */

    while (flv_read_tag(stdin, &tag)) {
        if (flvtag_avcpackettype_nalu == flvtag_avcpackettype(&tag)) {
            video_frames++;
            uint32_t pts_ms = flvtag_pts(&tag);
            const char* text = find_cue_at(pts_ms / 1000.0);

            /* Marker SEI on the first video NAL. Tells VLC's H.264 decoder
             * "this stream has CEA-608" before the first real cue appears
             * (which the SCC lead-in shift puts at 2.5s). Without this,
             * VLC's pre-play scan misses our SEI and the user has to play
             * 1-2s before "Closed Captions 1" shows in the Subtitle menu.
             * Empty/space caption is invisible — just registers the track. */
            if (first_nal) {
                flvtag_addcaption_text(&tag, (const utf8_char_t*)" ");
                first_nal = 0;
            }

            /* Inject every frame within a cue's window. CEA-608 is stateful;
             * VLC and other in-band decoders need continuous caption codes
             * to keep the display state alive. State-change-only injection
             * (pretty in extraction dumps) breaks VLC's CC renderer. */
            if (text && *text) {
                flvtag_addcaption_text_bottom(&tag, (const utf8_char_t*)text);
                injected++;
            }
        }
        flv_write_tag(stdout, &tag);
    }

    fprintf(stderr, "[CC] Done: %ld video frames, %ld with captions (%ld%%)\n",
            video_frames, injected,
            video_frames > 0 ? (injected * 100 / video_frames) : 0L);

    flvtag_free(&tag);
    return 0;
}
