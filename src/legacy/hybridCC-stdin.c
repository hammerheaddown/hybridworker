/*
 * hybridCC-stdin.exe -- CEA-608 caption injector (stdin/stdout FLV pipe)
 *
 * Reads FLV from stdin, injects CEA-608 SEI NAL units into H264 frames
 * using the latest text from a VTT/SRT file (hot-reloaded on change).
 * Writes captioned FLV to stdout.
 *
 * Usage: hybridCC-stdin captions.vtt < input.flv > output.flv
 *   Or piped: ffmpeg ... -f flv pipe:1 | hybridCC-stdin captions.vtt | ffmpeg -f flv -i pipe:0 ...
 *
 * The VTT file is re-read when its mtime changes. Only the LAST cue's
 * text is used -- injected into every video keyframe. No timestamp matching.
 * This is designed for live captioning where the VTT is continuously
 * updated by cc-segmenter.js.
 *
 * RECOVERED 2026-05-06 from chat session 5cfcd6ef line 3296 (Write tool call,
 * 2026-04-17). Original source was lost when Temp\libcaption\examples\ cleared.
 * Already Linux-portable: Windows _setmode patch is under #ifdef _WIN32.
 */

#include "caption/caption.h"
#include "flv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define stat _stat
#endif

#define MAX_VTT_SIZE (256 * 1024)

/* Current caption text -- updated from VTT file */
static char g_text[4096] = {0};
static char g_path[1024] = {0};
static time_t g_mtime = 0;
static int g_injected = 0;

/* Extract last cue text from VTT or SRT file */
static void reload_captions(void)
{
    struct stat st;
    if (stat(g_path, &st) != 0) return;
    if (st.st_mtime == g_mtime) return;
    g_mtime = st.st_mtime;

    FILE* f = fopen(g_path, "rb");
    if (!f) return;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz >= MAX_VTT_SIZE) { fclose(f); return; }

    char* buf = (char*)malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = 0;
    fclose(f);

    /* Find the last text block -- skip headers and timestamp lines */
    char* last_text = NULL;
    char* line = buf;
    int prev_was_timestamp = 0;

    while (*line) {
        char* eol = strstr(line, "\n");
        if (!eol) eol = line + strlen(line);

        int len = (int)(eol - line);
        /* Skip \r */
        if (len > 0 && line[len-1] == '\r') len--;

        if (len == 0) {
            prev_was_timestamp = 0;
        } else if (strstr(line, "-->") && (eol - line) < 80) {
            prev_was_timestamp = 1;
        } else if (prev_was_timestamp || (len > 0 && strncmp(line, "WEBVTT", 6) != 0)) {
            /* This is caption text */
            if (prev_was_timestamp) {
                last_text = line;
            }
            prev_was_timestamp = 0;
        }

        if (*eol == '\n') eol++;
        line = eol;
    }

    if (last_text) {
        /* Copy until next blank line or end */
        char* end = strstr(last_text, "\r\n\r\n");
        if (!end) end = strstr(last_text, "\n\n");
        if (!end) end = last_text + strlen(last_text);
        int tlen = (int)(end - last_text);
        if (tlen > (int)sizeof(g_text) - 1) tlen = sizeof(g_text) - 1;
        memcpy(g_text, last_text, tlen);
        g_text[tlen] = 0;
        /* Trim trailing whitespace */
        while (tlen > 0 && (g_text[tlen-1] == '\r' || g_text[tlen-1] == '\n' || g_text[tlen-1] == ' '))
            g_text[--tlen] = 0;
        fprintf(stderr, "[CC] VTT reloaded: %s\n", g_text);
        g_injected = 0;
    }

    free(buf);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "hybridCC-stdin -- CEA-608 live caption injector (stdin/stdout FLV)\n\n");
        fprintf(stderr, "Usage: %s captions.vtt < input.flv > output.flv\n", argv[0]);
        fprintf(stderr, "  Or: ffmpeg ... -f flv pipe:1 | %s captions.vtt | ffmpeg -f flv -i pipe:0 ...\n\n", argv[0]);
        fprintf(stderr, "VTT file hot-reloaded on change. Last cue injected as CEA-608 SEI.\n");
        return 1;
    }

#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    strncpy(g_path, argv[1], sizeof(g_path) - 1);
    reload_captions();

    FILE* in = stdin;
    FILE* out = stdout;

    flvtag_t tag;
    flvtag_init(&tag);
    int has_audio, has_video;
    int frame_count = 0;

    if (!flv_read_header(in, &has_audio, &has_video)) {
        fprintf(stderr, "[CC] Not a valid FLV on stdin\n");
        return 1;
    }

    flv_write_header(out, has_audio, has_video);
    fprintf(stderr, "[CC] Streaming (audio=%d video=%d)...\n", has_audio, has_video);

    while (flv_read_tag(in, &tag)) {
        /* Check for VTT changes every video frame */
        if (flvtag_avcpackettype_nalu == flvtag_avcpackettype(&tag)) {
            reload_captions();
            frame_count++;

            if (g_text[0] && !g_injected) {
                flvtag_addcaption_text(&tag, (const utf8_char_t*)g_text);
                g_injected = 1; /* Only inject once per VTT reload -- clear injects on next reload */
            } else if (g_text[0]) {
                /* Keep injecting the same text to maintain display */
                flvtag_addcaption_text(&tag, (const utf8_char_t*)g_text);
            }
        }

        flv_write_tag(out, &tag);
        fflush(out); /* Flush for pipe -- don't buffer */
    }

    fprintf(stderr, "[CC] Done, %d frames processed\n", frame_count);
    flvtag_free(&tag);
    return 0;
}
