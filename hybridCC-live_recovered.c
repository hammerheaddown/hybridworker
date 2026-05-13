/*
 * hybridCC-live.exe - Zero-latency CEA-608 caption injector
 *
 * Acts as RTSP proxy: reads from source, injects CC into video, forwards all
 * packets (video+audio) to a listening FFmpeg instance via TCP.
 *
 * Usage: hybridCC-live <rtsp_url> <caption.srt> <listen_port>
 *
 * FFmpeg connects to the listen port and receives interleaved RTP:
 *   hybridCC-live rtsp://localhost:8554/canvas-hls captions.srt 19800
 *   ffmpeg -rtsp_transport tcp -i rtsp://127.0.0.1:19800/live -c copy -f hls ...
 *
 * Actually simpler: output a raw RTP-like stream that FFmpeg reads as FLV.
 * We write FLV with video (CC injected) and pass audio as FLV audio tags
 * using the "Opus in FLV" extension that FFmpeg supports.
 */

#include "caption/caption.h"
#include "caption/srt.h"
#include "flv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#include <io.h>
#include <fcntl.h>
#define stat _stat
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#define SOCKET int
#define INVALID_SOCKET -1
#define closesocket close
#endif

#define MAX_SRT (1*1024*1024)
#define MAX_RTP 65536
#define MAX_NAL (2*1024*1024)

static volatile int g_run = 1;
static void on_sig(int s) { (void)s; g_run = 0; }

/* ---- SRT ---- */
static srt_t* g_srt = NULL;
static time_t g_mt = 0;
static char g_sp[1024];
static srt_cue_t* g_cue = NULL;
static double g_clr = -1;

static srt_t* srt_load(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0 || n >= MAX_SRT) { fclose(f); return NULL; }
    char* b = (char*)malloc(n + 1);
    fread(b, 1, n, f);
    b[n] = 0;
    fclose(f);
    srt_t* s = srt_parse((utf8_char_t*)b, n);
    free(b);
    return s;
}

static void srt_chk(double t) {
    struct stat st;
    if (stat(g_sp, &st) || st.st_mtime == g_mt) return;
    srt_t* s = srt_load(g_sp);
    if (!s) return;
    if (g_srt) srt_free(g_srt);
    g_srt = s;
    g_mt = st.st_mtime;
    g_cue = g_srt->cue_head;
    g_clr = -1;
    while (g_cue && (g_cue->timestamp + g_cue->duration) < t)
        g_cue = g_cue->next;
    fprintf(stderr, "[CC] SRT reloaded\n");
    fflush(stderr);
}

/* ---- RTSP ---- */
typedef struct {
    SOCKET s;
    int cseq;
    char sess[256];
    char url[1024];
    char host[256];
    int port;
} rtsp_t;

static void rtsp_parse(rtsp_t* r, const char* u) {
    strncpy(r->url, u, sizeof(r->url) - 1);
    r->port = 8554;
    const char* p = strstr(u, "://");
    if (!p) return;
    p += 3;
    const char* c = strchr(p, ':');
    const char* sl = strchr(p, '/');
    if (c && (!sl || c < sl)) {
        int h = (int)(c - p);
        memcpy(r->host, p, h);
        r->host[h] = 0;
        r->port = atoi(c + 1);
    } else if (sl) {
        int h = (int)(sl - p);
        memcpy(r->host, p, h);
        r->host[h] = 0;
    } else {
        strncpy(r->host, p, 255);
    }
}

static int rtsp_conn(rtsp_t* r) {
    struct addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    char ps[16];
    sprintf(ps, "%d", r->port);
    if (getaddrinfo(r->host, ps, &hints, &res)) return -1;
    r->s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    int rc = connect(r->s, res->ai_addr, (int)res->ai_addrlen);
    freeaddrinfo(res);
    return rc;
}

static int rtsp_cmd(rtsp_t* r, const char* m, const char* u, const char* x) {
    char b[2048];
    r->cseq++;
    int n = sprintf(b, "%s %s RTSP/1.0\r\nCSeq: %d\r\n", m, u, r->cseq);
    if (r->sess[0])
        n += sprintf(b + n, "Session: %s\r\n", r->sess);
    if (x)
        n += sprintf(b + n, "%s", x);
    n += sprintf(b + n, "\r\n");
    return send(r->s, b, n, 0);
}

static int rtsp_resp(rtsp_t* r, char* b, int sz) {
    int t = 0;
    while (t < sz - 1) {
        int n = recv(r->s, b + t, 1, 0);
        if (n <= 0) break;
        t += n;
        if (t >= 4 && memcmp(b + t - 4, "\r\n\r\n", 4) == 0) break;
    }
    b[t] = 0;
    char* sp = strstr(b, "Session: ");
    if (sp) {
        sp += 9;
        int i = 0;
        while (sp[i] && sp[i] != '\r' && sp[i] != ';' && i < 255) {
            r->sess[i] = sp[i];
            i++;
        }
        r->sess[i] = 0;
    }
    char* cl = strstr(b, "Content-Length: ");
    if (cl) {
        int bl = atoi(cl + 16);
        if (bl > 0) {
            char* tmp = (char*)malloc(bl);
            int g = 0;
            while (g < bl) {
                int n = recv(r->s, tmp + g, bl - g, 0);
                if (n <= 0) break;
                g += n;
            }
            free(tmp);
        }
    }
    return t;
}

/* ---- RTP interleaved read ---- */
static int rtp_rd(SOCKET s, uint8_t* b, int bsz, int* ch) {
    uint8_t hdr[4];
    while (1) {
        int n = recv(s, (char*)hdr, 1, 0);
        if (n <= 0) return -1;
        if (hdr[0] == 0x24) break;
    }
    int n = recv(s, (char*)hdr + 1, 3, MSG_WAITALL);
    if (n != 3) return -1;
    *ch = hdr[1];
    int len = (hdr[2] << 8) | hdr[3];
    if (len > bsz) return -1;
    int got = 0;
    while (got < len) {
        n = recv(s, (char*)b + got, len - got, 0);
        if (n <= 0) return -1;
        got += n;
    }
    return len;
}

/* ---- AVC sequence header ---- */
static uint8_t g_sps[512]; static int g_sps_len = 0;
static uint8_t g_pps[128]; static int g_pps_len = 0;
static int g_seq_written = 0;

static void write_avc_seq_header(FILE* out, uint32_t dts) {
    if (g_sps_len == 0 || g_pps_len == 0 || g_seq_written) return;
    int cfg_len = 11 + g_sps_len + g_pps_len;
    int tag_size = 5 + cfg_len;
    int total = 11 + tag_size + 4;
    uint8_t* buf = (uint8_t*)calloc(1, total);
    buf[0] = 0x09;
    buf[1] = (tag_size >> 16) & 0xFF;
    buf[2] = (tag_size >> 8) & 0xFF;
    buf[3] = tag_size & 0xFF;
    buf[4] = (dts >> 16) & 0xFF;
    buf[5] = (dts >> 8) & 0xFF;
    buf[6] = dts & 0xFF;
    buf[7] = (dts >> 24) & 0xFF;
    int p = 11;
    buf[p++] = 0x17;
    buf[p++] = 0x00;
    buf[p++] = 0x00; buf[p++] = 0x00; buf[p++] = 0x00;
    buf[p++] = 0x01;
    buf[p++] = g_sps[1];
    buf[p++] = g_sps[2];
    buf[p++] = g_sps[3];
    buf[p++] = 0xFF;
    buf[p++] = 0xE1;
    buf[p++] = (g_sps_len >> 8) & 0xFF;
    buf[p++] = g_sps_len & 0xFF;
    memcpy(buf + p, g_sps, g_sps_len); p += g_sps_len;
    buf[p++] = 0x01;
    buf[p++] = (g_pps_len >> 8) & 0xFF;
    buf[p++] = g_pps_len & 0xFF;
    memcpy(buf + p, g_pps, g_pps_len); p += g_pps_len;
    int prev = 11 + tag_size;
    buf[p++] = (prev >> 24) & 0xFF;
    buf[p++] = (prev >> 16) & 0xFF;
    buf[p++] = (prev >> 8) & 0xFF;
    buf[p++] = prev & 0xFF;
    fwrite(buf, 1, p, out);
    fflush(out);
    free(buf);
    g_seq_written = 1;
    fprintf(stderr, "[CC] AVC seq header (SPS=%d PPS=%d)\n", g_sps_len, g_pps_len);
    fflush(stderr);
}

static void capture_sps_pps(uint8_t* pl, int psz) {
    int pos = 1;
    while (pos + 2 < psz) {
        int ns = (pl[pos] << 8) | pl[pos + 1];
        pos += 2;
        if (pos + ns > psz) break;
        uint8_t nt = pl[pos] & 0x1F;
        if (nt == 7 && ns <= (int)sizeof(g_sps)) {
            memcpy(g_sps, pl + pos, ns);
            g_sps_len = ns;
        } else if (nt == 8 && ns <= (int)sizeof(g_pps)) {
            memcpy(g_pps, pl + pos, ns);
            g_pps_len = ns;
        }
        pos += ns;
    }
}

/* ---- FLV audio tag writer for Opus ---- */
/* FLV extended audio: tag type 0x08, codec=13 (Opus) or use raw passthrough */
static void write_flv_audio_tag(FILE* out, uint8_t* rtp_buf, int rtp_len, uint32_t dts) {
    /* Extract RTP payload (skip 12-byte RTP header + CSRC + extension) */
    if (rtp_len < 12) return;
    int off = 12 + (rtp_buf[0] & 0x0F) * 4;
    if (rtp_buf[0] & 0x10) {
        if (off + 4 > rtp_len) return;
        off += 4 + (((int)rtp_buf[off + 2] << 8) | rtp_buf[off + 3]) * 4;
    }
    if (off >= rtp_len) return;

    uint8_t* payload = rtp_buf + off;
    int payload_len = rtp_len - off;
    if (payload_len <= 0) return;

    /* Write FLV audio tag with Opus codec ID */
    /* FLV enhanced audio: 0xAF for AAC-like header, but for Opus we use
       the "enhanced FLV" codec ID. FFmpeg's FLV demuxer reads Opus if
       we use soundformat=13 (Opus) in the audio tag header byte.
       Byte: (13 << 4) | 0x0E = 0xDE  (Opus, 48kHz, stereo, 16-bit) */
    int tag_payload = 1 + payload_len; /* 1 byte audio header + data */
    int tag_size = tag_payload;
    int total = 11 + tag_size + 4;
    uint8_t* buf = (uint8_t*)calloc(1, total);

    buf[0] = 0x08; /* audio tag */
    buf[1] = (tag_size >> 16) & 0xFF;
    buf[2] = (tag_size >> 8) & 0xFF;
    buf[3] = tag_size & 0xFF;
    buf[4] = (dts >> 16) & 0xFF;
    buf[5] = (dts >> 8) & 0xFF;
    buf[6] = dts & 0xFF;
    buf[7] = (dts >> 24) & 0xFF;
    /* stream ID = 0 */

    /* Audio data: soundformat=13(Opus), rate=3(44kHz placeholder), size=1(16bit), type=1(stereo) */
    buf[11] = 0xDF; /* (13<<4) | 0x0F = Opus, 44kHz, 16bit, stereo */
    memcpy(buf + 12, payload, payload_len);

    /* PreviousTagSize */
    int prev = 11 + tag_size;
    int p = 11 + tag_size;
    buf[p++] = (prev >> 24) & 0xFF;
    buf[p++] = (prev >> 16) & 0xFF;
    buf[p++] = (prev >> 8) & 0xFF;
    buf[p++] = prev & 0xFF;

    fwrite(buf, 1, total, out);
    free(buf);
}

/* ---- inject CC helper ---- */
static void try_inject(flvtag_t* tag, double t, unsigned long* cc) {
    if (g_cue && g_cue->timestamp <= t) {
        flvtag_addcaption_text(tag, srt_cue_data(g_cue));
        g_clr = g_cue->timestamp + g_cue->duration;
        fprintf(stderr, "[CC] T=%.1f: %.50s%s\n", t,
            srt_cue_data(g_cue),
            strlen(srt_cue_data(g_cue)) > 50 ? "..." : "");
        fflush(stderr);
        (*cc)++;
        g_cue = g_cue->next;
    } else if (g_clr >= 0 && g_clr <= t) {
        flvtag_addcaption_text(tag, NULL);
        g_clr = -1;
    }
}

/* ---- main ---- */
int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "hybridCC-live - CEA-608 live caption injector\n\n");
        fprintf(stderr, "Usage: %s rtsp://host:port/path captions.srt output.flv\n\n", argv[0]);
        fprintf(stderr, "Reads RTSP (video+audio), injects CEA-608, writes FLV with both.\n");
        fprintf(stderr, "SRT file hot-reloaded on change. Ctrl+C to stop.\n");
        return 1;
    }

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    strncpy(g_sp, argv[2], sizeof(g_sp) - 1);
    g_srt = srt_load(g_sp);
    if (g_srt) {
        struct stat st;
        if (!stat(g_sp, &st)) g_mt = st.st_mtime;
        g_cue = g_srt->cue_head;
        fprintf(stderr, "[CC] SRT: %s\n", g_sp);
    } else {
        fprintf(stderr, "[CC] no SRT: %s\n", g_sp);
    }
    fflush(stderr);

    rtsp_t r;
    memset(&r, 0, sizeof(r));
    rtsp_parse(&r, argv[1]);
    fprintf(stderr, "[CC] RTSP %s:%d\n", r.host, r.port);
    fflush(stderr);

    if (rtsp_conn(&r)) {
        fprintf(stderr, "[CC] connect failed\n");
        return 1;
    }
    fprintf(stderr, "[CC] connected\n");
    fflush(stderr);

    char resp[4096];
    int rl;

    /* DESCRIBE */
    rtsp_cmd(&r, "DESCRIBE", r.url, "Accept: application/sdp\r\n");
    rl = rtsp_resp(&r, resp, sizeof(resp));
    fprintf(stderr, "[CC] DESCRIBE %d\n", rl);
    fflush(stderr);

    /* SETUP audio (channels 2-3) */
    {
        char tu[1024];
        snprintf(tu, sizeof(tu), "%s/trackID=0", r.url);
        rtsp_cmd(&r, "SETUP", tu, "Transport: RTP/AVP/TCP;unicast;interleaved=2-3\r\n");
        rl = rtsp_resp(&r, resp, sizeof(resp));
        fprintf(stderr, "[CC] SETUP audio %d sess=%s\n", rl, r.sess);
        fflush(stderr);
    }

    /* SETUP video (channels 0-1) */
    {
        char tu[1024];
        snprintf(tu, sizeof(tu), "%s/trackID=1", r.url);
        rtsp_cmd(&r, "SETUP", tu, "Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n");
        rl = rtsp_resp(&r, resp, sizeof(resp));
        fprintf(stderr, "[CC] SETUP video %d sess=%s\n", rl, r.sess);
        fflush(stderr);
    }

    /* PLAY */
    rtsp_cmd(&r, "PLAY", r.url, "Range: npt=0.000-\r\n");
    rl = rtsp_resp(&r, resp, sizeof(resp));
    fprintf(stderr, "[CC] PLAY %d\n", rl);
    fflush(stderr);

    fprintf(stderr, "[CC] streaming...\n");
    fflush(stderr);

    /* FLV output with audio + video */
    FILE* out = flv_open_write(argv[3]);
    if (!out) {
        fprintf(stderr, "[CC] cant open %s\n", argv[3]);
        return 1;
    }
    flv_write_header(out, 1, 1); /* audio=1 video=1 */

    uint8_t rb[MAX_RTP];
    flvtag_t tag;
    flvtag_init(&tag);

    unsigned long fr = 0, cc = 0, pk = 0, audio_tags = 0;
    uint32_t ts0_video = 0, ts0_audio = 0;
    int ts0v_set = 0, ts0a_set = 0;

    uint8_t* nb = (uint8_t*)malloc(MAX_NAL);
    int nl = 0, ni = 0;

    while (g_run) {
        int ch;
        int len = rtp_rd(r.s, rb, sizeof(rb), &ch);
        if (len < 0) {
            fprintf(stderr, "[CC] rtp err pk=%lu\n", pk);
            fflush(stderr);
            break;
        }
        pk++;

        if (pk <= 10) {
            fprintf(stderr, "[CC] pkt#%lu ch=%d len=%d\n", pk, ch, len);
            fflush(stderr);
        }

        /* ---- AUDIO (channel 2) ---- */
        if (ch == 2 && len >= 12) {
            uint32_t rts = ((uint32_t)rb[4] << 24) | ((uint32_t)rb[5] << 16) |
                           ((uint32_t)rb[6] << 8) | rb[7];
            if (!ts0a_set) { ts0_audio = rts; ts0a_set = 1; }
            /* Opus RTP clock is 48kHz */
            uint32_t dts = (uint32_t)(((double)(rts - ts0_audio) / 48000.0) * 1000.0);

            if (g_seq_written) { /* only write audio after video seq header */
                write_flv_audio_tag(out, rb, len, dts);
                audio_tags++;
            }
            continue;
        }

        /* ---- VIDEO (channel 0) ---- */
        if (ch != 0 || len < 13) continue;

        uint32_t rts = ((uint32_t)rb[4] << 24) | ((uint32_t)rb[5] << 16) |
                       ((uint32_t)rb[6] << 8) | rb[7];
        if (!ts0v_set) { ts0_video = rts; ts0v_set = 1; }
        double t = (double)(rts - ts0_video) / 90000.0;

        int off = 12 + (rb[0] & 0x0F) * 4;
        if (rb[0] & 0x10) {
            off += 4 + (((int)rb[off + 2] << 8) | rb[off + 3]) * 4;
        }
        if (off >= len) continue;

        uint8_t* pl = rb + off;
        int psz = len - off;
        if (psz < 1) continue;

        uint8_t nt = pl[0] & 0x1F;
        uint8_t nri = pl[0] & 0x60;

        if (fr % 90 == 0) srt_chk(t);

        /* Capture SPS/PPS */
        if (nt == 24) capture_sps_pps(pl, psz);
        else if (nt == 7 && psz <= (int)sizeof(g_sps)) { memcpy(g_sps, pl, psz); g_sps_len = psz; }
        else if (nt == 8 && psz <= (int)sizeof(g_pps)) { memcpy(g_pps, pl, psz); g_pps_len = psz; }

        /* Write seq header before first frame */
        if (!g_seq_written && g_sps_len > 0 && g_pps_len > 0) {
            write_avc_seq_header(out, (uint32_t)(t * 1000));
        }

        /* Skip until seq header written */
        if (nt == 7 || nt == 8 || nt == 9) continue;
        if (!g_seq_written) continue;

        /* Single NAL (1-23) */
        if (nt >= 1 && nt <= 23) {
            int idr = (nt == 5);
            uint32_t dts = (uint32_t)(t * 1000);
            flvtag_initavc(&tag, dts, 0,
                idr ? flvtag_frametype_keyframe : flvtag_frametype_interframe);
            flvtag_avcwritenal(&tag, pl, psz);
            try_inject(&tag, t, &cc);
            flv_write_tag(out, &tag);
            fr++;
        }
        /* FU-A (28) */
        else if (nt == 28 && psz >= 2) {
            uint8_t fh = pl[1];
            int fs = (fh >> 7) & 1;
            int fe = (fh >> 6) & 1;
            uint8_t rt = fh & 0x1F;

            if (fs) {
                nb[0] = nri | rt;
                memcpy(nb + 1, pl + 2, psz - 2);
                nl = 1 + (psz - 2);
                ni = (rt == 5);
            } else if (nl > 0 && nl + (psz - 2) < MAX_NAL) {
                memcpy(nb + nl, pl + 2, psz - 2);
                nl += (psz - 2);
            }

            if (fe && nl > 0) {
                uint32_t dts = (uint32_t)(t * 1000);
                flvtag_initavc(&tag, dts, 0,
                    ni ? flvtag_frametype_keyframe : flvtag_frametype_interframe);
                flvtag_avcwritenal(&tag, nb, nl);
                try_inject(&tag, t, &cc);
                flv_write_tag(out, &tag);
                fr++;
                nl = 0;
            }
        }
        /* STAP-A (24) â€” write non-SPS/PPS NALs as frames */
        else if (nt == 24 && psz > 1) {
            int pos = 1;
            while (pos + 2 < psz) {
                int ns = (pl[pos] << 8) | pl[pos + 1];
                pos += 2;
                if (pos + ns > psz) break;
                uint8_t st = pl[pos] & 0x1F;
                /* Skip SPS/PPS in STAP-A, only write slice NALs */
                if (st != 7 && st != 8 && st != 9) {
                    int idr = (st == 5);
                    uint32_t dts = (uint32_t)(t * 1000);
                    flvtag_initavc(&tag, dts, 0,
                        idr ? flvtag_frametype_keyframe : flvtag_frametype_interframe);
                    flvtag_avcwritenal(&tag, pl + pos, ns);
                    try_inject(&tag, t, &cc);
                    flv_write_tag(out, &tag);
                    fr++;
                }
                pos += ns;
            }
        }

        /* Periodic flush */
        if (fr % 30 == 0) fflush(out);
    }

    free(nb);
    fprintf(stderr, "[CC] done fr=%lu cc=%lu audio=%lu pk=%lu\n", fr, cc, audio_tags, pk);
    fflush(stderr);

    rtsp_cmd(&r, "TEARDOWN", r.url, NULL);
    flvtag_free(&tag);
    flv_close(out);
    if (g_srt) srt_free(g_srt);
    closesocket(r.s);

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}

