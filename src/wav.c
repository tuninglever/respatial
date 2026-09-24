#include "respatial.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* RIFF/WAVE, little-endian. Read: 16/24-bit PCM or 32-bit float.
 * Write: 16-bit PCM. Samples are channel-major (nchan x nsamp).
 */

static unsigned int rd_u32le(const unsigned char *p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) |
           ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

static unsigned short rd_u16le(const unsigned char *p) {
    return (unsigned short)(p[0] | (p[1] << 8));
}

static short rd_s16le(const unsigned char *p) {
    return (short)rd_u16le(p);
}

static float rd_s24le(const unsigned char *p) {
    int v = (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16);
    if (v & 0x800000)
        v -= 0x1000000;
    return (float)v / 8388608.0f;
}

static int parse_header(const unsigned char *buf, long sz,
                        int *fmt, int *nch, int *sr, int *bits,
                        long *datapos, long *datalen) {
    const unsigned char *p;
    if (sz < 44 || memcmp(buf, "RIFF", 4) != 0 || memcmp(buf + 8, "WAVE", 4) != 0)
        return -1;
    *fmt = 0; *nch = 0; *sr = 0; *bits = 0; *datapos = 0; *datalen = 0;
    p = buf + 12;
    while (p + 8 <= buf + sz) {
        char id[5];
        long csize = (long)rd_u32le(p + 4);
        memcpy(id, p, 4);
        id[4] = 0;
        if (csize < 0 || p + 8 + csize > buf + sz)
            break;
        if (strcmp(id, "fmt ") == 0 && csize >= 16) {
            *fmt = rd_u16le(p + 8);
            *nch = rd_u16le(p + 10);
            *sr = (int)rd_u32le(p + 12);
            *bits = rd_u16le(p + 22);
        } else if (strcmp(id, "data") == 0) {
            *datapos = (long)(p - buf) + 8;
            *datalen = csize;
        }
        p += 8 + csize + (csize & 1);
    }
    return (*fmt && *nch && *sr && *bits && *datalen > 0) ? 0 : -1;
}

static unsigned char *read_whole(const char *path, long *sz) {
    FILE *f = fopen(path, "rb");
    unsigned char *buf;
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    *sz = ftell(f);
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    buf = malloc((size_t)*sz);
    if (!buf || fread(buf, 1, (size_t)*sz, f) != (size_t)*sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return buf;
}

int wav_info(const char *path, int *nsamp, int *nchan, int *srate) {
    long sz;
    unsigned char *buf = read_whole(path, &sz);
    int fmt, nch, sr, bits;
    long dp, dl;
    int rc;
    if (!buf)
        return -1;
    rc = parse_header(buf, sz, &fmt, &nch, &sr, &bits, &dp, &dl);
    free(buf);
    if (rc)
        return -1;
    if (!((fmt == 1 && (bits == 16 || bits == 24)) || (fmt == 3 && bits == 32)))
        return -1;
    *nsamp = (int)(dl / ((long)nch * (bits / 8)));
    *nchan = nch;
    *srate = sr;
    return 0;
}

int wav_read_samples(const char *path, float *data) {
    long sz;
    unsigned char *buf = read_whole(path, &sz);
    int fmt, nch, sr, bits;
    long dp, dl;
    int nframes, c, i;
    const unsigned char *d;
    if (!buf)
        return -1;
    if (parse_header(buf, sz, &fmt, &nch, &sr, &bits, &dp, &dl)) {
        free(buf);
        return -1;
    }
    nframes = (int)(dl / ((long)nch * (bits / 8)));
    d = buf + dp;
    if (fmt == 1 && bits == 16) {
        for (c = 0; c < nch; c++)
            for (i = 0; i < nframes; i++)
                data[(size_t)c * nframes + i] =
                    (float)rd_s16le(d + ((size_t)i * nch + c) * 2) / 32768.0f;
    } else if (fmt == 1 && bits == 24) {
        for (c = 0; c < nch; c++)
            for (i = 0; i < nframes; i++)
                data[(size_t)c * nframes + i] =
                    rd_s24le(d + ((size_t)i * nch + c) * 3);
    } else {
        for (c = 0; c < nch; c++)
            for (i = 0; i < nframes; i++) {
                float v;
                memcpy(&v, d + ((size_t)i * nch + c) * 4, 4);
                data[(size_t)c * nframes + i] = v;
            }
    }
    free(buf);
    return 0;
}

int wav_write(const char *path, const float *data, int nsamp, int nchan, int srate) {
    FILE *f = fopen(path, "wb");
    unsigned char hdr[44];
    unsigned char *p;
    long datalen = (long)nsamp * nchan * 2;
    unsigned int u32;
    unsigned short u16;
    int i, c;
    if (!f)
        return -1;
    p = hdr;
    memcpy(p, "RIFF", 4); p += 4;
    u32 = (unsigned int)(36 + datalen);
    memcpy(p, &u32, 4); p += 4;
    memcpy(p, "WAVE", 4); p += 4;
    memcpy(p, "fmt ", 4); p += 4;
    u32 = 16;
    memcpy(p, &u32, 4); p += 4;
    u16 = 1;
    memcpy(p, &u16, 2); p += 2;
    u16 = (unsigned short)nchan;
    memcpy(p, &u16, 2); p += 2;
    u32 = (unsigned int)srate;
    memcpy(p, &u32, 4); p += 4;
    u32 = (unsigned int)srate * nchan * 2;
    memcpy(p, &u32, 4); p += 4;
    u16 = (unsigned short)(nchan * 2);
    memcpy(p, &u16, 2); p += 2;
    u16 = 16;
    memcpy(p, &u16, 2); p += 2;
    memcpy(p, "data", 4); p += 4;
    u32 = (unsigned int)datalen;
    memcpy(p, &u32, 4); p += 4;
    if (fwrite(hdr, 1, 44, f) != 44) { fclose(f); return -1; }
    for (i = 0; i < nsamp; i++)
        for (c = 0; c < nchan; c++) {
            float v = data[(size_t)c * nsamp + i];
            unsigned char s[2];
            short sv;
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            sv = (short)(v * 32767.0f);
            s[0] = (unsigned char)(sv & 0xff);
            s[1] = (unsigned char)((sv >> 8) & 0xff);
            if (fwrite(s, 1, 2, f) != 2) { fclose(f); return -1; }
        }
    fclose(f);
    return 0;
}
