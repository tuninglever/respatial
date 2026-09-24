#include "respatial.h"
#include <math.h>
#include <stdlib.h>

int stft_nframes(int n, int frame, int hop) {
    return n >= frame ? (n - frame) / hop + 1 : 0;
}

static float *make_window(int frame) {
    float *w = malloc(sizeof(float) * (size_t)frame);
    int i;
    for (i = 0; i < frame; i++)
        w[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / frame));
    return w;
}

void stft_forward(const float *x, int n, int frame, int hop, float *X) {
    int nt = stft_nframes(n, frame, hop);
    int nf = frame / 2 + 1;
    float *win = make_window(frame);
    float *re = malloc(sizeof(float) * (size_t)frame);
    float *im = malloc(sizeof(float) * (size_t)frame);
    int i, k, f;
    for (i = 0; i < nt; i++) {
        const float *p = x + i * hop;
        for (k = 0; k < frame; k++) {
            re[k] = p[k] * win[k];
            im[k] = 0.0f;
        }
        fft(re, im, frame, -1);
        float *o = X + (size_t)i * nf * 2;
        for (f = 0; f < nf; f++) {
            o[2 * f] = re[f];
            o[2 * f + 1] = im[f];
        }
    }
    free(win);
    free(re);
    free(im);
}

void stft_inverse(const float *X, int nt, int n, int frame, int hop, float *y) {
    int nf = frame / 2 + 1;
    int total = frame + hop * (nt - 1);
    float *win = make_window(frame);
    float *out = calloc((size_t)total, sizeof(float));
    float *norm = calloc((size_t)total, sizeof(float));
    float *re = malloc(sizeof(float) * (size_t)frame);
    float *im = malloc(sizeof(float) * (size_t)frame);
    int i, k, f;
    for (i = 0; i < nt; i++) {
        const float *p = X + (size_t)i * nf * 2;
        for (f = 0; f < nf; f++) {
            re[f] = p[2 * f];
            im[f] = p[2 * f + 1];
        }
        for (f = nf; f < frame; f++) {
            re[f] = re[frame - f];
            im[f] = -im[frame - f];
        }
        fft(re, im, frame, +1);
        float inv = 1.0f / frame;
        for (k = 0; k < frame; k++) {
            out[i * hop + k] += re[k] * win[k] * inv;
            norm[i * hop + k] += win[k] * win[k];
        }
    }
    for (i = 0; i < n && i < total; i++)
        y[i] = norm[i] > 1e-8f ? out[i] / norm[i] : 0.0f;
    free(win);
    free(out);
    free(norm);
    free(re);
    free(im);
}
