#include "respatial.h"
#include <math.h>

void fft(float *re, float *im, int n, int sign) {
    int i, j, len, k, bit;
    for (i = 1, j = 0; i < n; i++) {
        bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (len = 2; len <= n; len <<= 1) {
        double ang = sign * 2.0 * M_PI / len;
        float wr = (float)cos(ang), wi = (float)sin(ang);
        for (i = 0; i < n; i += len) {
            float cur_r = 1.0f, cur_i = 0.0f;
            for (k = 0; k < len / 2; k++) {
                int a = i + k, b = i + k + len / 2;
                float ur = re[a], ui = im[a];
                float vr = re[b] * cur_r - im[b] * cur_i;
                float vi = re[b] * cur_i + im[b] * cur_r;
                re[a] = ur + vr; im[a] = ui + vi;
                re[b] = ur - vr; im[b] = ui - vi;
                float nr = cur_r * wr - cur_i * wi;
                cur_i = cur_r * wi + cur_i * wr;
                cur_r = nr;
            }
        }
    }
}
