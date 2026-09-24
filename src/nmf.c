#include "respatial.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Deterministic LCG so analyze and remix produce the same decomposition. */
static unsigned int s_seed;

static float frand(void) {
    s_seed = s_seed * 1103515245u + 12345u;
    return (float)((s_seed >> 8) / 16777216.0);
}

/* NMF of stacked magnitudes [VL; VR] (n x nf, n = 2*nt) with one shared
 * basis D (nf x rank) and one activation matrix W (rank x n), split back
 * into AL / AR. Multiplicative updates on the Euclidean objective.
 */
void nmf_shared(const float *VL, const float *VR, int nt, int nf, int rank,
                int iters, float *D, float *AL, float *AR) {
    size_t m = (size_t)nf, n = (size_t)2 * nt;
    float *Vc = malloc(sizeof(float) * n * m);
    memcpy(Vc, VL, sizeof(float) * (size_t)nt * nf);
    memcpy(Vc + (size_t)nt * nf, VR, sizeof(float) * (size_t)nt * nf);

    s_seed = 12345u;
    double vsum = 0.0;
    size_t i, t;
    int f, k, j, it;
    for (i = 0; i < n * m; i++)
        vsum += Vc[i];
    float scale = (float)(vsum / (n * m));
    for (k = 0; k < rank; k++)
        for (f = 0; f < nf; f++)
            D[(size_t)f * rank + k] = scale * (0.5f + frand());

    float *W = malloc(sizeof(float) * (size_t)rank * n);
    for (i = 0; i < (size_t)rank * n; i++)
        W[i] = 0.5f + frand();

    float *DTV = malloc(sizeof(float) * (size_t)rank * n);
    float *DTDW = malloc(sizeof(float) * (size_t)rank * n);
    float *VWt = malloc(sizeof(float) * (size_t)nf * rank);
    float *DWWt = malloc(sizeof(float) * (size_t)nf * rank);
    float *DTD = malloc(sizeof(float) * (size_t)rank * rank);
    float *WWT = malloc(sizeof(float) * (size_t)rank * rank);
    const float eps = 1e-10f;

    for (it = 0; it < iters; it++) {
        for (t = 0; t < n; t++) {
            const float *v = Vc + t * m;
            for (k = 0; k < rank; k++) {
                double acc = 0.0;
                for (f = 0; f < nf; f++)
                    acc += (double)D[(size_t)f * rank + k] * v[f];
                DTV[(size_t)k * n + t] = (float)acc;
            }
        }
        for (k = 0; k < rank; k++)
            for (j = k; j < rank; j++) {
                double acc = 0.0;
                for (f = 0; f < nf; f++)
                    acc += (double)D[(size_t)f * rank + k] * D[(size_t)f * rank + j];
                DTD[(size_t)k * rank + j] = (float)acc;
                DTD[(size_t)j * rank + k] = (float)acc;
            }
        for (t = 0; t < n; t++)
            for (k = 0; k < rank; k++) {
                double acc = 0.0;
                for (j = 0; j < rank; j++)
                    acc += (double)DTD[(size_t)k * rank + j] * W[(size_t)j * n + t];
                DTDW[(size_t)k * n + t] = (float)acc;
            }
        for (i = 0; i < (size_t)rank * n; i++)
            W[i] *= DTV[i] / (DTDW[i] + eps);

        for (f = 0; f < nf; f++)
            for (k = 0; k < rank; k++) {
                double acc = 0.0;
                for (t = 0; t < n; t++)
                    acc += (double)Vc[(size_t)t * m + f] * W[(size_t)k * n + t];
                VWt[(size_t)f * rank + k] = (float)acc;
            }
        for (k = 0; k < rank; k++)
            for (j = k; j < rank; j++) {
                double acc = 0.0;
                for (t = 0; t < n; t++)
                    acc += (double)W[(size_t)k * n + t] * W[(size_t)j * n + t];
                WWT[(size_t)k * rank + j] = (float)acc;
                WWT[(size_t)j * rank + k] = (float)acc;
            }
        for (f = 0; f < nf; f++)
            for (k = 0; k < rank; k++) {
                double acc = 0.0;
                for (j = 0; j < rank; j++)
                    acc += (double)D[(size_t)f * rank + j] * WWT[(size_t)j * rank + k];
                DWWt[(size_t)f * rank + k] = (float)acc;
            }
        for (i = 0; i < (size_t)nf * rank; i++)
            D[i] *= VWt[i] / (DWWt[i] + eps);
    }

    for (k = 0; k < rank; k++)
        for (t = 0; t < (size_t)nt; t++) {
            AL[(size_t)t * rank + k] = W[(size_t)k * n + t];
            AR[(size_t)t * rank + k] = W[(size_t)k * n + (size_t)nt + t];
        }
    free(Vc);
    free(W);
    free(DTV);
    free(DTDW);
    free(VWt);
    free(DWWt);
    free(DTD);
    free(WWT);
}
