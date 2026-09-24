#include "respatial.h"
#include <math.h>
#include <stdlib.h>

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Mask-weighted spatial fingerprint: median ILD (dB) and circular-mean
 * ICPD (rad) over bins where mask > 0.01 and the bin carries real energy
 * (max(|L|,|R|) > 1% of the loudest bin). Silent bins have arbitrary L/R
 * ratios from leakage and would otherwise contaminate the median.
 */
void fingerprint(const float *XL, const float *XR, const float *mask,
                 int nt, int nf, double *ild_db, double *icpd_rad, int *n_bins) {
    size_t cap = (size_t)nt * nf;
    double *vals = malloc(sizeof(double) * cap);
    size_t nv = 0;
    double sr = 0.0, si = 0.0;
    const double eps = 1e-12;
    int t, f;

    double maxmag = 0.0;
    for (t = 0; t < nt; t++)
        for (f = 0; f < nf; f++) {
            size_t idx = (size_t)t * nf + f;
            double lr = XL[2 * idx], li = XL[2 * idx + 1];
            double rr = XR[2 * idx], ri = XR[2 * idx + 1];
            double ml = sqrt(lr * lr + li * li);
            double mr = sqrt(rr * rr + ri * ri);
            double m = ml > mr ? ml : mr;
            if (m > maxmag)
                maxmag = m;
        }
    double gate = 0.01 * maxmag;

    for (t = 0; t < nt; t++)
        for (f = 0; f < nf; f++) {
            size_t idx = (size_t)t * nf + f;
            double w = mask[idx];
            if (w < 0.01)
                continue;
            double lr = XL[2 * idx], li = XL[2 * idx + 1];
            double rr = XR[2 * idx], ri = XR[2 * idx + 1];
            double magl = sqrt(lr * lr + li * li);
            double magr = sqrt(rr * rr + ri * ri);
            if (magl < gate && magr < gate)
                continue;
            if (magl < eps || magr < eps)
                continue;
            vals[nv++] = 20.0 * log10(magl / magr);
            double cr = lr * rr + li * ri;
            double ci = li * rr - lr * ri;
            double ph = atan2(ci, cr);
            sr += w * cos(ph);
            si += w * sin(ph);
        }
    *n_bins = (int)nv;
    if (nv > 0) {
        qsort(vals, nv, sizeof(double), cmp_double);
        *ild_db = nv % 2 ? vals[nv / 2] : 0.5 * (vals[nv / 2 - 1] + vals[nv / 2]);
        *icpd_rad = atan2(si, sr);
    } else {
        *ild_db = 0.0;
        *icpd_rad = 0.0;
    }
    free(vals);
}
