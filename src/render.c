#include "respatial.h"
#include <stddef.h>

/* Soft mask for component k:
 *   mask(t,f) = D(f,k) * a_k(t) / sum_j D(f,j) * a_j(t),
 * with a_j(t) = (AL(t,j) + AR(t,j)) / 2. Masks partition each bin,
 * so sum_k mask_k(t,f) = 1.
 */
void stem_mask(const float *D, const float *AL, const float *AR,
               int nt, int nf, int rank, int k, float *mask) {
    int t, f, j;
    for (t = 0; t < nt; t++) {
        for (f = 0; f < nf; f++) {
            double tot = 0.0, sel = 0.0;
            for (j = 0; j < rank; j++) {
                double a = 0.5 * (double)(AL[(size_t)t * rank + j] + AR[(size_t)t * rank + j]);
                double c = (double)D[(size_t)f * rank + j] * a;
                tot += c;
                if (j == k)
                    sel = c;
            }
            mask[(size_t)t * nf + f] = (float)(sel / (tot + 1e-12));
        }
    }
}
