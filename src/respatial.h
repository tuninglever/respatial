#ifndef RESPATIAL_H
#define RESPATIAL_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Flat, ctypes-friendly API: raw float buffers, int sizes, no structs by
 * value, no returned malloc'd memory.
 *
 * STFT buffer layout: (nframes x nfreq x 2) float32, re/im interleaved,
 * nfreq = frame/2 + 1.
 * NMF layout: D is (nf x rank), activations are (nt x rank), row-major.
 * wav samples: channel-major (nchan x nsamp) float32.
 */

int  stft_nframes(int n, int frame, int hop);
void stft_forward(const float *x, int n, int frame, int hop, float *X);
void stft_inverse(const float *X, int nt, int n, int frame, int hop, float *y);

void nmf_shared(const float *VL, const float *VR, int nt, int nf, int rank,
                int iters, float *D, float *AL, float *AR);

void fingerprint(const float *XL, const float *XR, const float *mask,
                 int nt, int nf, double *ild_db, double *icpd_rad, int *n_bins);

void stem_mask(const float *D, const float *AL, const float *AR,
               int nt, int nf, int rank, int k, float *mask);

int wav_info(const char *path, int *nsamp, int *nchan, int *srate);
int wav_read_samples(const char *path, float *data);
int wav_write(const char *path, const float *data, int nsamp, int nchan, int srate);

void fft(float *re, float *im, int n, int sign);

#ifdef __cplusplus
}
#endif

#endif
