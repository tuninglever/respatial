"""Hann vs Hamming window experiment for the respatial STFT pipeline.

Reproduces the pipeline's STFT/iSTFT (periodic window, frame 4096, hop 1024,
iSTFT = overlap-add of ifft*w / sum(win^2), one-frame zero padding per side)
in numpy, then compares:
  1. identity path (x -> STFT -> iSTFT) reconstruction error
  2. per-bin spatial statistics (median/p5/p95/range/ILD) on Mahler 0-66 s
  3. leakage: weak tone adjacent to a 40 dB stronger tone
"""
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("respatial", "respatial.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)

N, H = 4096, 1024
ii = np.arange(N)
WINS = {
    "hann": 0.5 * (1.0 - np.cos(2 * np.pi * ii / N)),
    "hamming": 0.54 - 0.46 * np.cos(2 * np.pi * ii / N),
}


def stft_padded(x, w):
    xp = np.concatenate([np.zeros(N), x, np.zeros(N)])
    fr = np.lib.stride_tricks.sliding_window_view(xp, N)[::H]
    return np.fft.rfft(fr * w, axis=1)


def norm_array(nt, w):
    total = N + H * (nt - 1)
    norm = np.zeros(total)
    for k in range(nt):
        norm[k * H:k * H + N] += w * w
    return norm


def istft_unpad(X, w, n):
    y = np.fft.irfft(X, axis=1) * w
    nt = X.shape[0]
    total = N + H * (nt - 1)
    out = np.zeros(total)
    for k in range(nt):
        out[k * H:k * H + N] += y[k]
    return out[N:N + n] / norm_array(nt, w)[N:N + n]


def per_bin_stats(xL, xR, w):
    XL = stft_padded(xL, w)
    XR = stft_padded(xR, w)
    magL, magR = np.abs(XL), np.abs(XR)
    amp = magL + magR
    m = amp > 0.05 * amp.max()
    ild = 20 * np.log10(np.maximum(magL, 1e-9) / np.maximum(magR, 1e-9))
    d = 2 * np.degrees(np.arctan(10 ** (-ild / 20))) - 90
    d = d[m]
    return dict(
        n_active=int(m.sum()),
        med=float(np.median(d)), p5=float(np.percentile(d, 5)),
        p95=float(np.percentile(d, 95)),
        p95abs=float(np.percentile(np.abs(d), 95)),
        ild_med=float(np.median(ild[m])),
    )


def main():
    L, R, sr = rs.load_stereo("report/Mahler/Mahler-7-2mvt.wav")
    n = min(int(66 * sr), len(L))
    xL, xR = L[:n].astype(np.float64), R[:n].astype(np.float64)

    print("== 1. identity path (padded STFT/iSTFT, no modification) ==")
    for name, w in WINS.items():
        X = stft_padded(xL, w)
        y = istft_unpad(X, w, n)
        err = np.abs(y - xL).max()
        snr = 10 * np.log10(np.sum(xL ** 2) / np.sum((y - xL) ** 2))
        print(f"  {name:8s} max err {err:.2e}  SNR {snr:7.1f} dB")

    print("== 2. per-bin spatial stats, Mahler 0-66 s (both channels) ==")
    stats = {}
    for name, w in WINS.items():
        stats[name] = per_bin_stats(xL, xR, w)
        s = stats[name]
        print(f"  {name:8s} active {s['n_active']:6d}  median {s['med']:+6.2f}  "
              f"p5 {s['p5']:+6.2f}  p95 {s['p95']:+6.2f}  "
              f"range {s['p95']-s['p5']:5.2f}  p95|d| {s['p95abs']:5.2f}  "
              f"ILD med {s['ild_med']:+5.2f}")
    a, b = stats["hann"], stats["hamming"]
    print(f"  |diff|          median {abs(a['med']-b['med']):.3f}  "
          f"p5 {abs(a['p5']-b['p5']):.3f}  p95 {abs(a['p95']-b['p95']):.3f}  "
          f"p95|d| {abs(a['p95abs']-b['p95abs']):.3f}  "
          f"ILD med {abs(a['ild_med']-b['ild_med']):.3f} dB")

    print("== 3. leakage: tone at bin 101, 40 dB below tone at bin 100 ==")
    t = np.arange(8 * N) / sr
    f1, f2 = 100 * sr / N, 101 * sr / N
    x = 1.0 * np.cos(2 * np.pi * f1 * t) + 1e-2 * np.cos(2 * np.pi * f2 * t)
    for name, w in WINS.items():
        X = stft_padded(x, w)[8]
        mag = np.abs(X)
        bias = 20 * np.log10(mag[101] / 1e-2 / mag[100])
        print(f"  {name:8s} weak-tone measured level {bias:+6.2f} dB above true "
              f"(-40 dB)  [leakage from neighbor dominates]")
    print("  on-grid kernel (response at bin offset m, rel to main peak):")
    for name, w in WINS.items():
        W = np.fft.rfft(w)
        rel = {m: abs(W[m]) / abs(W[0]) for m in (1, 2, 3, 4)}
        print(f"  {name:8s} m=1 {20*np.log10(rel[1]):6.2f} dB  "
              f"m=2 {20*np.log10(max(rel[2],1e-12)):6.2f} dB  "
              f"m=3 {20*np.log10(max(rel[3],1e-12)):6.2f} dB  "
              f"m=4 {20*np.log10(max(rel[4],1e-12)):6.2f} dB")


if __name__ == "__main__":
    main()
