"""ctypes interface to the respatial C kernels (librespatial).

C owns the compute (STFT, NMF, fingerprint, masks, wav I/O); Python owns
the pipeline orchestration.

Spatial convention: deviation from center d in degrees, -90..+90 (each
channel spans 90 deg; total stage up to 180 deg). Pan parameter
p = d/90, equal-power g_L = cos(pi/4(1+p)), g_R = sin(pi/4(1+p)).
"""
import argparse
import ctypes as ct
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = ct.CDLL(os.path.join(_HERE, "librespatial.dylib"))
_P = ct.POINTER(ct.c_float)

_LIB.stft_nframes.argtypes = [ct.c_int, ct.c_int, ct.c_int]
_LIB.stft_nframes.restype = ct.c_int
_LIB.stft_forward.argtypes = [_P, ct.c_int, ct.c_int, ct.c_int, _P]
_LIB.stft_inverse.argtypes = [_P, ct.c_int, ct.c_int, ct.c_int, ct.c_int, _P]
_LIB.nmf_shared.argtypes = [_P, _P, ct.c_int, ct.c_int, ct.c_int, ct.c_int, _P, _P, _P]
_LIB.fingerprint.argtypes = [_P, _P, _P, ct.c_int, ct.c_int,
                             ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                             ct.POINTER(ct.c_int)]
_LIB.stem_mask.argtypes = [_P, _P, _P, ct.c_int, ct.c_int, ct.c_int, ct.c_int, _P]
_LIB.wav_info.argtypes = [ct.c_char_p, ct.POINTER(ct.c_int),
                          ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
_LIB.wav_info.restype = ct.c_int
_LIB.wav_read_samples.argtypes = [ct.c_char_p, _P]
_LIB.wav_read_samples.restype = ct.c_int
_LIB.wav_write.argtypes = [ct.c_char_p, _P, ct.c_int, ct.c_int, ct.c_int]
_LIB.wav_write.restype = ct.c_int


def _f(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a.ctypes.data_as(_P)


def _interleaved(X):
    X = np.ascontiguousarray(X)
    out = np.empty(X.shape + (2,), dtype=np.float32)
    out[..., 0] = X.real
    out[..., 1] = X.imag
    return out


def load_stereo(path):
    ns, nc, sr = ct.c_int(), ct.c_int(), ct.c_int()
    if _LIB.wav_info(os.fsencode(path), ct.byref(ns), ct.byref(nc), ct.byref(sr)):
        raise IOError(f"cannot read wav: {path}")
    if nc.value != 2:
        raise IOError(f"expected stereo, got {nc.value} channels: {path}")
    data = np.empty((2, ns.value), dtype=np.float32)
    if _LIB.wav_read_samples(os.fsencode(path), data.ctypes.data_as(_P)):
        raise IOError(f"cannot read samples: {path}")
    return data[0], data[1], sr.value


def save_stereo(path, L, R, sr):
    data = np.empty((2, len(L)), dtype=np.float32)
    data[0] = L
    data[1] = R
    if _LIB.wav_write(os.fsencode(path), data.ctypes.data_as(_P), len(L), 2, int(sr)):
        raise IOError(f"cannot write wav: {path}")


def save_mono(path, x, sr):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if _LIB.wav_write(os.fsencode(path), x.ctypes.data_as(_P), len(x), 1, int(sr)):
        raise IOError(f"cannot write wav: {path}")


def ild_to_dev_deg(ild_db):
    """median ILD (dB, L/R) -> deviation from center, -90..+90 deg.
    d = 2*atan(1/r) - 90 with r = 10^(ild/20); +ILD (L louder) -> left (-)."""
    r = 10.0 ** (ild_db / 20.0)
    return 2.0 * np.degrees(np.arctan(1.0 / r)) - 90.0


def dev_to_p(dev_deg):
    return float(np.clip(dev_deg / 90.0, -1.0, 1.0))


def parse_pans(spec):
    """pans spec -> {k: p}. spec is a pans file (first token k, last token
    pan_deg), or an inline 'k=deg,k=deg,...' string. deg = deviation from
    center, -90..+90."""
    if os.path.exists(spec) or "=" not in spec:
        pans = {}
        with open(spec) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                toks = line.split()
                pans[int(toks[0])] = float(toks[-1])
    else:
        pans = {}
        for part in spec.split(","):
            part = part.strip()
            if part:
                k, d = part.split("=")
                pans[int(k)] = float(d)
    return {k: dev_to_p(d) for k, d in pans.items()}


class Respatial:
    def __init__(self, frame=4096, hop=1024):
        if frame & (frame - 1):
            raise ValueError("frame must be a power of two")
        self.frame = frame
        self.hop = hop
        self.nfreq = frame // 2 + 1

    def nframes(self, n):
        return _LIB.stft_nframes(n, self.frame, self.hop)

    def stft(self, x):
        n = len(x)
        nt = self.nframes(n)
        buf = np.empty((nt, self.nfreq, 2), dtype=np.float32)
        _LIB.stft_forward(_f(x), n, self.frame, self.hop, buf.ctypes.data_as(_P))
        return buf[..., 0] + 1j * buf[..., 1]

    def istft(self, X, n):
        nt = X.shape[0]
        y = np.empty(n, dtype=np.float32)
        _LIB.stft_inverse(_interleaved(X).ctypes.data_as(_P), nt, n,
                          self.frame, self.hop, y.ctypes.data_as(_P))
        return y

    def stft_padded(self, x):
        """STFT of x zero-padded by one frame on each side. Per-bin modified
        frames (warp/mask) then iSTFT without edge blow-up in the interior:
        every interior sample is covered by a frame with win >= 0.5.
        Returns (X, pad)."""
        pad = self.frame
        xp = np.zeros(len(x) + 2 * pad, dtype=np.float32)
        xp[pad:pad + len(x)] = x
        return self.stft(xp), pad

    def istft_unpad(self, X, n, pad):
        total = self.frame + self.hop * (X.shape[0] - 1)
        y = self.istft(X, total)
        return y[pad:pad + n]

    def nmf(self, VL, VR, rank, iters=300):
        nt, nf = VL.shape
        D = np.empty((nf, rank), dtype=np.float32)
        AL = np.empty((nt, rank), dtype=np.float32)
        AR = np.empty((nt, rank), dtype=np.float32)
        _LIB.nmf_shared(_f(VL), _f(VR), nt, nf, rank, iters,
                        D.ctypes.data_as(_P), AL.ctypes.data_as(_P),
                        AR.ctypes.data_as(_P))
        return D, AL, AR

    def fingerprint(self, XL, XR, mask):
        nt, nf = XL.shape
        ild, icpd, nb = ct.c_double(), ct.c_double(), ct.c_int()
        _LIB.fingerprint(_interleaved(XL).ctypes.data_as(_P),
                         _interleaved(XR).ctypes.data_as(_P), _f(mask),
                         nt, nf, ct.byref(ild), ct.byref(icpd), ct.byref(nb))
        return ild.value, icpd.value, nb.value

    def stem_mask(self, D, AL, AR, k):
        nt, rank = AL.shape
        nf = D.shape[0]
        mask = np.empty((nt, nf), dtype=np.float32)
        _LIB.stem_mask(_f(D), _f(AL), _f(AR), nt, nf, rank, k,
                       mask.ctypes.data_as(_P))
        return mask

    def decompose(self, path, rank=8, iters=300, t0=0.0, t1=None):
        """STFT + shared-D NMF + spatial fingerprints for [t0, t1] (s).
        Returns dict with the segment arrays, STFTs, NMF factors, and comps."""
        L, R, sr = load_stereo(path)
        i0 = max(0, min(int(t0 * sr), len(L)))
        i1 = len(L) if t1 is None else max(i0, min(int(t1 * sr), len(L)))
        Ls, Rs = L[i0:i1], R[i0:i1]
        XL, pad = self.stft_padded(Ls)
        XR, _ = self.stft_padded(Rs)
        D, AL, AR = self.nmf(np.abs(XL), np.abs(XR), rank, iters)
        freqs = np.arange(self.nfreq) * sr / self.frame
        etot = float(np.sum(AL.astype(np.float64) ** 2) +
                     np.sum(AR.astype(np.float64) ** 2))
        comps = []
        for k in range(rank):
            mask = self.stem_mask(D, AL, AR, k)
            ild, icpd, nb = self.fingerprint(XL, XR, mask)
            dk = D[:, k]
            centroid = float(np.sum(freqs * dk) / np.sum(dk)) if np.sum(dk) > 0 else 0.0
            e = float(np.sum(AL[:, k] ** 2) + np.sum(AR[:, k] ** 2))
            comps.append({
                "k": k,
                "ild_db": ild,
                "icpd_rad": icpd,
                "centroid_hz": centroid,
                "act_frac": e / etot if etot > 0 else 0.0,
                "n_bins": nb,
                "spectrum": dk.copy(),
                "act_L": AL[:, k].copy(),
                "act_R": AR[:, k].copy(),
                "dev_deg": ild_to_dev_deg(ild),
            })
        return {"L": Ls, "R": Rs, "sr": sr, "XL": XL, "XR": XR, "pad": pad,
                "D": D, "AL": AL, "AR": AR, "i0": i0, "i1": i1, "comps": comps}

    def analyze(self, path, rank=8, iters=300, t0=0.0, t1=None,
                outdir=None, clip_dur=3.0):
        X = self.decompose(path, rank, iters, t0, t1)
        if outdir is not None:
            self.export_clips(outdir, path, X, clip_dur=clip_dur)
        return X["comps"], X["sr"]

    def export_clips(self, outdir, path, X, clip_dur=3.0):
        """mono clip per activity (most active clip_dur window, peak-normalized)
        plus a pans.txt template; initial pan = detected deviation."""
        os.makedirs(outdir, exist_ok=True)
        n = X["i1"] - X["i0"]
        sr = X["sr"]
        mid = (X["XL"] + X["XR"]) * 0.5
        nt = X["AL"].shape[0]
        win = max(1, min(int(0.25 * sr / self.hop), nt))
        kernel = np.ones(win) / win
        lines = [
            f"# respatial pans  {os.path.basename(path)}  [{X['i0'] / sr:.2f}..{X['i1'] / sr:.2f}]s",
            "# columns: k  centroid_hz  ild_db  dev_deg  clip  pan_deg",
            "# pan_deg: target deviation from center, -90..+90 (0 = center)",
        ]
        for c in X["comps"]:
            k = c["k"]
            mask = self.stem_mask(X["D"], X["AL"], X["AR"], k)
            stem = self.istft_unpad(mask * mid, n, X["pad"])
            a = np.hypot(X["AL"][:, k].astype(np.float64),
                         X["AR"][:, k].astype(np.float64))
            tstar = int(np.argmax(np.convolve(a, kernel, mode="same")))
            cs = min(int(clip_dur * sr), n)
            start = max(0, min(tstar * self.hop - cs // 2, n - cs))
            clip = stem[start:start + cs]
            peak = float(np.max(np.abs(clip)))
            if peak > 1e-6:
                clip = clip * (0.9 / peak)
            name = f"clip_{k:02d}.wav"
            save_mono(os.path.join(outdir, name), clip, sr)
            lines.append(f"{k}  {c['centroid_hz']:.0f}  {c['ild_db']:+.2f}  "
                         f"{c['dev_deg']:+.1f}  {name}  {c['dev_deg']:+.1f}")
        with open(os.path.join(outdir, "pans.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def remix(self, path, pans, rank=8, iters=300, out=None, t0=0.0, t1=None,
              hard=False, depth=2.0):
        """pans: {k: p} with p in [-1, 1] (p = dev_deg/90). Soft (default):
        re-renders each stem mono -> equal-power pan, stems sum per bin so
        the bin's position is the mask-weighted average of the target pans.
        Hard: each bin is panned to its dominant component's target (no
        averaging; wider stage, more position jumping). Residual
        (1 - sum masks) stays center in soft mode. depth > 0 scales each
        component by the inverse-distance gain r(d0)/r(d1) between its
        detected deviation d0 and target d1 (listener at `depth` half-stage
        widths; 0 = off). Output is [t0, t1]."""
        X = self.decompose(path, rank, iters, t0, t1)
        n = X["i1"] - X["i0"]
        XL, XR = X["XL"], X["XR"]
        D, AL, AR = X["D"], X["AL"], X["AR"]
        mid = (XL + XR) * 0.5
        nt, nf = XL.shape

        def dgain(k, p):
            if depth <= 0.0:
                return 1.0
            d0 = X["comps"][k]["dev_deg"]
            d1 = 90.0 * p
            return float(np.sqrt(depth ** 2 + (d0 / 90.0) ** 2) /
                         np.sqrt(depth ** 2 + (d1 / 90.0) ** 2))

        if hard:
            a = 0.5 * (AL.astype(np.float64) + AR.astype(np.float64))
            kstar = np.argmax(a[:, None, :] * D[np.newaxis, :, :], axis=2)
            ps = [float(np.clip(pans.get(k, 0.0), -1.0, 1.0)) for k in range(rank)]
            ph = (1.0 + np.array(ps)) * np.pi / 4.0
            gk = np.array([dgain(k, p) for k, p in enumerate(ps)])
            sel = ph[kstar]
            gs = gk[kstar]
            outL = self.istft_unpad(np.asarray(mid * np.cos(sel) * gs,
                                               dtype=np.complex64), n, X["pad"])
            outR = self.istft_unpad(np.asarray(mid * np.sin(sel) * gs,
                                               dtype=np.complex64), n, X["pad"])
            if out is not None:
                save_stereo(out, outL, outR, X["sr"])
            return outL, outR, X["sr"]
        outL = np.zeros(n, dtype=np.float32)
        outR = np.zeros(n, dtype=np.float32)
        summask = np.zeros((nt, nf), dtype=np.float32)
        for k in range(rank):
            mask = self.stem_mask(D, AL, AR, k)
            summask += mask
            stem = self.istft_unpad(mask * mid, n, X["pad"])
            p = float(np.clip(pans.get(k, 0.0), -1.0, 1.0))
            phi = (1.0 + p) * np.pi / 4.0
            g = dgain(k, p)
            outL += g * np.cos(phi) * stem
            outR += g * np.sin(phi) * stem
        residual = self.istft_unpad((1.0 - summask) * mid, n, X["pad"])
        outL += residual
        outR += residual
        if out is not None:
            save_stereo(out, outL, outR, X["sr"])
        return outL, outR, X["sr"]

    def warp_stage(self, path, stage_deg=None, blend=1.0, depth=2.0, rotate=0.0,
                   t0=0.0, t1=None, out=None):
        """expand/contract/rotate: per-bin ILD warp. Each bin's position d is
        first rigidly shifted by `rotate` degrees (d_rot = clip(d + rotate),
        the 'rotate the stereo mic' model), then the ILD is power-warped
        (ild_final = gamma * ild_rot). stage_deg (optional) solves for gamma
        so p95(|d_final|) = stage/2 over active bins; gamma = 1 when
        stage_deg is None (pure rotation). Each bin is renormalized to
        preserve its energy (|L'|^2+|R'|^2 = |L|^2+|R|^2); the per-bin gains
        are real and non-negative, so phase/ICPD are preserved. blend in
        [0,1] crossfades original -> warped (pure amplitude interpolation).
        depth > 0 adds a per-bin inverse-distance gain r(d0)/r(d_final)
        (listener at `depth` half-stage widths from the stage plane): content
        moved toward center gets a slight boost, as on a physical stage.
        Output is RMS-matched to the input (per-bin energy does not survive
        overlap-add under strong warps). Returns (L, R, sr, gamma,
        final_p95_abs_dev_deg)."""
        L, R, sr = load_stereo(path)
        i0 = max(0, min(int(t0 * sr), len(L)))
        i1 = len(L) if t1 is None else max(i0, min(int(t1 * sr), len(L)))
        Ls, Rs = L[i0:i1], R[i0:i1]
        n = i1 - i0
        XL, pad = self.stft_padded(Ls)
        XR, _ = self.stft_padded(Rs)
        magL, magR = np.abs(XL), np.abs(XR)
        amp = magL + magR
        m = amp > 0.05 * amp.max()
        ild = 20.0 * np.log10(np.maximum(magL, 1e-9).astype(np.float64) /
                              np.maximum(magR, 1e-9).astype(np.float64))

        def dev(ild_):
            return 2.0 * np.degrees(np.arctan(10.0 ** (-ild_ / 20.0))) - 90.0

        def dev_to_ild(d_):
            return 20.0 * np.log10(np.tan((90.0 - d_) * np.pi / 360.0))

        d0 = dev(ild)
        d_rot = np.clip(d0 + rotate, -89.9, 89.9)
        ild_rot = dev_to_ild(d_rot)

        if stage_deg is None:
            gamma = 1.0
        else:
            def stage_of(g):
                return float(np.percentile(np.abs(dev(g * ild_rot[m])), 95))

            target = stage_deg / 2.0
            lo, hi = 1e-3, 100.0
            if stage_of(hi) < target:
                gamma = hi
            elif stage_of(lo) > target:
                gamma = lo
            else:
                for _ in range(45):
                    mid = 0.5 * (lo + hi)
                    if stage_of(mid) < target:
                        lo = mid
                    else:
                        hi = mid
                gamma = 0.5 * (lo + hi)

        ild_final = gamma * ild_rot
        rp = 10.0 ** (ild_final / 20.0)
        eps = 1e-12
        A = np.hypot(magL.astype(np.float64), magR.astype(np.float64))
        tL = A * rp / np.sqrt(1.0 + rp * rp)
        tR = A / np.sqrt(1.0 + rp * rp)
        gL = np.zeros_like(A)
        gR = np.zeros_like(A)
        mskL = magL > eps
        mskR = magR > eps
        gL[mskL] = tL[mskL] / magL[mskL]
        gR[mskR] = tR[mskR] / magR[mskR]
        gdist = np.ones(magL.shape, dtype=np.float32)
        if depth > 0.0:
            d0f = d0.astype(np.float32)
            d1 = dev(ild_final).astype(np.float32)
            r0 = np.sqrt(depth ** 2 + (d0f / 90.0) ** 2)
            r1 = np.sqrt(depth ** 2 + (d1 / 90.0) ** 2)
            gdist = (r0 / r1).astype(np.float32)
        gL = (1.0 - blend) + blend * gL * gdist
        gR = (1.0 - blend) + blend * gR * gdist
        outL = self.istft_unpad(np.asarray(XL * gL, dtype=np.complex64), n, pad)
        outR = self.istft_unpad(np.asarray(XR * gR, dtype=np.complex64), n, pad)
        in_rms = float(np.sqrt(np.mean(Ls.astype(np.float64) ** 2) +
                               np.mean(Rs.astype(np.float64) ** 2)))
        out_rms = float(np.sqrt(np.mean(outL.astype(np.float64) ** 2) +
                                np.mean(outR.astype(np.float64) ** 2)))
        if out_rms > 1e-12:
            g = in_rms / out_rms
            outL = (outL * g).astype(np.float32)
            outR = (outR * g).astype(np.float32)
        if out is not None:
            save_stereo(out, outL, outR, sr)
        achieved = float(np.percentile(np.abs(dev(ild_final[m])), 95))
        return outL, outR, sr, gamma, achieved


    def _seg(self, path, t0=0.0, t1=None):
        """Per-bin quantities for a segment: deviation d, active mask
        (amp > 5% of max, same gate as the warp), ILD, per-channel RMS."""
        L, R, sr = load_stereo(path)
        i0 = max(0, min(int(t0 * sr), len(L)))
        i1 = len(L) if t1 is None else max(i0, min(int(t1 * sr), len(L)))
        Ls, Rs = L[i0:i1], R[i0:i1]
        XL, _ = self.stft_padded(Ls)
        XR, _ = self.stft_padded(Rs)
        magL, magR = np.abs(XL), np.abs(XR)
        amp = magL + magR
        m = amp > 0.05 * amp.max()
        ild = 20.0 * np.log10(np.maximum(magL, 1e-9).astype(np.float64) /
                              np.maximum(magR, 1e-9).astype(np.float64))
        d = 2.0 * np.degrees(np.arctan(10.0 ** (-ild / 20.0))) - 90.0
        return dict(sr=sr, t0=i0 / sr, t1=i1 / sr, n=len(Ls), Ls=Ls, Rs=Rs,
                    rmsL=float(np.sqrt(np.mean(Ls.astype(np.float64) ** 2))),
                    rmsR=float(np.sqrt(np.mean(Rs.astype(np.float64) ** 2))),
                    d=d, m=m, ild=ild, amp=amp)

    def measure(self, path, t0=0.0, t1=None):
        """Per-bin spatial statistics of a segment: no NMF, no modification.
        Active bins: amplitude > 5% of the segment max (same gate as the
        warp). Returns a stats dict to display before choosing
        --rotate/--stage parameters."""
        s = self._seg(path, t0, t1)
        dd = s["d"][s["m"]]
        ai = np.abs(s["ild"][s["m"]])
        hist, _ = np.histogram(dd, bins=18, range=(-90.0, 90.0))
        return dict(sr=s["sr"], t0=s["t0"], t1=s["t1"], n=s["n"],
                    rmsL=s["rmsL"], rmsR=s["rmsR"],
                    rms=float(np.hypot(s["rmsL"], s["rmsR"])),
                    peak=float(max(np.max(np.abs(s["Ls"])),
                                   np.max(np.abs(s["Rs"])))),
                    n_active=int(s["m"].sum()),
                    med=float(np.median(dd)), mean=float(np.mean(dd)),
                    p5=float(np.percentile(dd, 5)),
                    p95=float(np.percentile(dd, 95)),
                    p95abs=float(np.percentile(np.abs(dd), 95)),
                    ild_med=float(np.median(s["ild"][s["m"]])),
                    ild_p50=float(np.percentile(ai, 50)),
                    ild_p95=float(np.percentile(ai, 95)),
                    ild_p99=float(np.percentile(ai, 99)),
                    hist=hist)

    def verify(self, path_a, path_b, t0a=0.0, t1a=None, t0b=0.0, t1b=None):
        """Before/after spatial check of two files: balance, position
        statistics, and the per-bin position shift over the common active
        set. Segments must align (same sr and sample count)."""
        A = self._seg(path_a, t0a, t1a)
        B = self._seg(path_b, t0b, t1b)
        if A["sr"] != B["sr"]:
            raise ValueError(f"sample rates differ: {A['sr']} vs {B['sr']}")
        if A["n"] != B["n"]:
            raise ValueError(f"segment lengths differ: {A['n']} vs {B['n']} "
                             f"samples")
        common = A["m"] & B["m"]
        sh = B["d"][common] - A["d"][common]
        med = float(np.median(sh))
        return dict(A=A, B=B, common=int(common.sum()),
                    n_active_a=int(A["m"].sum()), n_active_b=int(B["m"].sum()),
                    sh_med=med, sh_p5=float(np.percentile(sh, 5)),
                    sh_p95=float(np.percentile(sh, 95)),
                    sh_scatter=float(np.percentile(np.abs(sh - med), 95)))


def report(comps):
    print(f"{'k':>3}  {'ild_db':>8}  {'dev_deg':>8}  {'centroid_hz':>12}  {'act_frac':>9}")
    for c in comps:
        print(f"{c['k']:>3}  {c['ild_db']:>+8.2f}  {c['dev_deg']:>+8.1f}  "
              f"{c['centroid_hz']:>12.1f}  {c['act_frac']:>9.3f}")


def measure_print(st):
    rng = st["p95"] - st["p5"]
    print(f"segment {st['t0']:g}-{st['t1']:g} s  sr={st['sr']}  "
          f"RMS L {st['rmsL']:.4f} / R {st['rmsR']:.4f}  peak {st['peak']:.3f}")
    print(f"active bins {st['n_active']} (amp > 5% of max)")
    print(f"position d (deg from center): median {st['med']:+6.1f}  "
          f"mean {st['mean']:+6.1f}  p5 {st['p5']:+6.1f}  p95 {st['p95']:+6.1f}  "
          f"range {rng:5.1f}  p95|d| {st['p95abs']:5.1f}")
    print(f"ILD (dB, L/R): median {st['ild_med']:+6.2f}  "
          f"|ILD| p50 {st['ild_p50']:5.2f}  p95 {st['ild_p95']:5.2f}  "
          f"p99 {st['ild_p99']:5.2f}")
    mx = int(st["hist"].max())
    print("position histogram (10 deg cells):")
    for i, c in enumerate(st["hist"]):
        lo = -90 + 10 * i
        bar = "#" * int(round(50 * c / mx)) if mx else ""
        print(f"  {lo:+4d}..{lo + 10:+4d}  {bar} {c}")
    if abs(st["med"]) >= 2.0:
        side = "right" if st["med"] > 0 else "left"
        print(f"suggestion: {abs(st['med']):.0f} deg {side} bias -> "
              f"remix --rotate {-st['med']:.0f} to recenter")
    keep = 2.0 * st["p95abs"]
    print(f"suggestion: width ~{rng:.0f} deg (p5..p95), p95|d| {st['p95abs']:.0f} "
          f"-> --stage {keep:.0f} keeps it, --stage 180 expands to full")


def verify_print(v):
    A, B = v["A"], v["B"]
    ba = 20.0 * np.log10(A["rmsL"] / A["rmsR"])
    bb = 20.0 * np.log10(B["rmsL"] / B["rmsR"])
    ra = float(np.hypot(A["rmsL"], A["rmsR"]))
    rb = float(np.hypot(B["rmsL"], B["rmsR"]))
    da, db = A["d"][A["m"]], B["d"][B["m"]]
    pa5, pa95 = np.percentile(da, 5), np.percentile(da, 95)
    pb5, pb95 = np.percentile(db, 5), np.percentile(db, 95)
    print(f"segment {A['t0']:g}-{A['t1']:g} s  {A['n']} samples  sr {A['sr']}")
    print(f"balance L/R: {ba:+6.2f} -> {bb:+6.2f} dB  (delta {bb - ba:+5.2f})  "
          f"combined RMS {ra:.4f} -> {rb:.4f} ({rb / ra:.3f})")
    print(f"active bins: {v['n_active_a']} -> {v['n_active_b']}  "
          f"(common {v['common']})")
    print(f"position d: median {np.median(da):+6.1f} -> {np.median(db):+6.1f}  "
          f"(delta {np.median(db) - np.median(da):+5.1f})   "
          f"p5 {pa5:+6.1f} -> {pb5:+6.1f}   p95 {pa95:+6.1f} -> {pb95:+6.1f}")
    print(f"             range {pa95 - pa5:5.1f} -> {pb95 - pb5:5.1f}   "
          f"p95|d| {np.percentile(np.abs(da), 95):5.1f} -> "
          f"{np.percentile(np.abs(db), 95):5.1f}")
    print(f"ILD median: {np.median(A['ild'][A['m']]):+6.2f} -> "
          f"{np.median(B['ild'][B['m']]):+6.2f} dB")
    print(f"per-bin shift (common set): median {v['sh_med']:+6.1f}  "
          f"p5 {v['sh_p5']:+5.1f}  p95 {v['sh_p95']:+5.1f}  "
          f"scatter p95 {v['sh_scatter']:4.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="respatial",
        description="detect spectral activities in a stereo recording and reposition them")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze",
                       help="decompose, export per-activity clips + pans template")
    a.add_argument("wav")
    a.add_argument("--rank", type=int, default=8)
    a.add_argument("--iters", type=int, default=300)
    a.add_argument("--t0", type=float, default=0.0, help="segment start (s)")
    a.add_argument("--t1", type=float, default=None, help="segment end (s)")
    a.add_argument("--outdir", default="report")
    a.add_argument("--clip-dur", type=float, default=3.0)

    q = sub.add_parser("measure",
                       help="print per-bin spatial statistics (no modification)")
    q.add_argument("wav")
    q.add_argument("--t0", type=float, default=0.0, help="segment start (s)")
    q.add_argument("--t1", type=float, default=None, help="segment end (s)")

    v = sub.add_parser("verify",
                       help="before/after spatial check of two files")
    v.add_argument("original")
    v.add_argument("modified")
    v.add_argument("--t0", type=float, default=0.0,
                   help="segment start (s), applied to both files")
    v.add_argument("--t1", type=float, default=None,
                   help="segment end (s), applied to both files")

    m = sub.add_parser("remix", help="re-render activities at target pans")
    m.add_argument("wav")
    m.add_argument("--out", required=True)
    g = m.add_mutually_exclusive_group()
    g.add_argument("--pans",
                   help="pans file (from analyze) or 'k=deg,k=deg' (deg from center)")
    g.add_argument("--stage", type=float,
                   help="expand/contract: per-bin ILD warp so content spans this "
                        "stage width (deg, <=180)")
    m.add_argument("--rotate", type=float, default=0.0,
                   help="shift all content by this many degrees (positive = "
                        "right); alone or with --stage; per-bin distance gain "
                        "applied")
    m.add_argument("--blend", type=float, default=1.0,
                   help="warp amount with --stage/--rotate: 0=original, 1=full")
    m.add_argument("--depth", type=float, default=2.0,
                   help="listener distance in half-stage widths; inverse-distance "
                        "gain for relocated content (per-bin with --stage, per "
                        "component with --pans; 0=off, 2≈+1 dB center vs edge)")
    m.add_argument("--rank", type=int, default=8)
    m.add_argument("--iters", type=int, default=300)
    m.add_argument("--t0", type=float, default=0.0, help="segment start (s)")
    m.add_argument("--t1", type=float, default=None, help="segment end (s)")
    m.add_argument("--hard", action="store_true",
                   help="competitive: pan each bin to its dominant component (no averaging)")
    m.add_argument("--check", action="store_true",
                   help="print a before/after spatial check of the written output")

    args = ap.parse_args(argv)
    r = Respatial()
    if args.cmd == "analyze":
        comps, sr = r.analyze(args.wav, rank=args.rank, iters=args.iters,
                              t0=args.t0, t1=args.t1,
                              outdir=args.outdir, clip_dur=args.clip_dur)
        report(comps)
        print(f"\nclips + pans.txt in {args.outdir}/")
    elif args.cmd == "measure":
        measure_print(r.measure(args.wav, t0=args.t0, t1=args.t1))
    elif args.cmd == "verify":
        verify_print(r.verify(args.original, args.modified,
                              t0a=args.t0, t1a=args.t1))
    else:
        if args.pans is not None and args.rotate:
            ap.error("--rotate cannot be combined with --pans")
        if args.pans is None and args.stage is None and not args.rotate:
            ap.error("one of --pans, --stage, or --rotate is required")
        dgain_note = ""
        if args.depth > 0.0:
            cg = 20.0 * np.log10(np.sqrt(args.depth ** 2 + 1.0) / args.depth)
            dgain_note = f", dist-gain +{cg:.2f} dB (center vs edge)"
        if args.pans is not None:
            pans = parse_pans(args.pans)
            r.remix(args.wav, pans, rank=args.rank, iters=args.iters,
                    out=args.out, t0=args.t0, t1=args.t1, hard=args.hard,
                    depth=args.depth)
            print(f"wrote {args.out}{dgain_note}")
        else:
            if args.hard:
                print("note: --hard ignored with --stage/--rotate (per-bin)")
            outL, outR, sr, gamma, achieved = r.warp_stage(
                args.wav, args.stage, blend=args.blend, depth=args.depth,
                rotate=args.rotate, t0=args.t0, t1=args.t1, out=args.out)
            what = (f"stage {args.stage:g} deg" if args.stage is not None
                    else f"rotate {args.rotate:+g} deg")
            print(f"{what}: gamma={gamma:.3f}, span {2 * achieved:.1f} deg "
                  f"(p95|d|), blend {args.blend:g}{dgain_note}, wrote {args.out}")
        if args.check:
            print()
            verify_print(r.verify(args.wav, args.out,
                                  t0a=args.t0, t1a=args.t1))


if __name__ == "__main__":
    main()
