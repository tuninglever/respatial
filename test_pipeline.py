#!/usr/bin/env python3
"""End-to-end synthetic test for the respatial pipeline.

Builds a 2-source stereo mix with known channel gains, checks that
analyze recovers the two spatial fingerprints, and that remix moves each
source toward its requested pan.
"""
import importlib.util
import os
import tempfile
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("respatial", os.path.join(HERE, "respatial.py"))
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)
SR = 44100
DUR = 3.0
FRAME, HOP, RANK, ITERS = 4096, 1024, 2, 300
TOL = 3.0


def source(f0, n_harm, rolloff, t0, t1, n, sr):
    t = np.arange(n) / sr
    x = np.zeros(n)
    for h in range(1, n_harm + 1):
        x += (1.0 / h ** rolloff) * np.sin(2.0 * np.pi * f0 * h * t)
    fade = int(0.01 * sr)
    env = np.ones(n)
    i0, i1 = int(t0 * sr), int(t1 * sr)
    env[:i0] = 0.0
    env[i1:] = 0.0
    w = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade) / fade))
    env[i0:i0 + fade] *= w
    env[max(i1 - fade, 0):i1] *= w[::-1]
    return x * env


def write_wav(path, L, R):
    inter = np.clip(np.column_stack([L, R]), -1.0, 1.0) * 32767.0
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(inter.astype("<i2").tobytes())


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 2 and w.getsampwidth() == 2
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    raw = raw.astype(np.float32) / 32768.0
    return raw[0::2], raw[1::2]


def band_ild(L, R, t0, t1, f0, f1):
    i0, i1 = int(t0 * SR), int(t1 * SR)
    seg = np.hanning(i1 - i0)
    sl = np.abs(np.fft.rfft(L[i0:i1] * seg)) ** 2
    sr_ = np.abs(np.fft.rfft(R[i0:i1] * seg)) ** 2
    fr = np.fft.rfftfreq(i1 - i0, 1.0 / SR)
    m = (fr >= f0) & (fr <= f1)
    return 20.0 * np.log10(np.sqrt(np.sum(sl[m])) / np.sqrt(np.sum(sr_[m])))


def main():
    n = int(SR * DUR)
    A = source(220.0, 6, 1.0, 0.0, 2.0, n, SR)
    B = source(500.0, 4, 2.0, 1.0, 3.0, n, SR)
    A /= np.max(np.abs(A))
    B /= np.max(np.abs(B))
    gA = (0.90, 0.35)
    gB = (0.35, 0.90)
    L = gA[0] * A + gB[0] * B
    R = gA[1] * A + gB[1] * B
    peak = max(float(np.max(np.abs(L))), float(np.max(np.abs(R))))
    L, R = 0.5 * L / peak, 0.5 * R / peak

    d = tempfile.mkdtemp(prefix="respatial_test_")
    inp = os.path.join(d, "in.wav")
    outp = os.path.join(d, "out.wav")
    write_wav(inp, L, R)

    r = rs.Respatial(FRAME, HOP)
    comps, sr = r.analyze(inp, rank=RANK, iters=ITERS)
    rs.report(comps)
    assert sr == SR
    ilds = [c["ild_db"] for c in comps]
    truth = 20.0 * np.log10(gA[0] / gA[1])
    assert abs(ilds[0]) > 4.0 and abs(ilds[1]) > 4.0, "components not clearly separated"
    kA = int(np.argmax(ilds))
    kB = int(np.argmin(ilds))
    assert kA != kB
    assert abs(ilds[kA] - truth) < TOL, f"left ILD {ilds[kA]:+.2f} vs truth {truth:+.2f}"
    assert abs(ilds[kB] + truth) < TOL, f"right ILD {ilds[kB]:+.2f} vs truth {-truth:+.2f}"
    print(f"fingerprint check OK (truth {truth:+.2f} / {-truth:+.2f} dB)")

    r.remix(inp, {kA: -0.9, kB: 0.9}, rank=RANK, iters=ITERS, out=outp)
    oL, oR = read_wav(outp)
    assert len(oL) == n

    ildA_in = band_ild(L, R, 0.2, 0.8, 180, 260)
    ildA_out = band_ild(oL, oR, 0.2, 0.8, 180, 260)
    ildB_in = band_ild(L, R, 2.2, 2.8, 450, 550)
    ildB_out = band_ild(oL, oR, 2.2, 2.8, 450, 550)
    print(f"A band ILD: in {ildA_in:+.2f} -> out {ildA_out:+.2f} dB (target ~+22)")
    print(f"B band ILD: in {ildB_in:+.2f} -> out {ildB_out:+.2f} dB (target ~-22)")
    assert ildA_out > ildA_in + 8.0, "source A did not move left"
    assert ildB_out < ildB_in - 8.0, "source B did not move right"

    e_in = float(np.sum(L.astype(np.float64) ** 2) + np.sum(R.astype(np.float64) ** 2))
    e_out = float(np.sum(oL.astype(np.float64) ** 2) + np.sum(oR.astype(np.float64) ** 2))
    print(f"energy ratio out/in: {e_out / e_in:.3f}")
    assert 0.2 < e_out / e_in < 5.0, "energy not roughly preserved"

    print("PASS")


if __name__ == "__main__":
    main()
