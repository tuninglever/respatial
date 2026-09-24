# AGENTS.md

## Project

`respatial` is a research prototype for enhancing stereo recordings of
multiphonic music (e.g. an orchestra). It detects the spectral activities
present in a recording — no target instrument a priori — presents them to a
user, and lets the user reposition each across the left/right field.
Motivating case: a recording whose instruments are "crushed" to the center,
which the transformation moves apart.

Concept notes (read first): `docs/multiphonic-remix-concepts.pdf`
(LaTeX source alongside; regenerate with `pdflatex`).

## Core model

The same instrument appears in both channels with varied level but nearly
identical spectrum, so in the STFT domain:

    L(t,f) ≈ Σᵢ aᵢ(t)·sᵢ(f)·e^{jφᵢᴸ},   R(t,f) ≈ Σᵢ bᵢ(t)·sᵢ(f)·e^{jφᵢᴿ}

Consequences the pipeline relies on:

1. Timbre is channel-invariant → cluster in one shared timbre space.
2. Per-bin L/R ratio is a spatial sample: where one activity dominates,
   |L/R| ≈ its ILD and ∠(L/R) ≈ its ICPD.

## Pipeline (v1)

1. **Detection**: STFT both channels → NMF of magnitudes with a shared
   timbre matrix D and per-channel activations (|L| ≈ D·A_L, |R| ≈ D·A_R)
   → per-activity spatial fingerprint (median ILD/ICPD)
   → user names activities and assigns target pans.
2. **Rendering**: soft mask per activity (masks partition each bin) → mono
   stem from the mid → equal-power pan (p ∈ [-1,1]: φ = π/4·(1+p),
   g_L = cos φ, g_R = sin φ) → sum; residual (1−Σmask)·mid stays center.
   `--hard` variant: each bin is panned to its dominant component's target
   (no per-bin averaging; wider stage, more position jumping). Each
   component is scaled by the inverse-distance gain r(d₀)/r(d₁) between its
   detected deviation d₀ and target d₁ (`--depth`, default 2 ≈ +1 dB center
   vs edge; 0=off) — level follows position, as on a physical stage.
3. **Expand/contract/rotate (warp, `--stage`/`--rotate`)**: no NMF. Each
   bin's position is first rigidly shifted by `--rotate θ` (d_rot =
   clip(d+θ, ±89.9°), the "rotate the stereo mic" model; positive θ = right),
   then the ILD is power-warped (ild_final = γ·ild_rot; ild_rot = the ILD of
   d_rot, inverse of the ILD→d map, r(d) = tan((90°−d)/2)). Unified
   energy-preserving gain |L′| = A·r′/√(1+r′²), r′ = 10^(ild_final/20)
   (|L′|²+|R′|² = |L|²+|R|²); the warp is the r′ = r^γ special case.
   Phase/ICPD untouched (real non-negative per-bin gains). γ is solved by
   bisection so p95(|d_final|) = stage/2 over active bins; γ = 1 when
   `--stage` is absent (pure rotation). Rotate applies BEFORE the warp
   (corrects the capture bias, then expands). `--blend α` crossfades
   original→final (pure amplitude interpolation). `--depth` adds the same
   inverse-distance gain per bin (r(d₀)/r(d_final) between the bin's original
   and final deviations). Output RMS-matched to input.
   Edge handling (all per-bin-modified iSTFTs: warp, stems, residual): the
   STFT is computed on the segment zero-padded by one frame per side
   (`stft_padded`/`istft_unpad`), because iSTFT divides each sample by the
   Σwin² of covering frames, which → 0 at segment edges — modified frames
   then blow up there.

Spatial convention: deviation from center d ∈ [−90°, +90°] (each channel
spans 90°; total stage up to 180°), p = d/90°. Measured ILD → d:
d = 2·atan(1/r) − 90°, r = 10^(ild_db/20) (+ILD = L louder = left = −d).

Key decision: re-render stems (mono → panned) rather than tweaking
per-channel gains; centered sources have little spatial detail to preserve.

## Layout

- `docs/` — `multiphonic-remix-concepts.pdf` (concept notes),
  `respatial-design-notes.pdf` (project overview, maths, design
  considerations), `respatial-rotation-notes.pdf` (rotation mode),
  `respatial-prior-art.pdf` (related-tools survey, Sep 20; all regenerate
  with `pdflatex`, sources alongside).
- `src/` — C99 kernels (own the compute); `respatial.h` is the flat,
  ctypes-friendly API:
  - `fft.c` — radix-2 FFT
  - `stft.c` — Hann-windowed STFT/iSTFT (re/im interleaved)
  - `nmf.c` — NMF with shared D, per-channel activations (fixed-seed LCG)
  - `spatial.c` — energy-gated median-ILD / circular-mean-ICPD fingerprint
  - `render.c` — soft stem masks
  - `wav.c` — 16/24-bit PCM / 32-bit float read, 16-bit PCM write
- `Makefile` — builds `librespatial.dylib`; `make test` runs the pipeline test.
- `respatial.py` — ctypes wrapper + pipeline orchestration + CLI:
  - `analyze <wav> [--t0 s --t1 s] [--outdir d] [--clip-dur s]` — decompose,
    export a mono clip per activity (most active window, peak-normalized)
    and a `pans.txt` template (initial pan = detected deviation).
  - `remix <wav> --out <wav> (--pans <file|k=deg,...> | --stage <deg> |
    --rotate <deg>) [--rotate <deg>] [--blend α] [--hard] [--depth d]` —
    re-render at target pans, per-bin ILD warp to a stage of that total
    width, and/or a rigid position shift (`--rotate` deg, positive = right;
    alone or with `--stage`, where rotate applies first); `--blend` = warp
    amount 0..1; `--depth` = inverse-distance gain (default 2, 0=off).
  - `measure <wav> [--t0 s --t1 s]` — print per-bin spatial statistics
    (position median/mean/p5/p95/range, ILD bias + |ILD| contrast, 10°
    histogram) with rotate/stage suggestions; no NMF, no modification.
  - `verify <orig> <mod> [--t0 s --t1 s]` — before/after spatial check of
    two files (balance L/R, position stats, per-bin shift on the common
    active set); `remix --check` prints it for the written output.
- `test_pipeline.py` — synthetic end-to-end check: two harmonic sources with
  known L/R gains; fingerprints must match ground truth and remix must move
  each source toward its target pan.

## Environment

- Python 3.14, `numpy` system-wide (only dependency; wav I/O lives in C).
- Build: `make` (produces `librespatial.dylib` next to `respatial.py`).
- Test: `make test` or `python3 test_pipeline.py` (plain asserts; no pytest).
- `test_pipeline.py` loads `respatial.py` by file path
  (`importlib.util.spec_from_file_location`) rather than `import respatial`:
  name-based import fails intermittently on this external volume (stale
  directory-entry cache) while direct path lookup works. Don't "fix" it back.

## Conventions

- Prototype stage: readable > clever; vectorize with numpy where easy.
- No code comments unless a formula or non-obvious reason needs stating.
- Audio arrays: shape (channels, samples), float32.
- Keep the concept doc in sync when a design decision changes.

## Status

v1 pipeline + CLI implemented; the synthetic end-to-end test passes. Three
relocation modes are available — per-activity re-rendering (`--pans`), the
per-bin ILD warp (`--stage`), and the rigid position shift (`--rotate`) —
plus `measure` and `verify` for before/after spatial checks. Design
rationale, maths, and listening results live in the `docs/` PDFs.
