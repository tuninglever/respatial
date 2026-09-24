# respatial

Enhance stereo recordings of multiphonic music (e.g. an orchestra): detect
the spectral activities present in a recording — no target instrument
a priori — present them to a user, and reposition them across the left/right
field. Motivating case: a recording whose instruments are "crushed" to the
center, which the transformation moves apart.

## Motivation

The project was created to experiment with shaping the soundstage, especially for orchestral recordings.  While listening to a concerto recording, I found that the soloist is great, but the orchestra parts seem to be crumbled up in the middle behind the soloist. It will be an interesting experiment to:

1. Use agent AI to create an app just by discussing and commenting on the methods and results.
The project used llama-server serving locally with unsloth's Qwen3.8-27B-GGUF model, with opencode on a 64G Mac-mini M4.
2. To create utilities to play with some simple soundstage experiences.

## How it works (short)

- **Detection** — STFT both channels → NMF of the magnitudes with a *shared*
  timbre matrix and per-channel activations (`|L| ≈ D·A_L`, `|R| ≈ D·A_R`)
  → per-activity spatial fingerprint (median ILD, circular-mean ICPD).
- **Re-rendering** — soft mask per activity (masks partition each bin) →
  mono stem from the mid → equal-power pan at the user's target → sum; the
  residual stays centered.
- **Expand/contract** — a per-bin ILD power warp (`ild′ = γ·ild`) with
  per-bin energy normalization; phase and ICPD are untouched. No NMF
  involved. A blend amount and an inverse-distance level gain are available.
- **Rotation** — a rigid per-bin shift of every position by θ
  (`d′ = clip(d + θ, ±90°)`, the "rotate the stereo mic" model); moves the
  whole scene without changing its spread. No NMF involved; composes with
  expand/contract (rotation applied first).

Full maths and design considerations: [`docs/respatial-design-notes.pdf`](docs/respatial-design-notes.pdf).
Rotation mode: [`docs/respatial-rotation-notes.pdf`](docs/respatial-rotation-notes.pdf).
Original concept notes: [`docs/multiphonic-remix-concepts.pdf`](docs/multiphonic-remix-concepts.pdf).

## Prior art (brief)

The individual concepts exist: stereo width/imaging plugins (Waves, iZotope
Ozone Imager, free LPanner), "rotation" plugins (LPanner, Noisebud 360 —
mostly M/S-vector or phase-quadrature mechanisms), and AI stem separation
(Demucs, Spleeter — fixed source types, no re-positioning). What does not
appear in existing free or commercial tools is the combination: measuring a
recording's per-bin spatial distribution, repositioning by a rigid
position-space shift or target-width ILD warp (phase-preserving,
energy-normalized), and numerically verifying the result against the
command's intent. Full comparison:
[`docs/respatial-prior-art.pdf`](docs/respatial-prior-art.pdf).

## Requirements

- C compiler (`cc`) and `make`
- Python 3 with `numpy` (the only dependency; wav I/O lives in C)

## Build

```sh
make            # builds librespatial.dylib (macOS)
make test       # runs the synthetic end-to-end test
```

On Linux: change `-dynamiclib` to `-shared` and the output name to
`librespatial.so` in the Makefile (and the library name in `respatial.py`).

## Usage

All commands run from the repository root.

### Typical workflow

1. **Look before you move.** `measure` prints per-bin spatial statistics
   (bias, width, histogram) with concrete `--rotate`/`--stage` suggestions;
   `analyze` goes further and decomposes the segment into named activities.
   Neither modifies the recording.
2. **Transform with `remix`.** Either re-render the analyzed activities at
   target pans (`--pans`), or apply a decomposition-free move: expand/contract
   the stage (`--stage`) and/or rotate the whole scene (`--rotate`). The
   latter two compose (rotation is applied first).
3. **Verify, then A/B.** `--check` (or `verify`) prints the before/after
   balance and per-bin position shift, so you can confirm numerically that
   the move did what the command said. Outputs are RMS-matched, so any level
   difference you hear is real.

```sh
# 1. what's there?
python3 respatial.py measure recording.wav --t0 16 --t1 36
python3 respatial.py analyze recording.wav --t0 16 --t1 36 --outdir report/seg

# 2. move it (pick one, or combine)
python3 respatial.py remix recording.wav --out out.wav \
    --pans report/seg/pans.txt --t0 16 --t1 36
python3 respatial.py remix recording.wav --out out.wav \
    --rotate -14 --stage 180 --blend 0.5 --t0 16 --t1 36
```

### 1. Measure (statistics, no modification)

Per-bin spatial statistics of a segment — no NMF, no output file. Use it to
decide whether the recording is biased (needs `--rotate`) and how wide it
already is (needs `--stage` to expand):

```sh
python3 respatial.py measure recording.wav --t0 16 --t1 36
```

Prints RMS/peak, the active-bin count, position statistics (median/mean/
p5/p95/range of `d`), ILD statistics, a 10° position histogram, and
suggestions, e.g.:

```
position d (deg from center): median  +14.4  ...  range 125.9  p95|d|  72.8
ILD (dB, L/R): median  -2.20  ...
suggestion: 14 deg right bias -> remix --rotate -14 to recenter
suggestion: width ~126 deg (p5..p95) -> --stage 126 keeps it, --stage 180 expands to full
```

### 2. Analyze a recording

Decompose a segment, export a mono clip per activity (its most active
window, peak-normalized) and a `pans.txt` template:

```sh
python3 respatial.py analyze recording.wav --t0 16 --t1 36 --outdir report/seg
```

The activity table printed looks like:

```
  k    ild_db   dev_deg   centroid_hz   act_frac
  0     +1.50      -9.8        1037.4      0.114
  1     +2.07     -13.5        2016.3      0.128
  ...
```

Listen to `report/seg/clip_00.wav` … `clip_07.wav`, name each activity, and
edit the last column of `report/seg/pans.txt` — the target deviation from
center in degrees (−90…+90; 0 = center).

### 3. Re-render at target pans

```sh
python3 respatial.py remix recording.wav --out out_pans.wav \
    --pans report/seg/pans.txt --t0 16 --t1 36
```

Pans can also be given inline (activity = degrees from center):

```sh
python3 respatial.py remix recording.wav --out out_pans.wav \
    --pans 0=-30,3=40 --t0 16 --t1 36
```

Useful options: `--hard` (pan each bin to its dominant component — wider
stage, more position jumping), `--rank N` / `--iters N` (NMF), `--depth d`
(inverse-distance gain, see below).

### 4. Expand / contract the stage (no decomposition)

```sh
# expand so the content spans a 180° stage
python3 respatial.py remix recording.wav --out out_stage180.wav \
    --stage 180 --t0 16 --t1 36

# half-strength expansion (effective ~150°)
python3 respatial.py remix recording.wav --out out_stage180_blend05.wav \
    --stage 180 --blend 0.5 --t0 16 --t1 36

# contract to a 120° stage
python3 respatial.py remix recording.wav --out out_stage120.wav \
    --stage 120 --t0 16 --t1 36
```

`--stage S` solves for the warp exponent γ so the content spans a stage of
total width S (degrees, ≤ 180); `--blend α` sets the amount (0 = original,
1 = full warp). The output is RMS-matched to the input, so A/B comparisons
are level-matched.

### 5. Rotate the scene (no decomposition)

Rigidly shift every position by θ degrees (positive = right) — the "rotate
the stereo mic" model. It moves the whole scene without changing its spread,
so it corrects a global left/right bias that `--stage` cannot:

```sh
# recenter a 14° right-biased recording
python3 respatial.py remix recording.wav --out out_rot.wav \
    --rotate -14 --t0 16 --t1 36

# recenter, then expand to a full stage at half strength
python3 respatial.py remix recording.wav --out out_rot_stage.wav \
    --rotate -14 --stage 180 --blend 0.5 --t0 16 --t1 36
```

With `--stage`, rotation is applied first (correct the bias, then expand
about the corrected center). `--blend` and the inverse-distance gain apply as
in the stage mode. See [`docs/respatial-rotation-notes.pdf`](docs/respatial-rotation-notes.pdf)
for the maths and the realized-vs-designed shift.

### 6. Verify (before/after check)

Numerically confirm that a transformation did what the command intended —
balance, position statistics, and the per-bin position shift over the common
active set:

```sh
python3 respatial.py verify original.wav modified.wav        # both full files
python3 respatial.py verify original_30_60.wav rotate14.wav  # matching segments
```

or automatically after writing, with `--check`:

```sh
python3 respatial.py remix recording.wav --out out.wav \
    --rotate -14 --t0 30 --t1 60 --check
```

Example (rotate −14 on a 14° right-biased segment):

```
balance L/R:  -1.45 ->  +0.03 dB  (delta +1.48)  combined RMS ... (1.000)
position d: median  +14.4 ->   +2.8  (delta -11.6)
             range 125.9 -> 125.6
per-bin shift (common set): median  -13.2  p5 -14.9  p95  -7.0  scatter p95  6.3
```

For a rotation the per-bin shift median should be ≈ θ (slightly less, and the
overall median less still — see the rotation notes on the realized-vs-designed
shift); for `--stage S` the after `p95|d|` should be ≈ S/2.

## Spatial convention

Deviation from center `d` in [−90°, +90°] (each channel spans 90°; total
stage up to 180°). Measured ILD → `d`: `d = 2·atan(1/r) − 90°` with
`r = 10^(ild_db/20)`; +ILD (left louder) = left = negative `d`.
Equal-power pan: `p = d/90`, `φ = π/4·(1+p)`, `g_L = cos φ`, `g_R = sin φ`.

## Inverse-distance gain (`--depth`)

On a physical stage a source at center is closer to a center listener than
one at the edge, so relocated content is level-adjusted by the
inverse-distance gain `r(d₀)/r(d₁)` between its original and target
deviation (per component with `--pans`, per bin with `--stage`/`--rotate`).
`--depth`
is the listener distance in half-stage widths: default 2 ≈ +1 dB center vs
edge; 0 disables it.

## Listening examples

(See examples.txt for the commands that generated these.)

Original YouTube music video:
https://www.youtube.com/watch?v=UzaKmsznTPM

In report/Mahler:

Original:
Mahler-7-2mvt.wav

Rotated:
Mahler-7-2mvt.rotate.6.wav

Expanded:
Mahler-7-2mvt.rotate.6.stage.180.blend.0.5.wav

Contracted:
Mahler-7-2mvt.rotate.6.stage.90.blend.0.5.wav

## Acknowledgements

This project was built with the help of:

- **opencode** (https://opencode.ai) — the AI coding agent used to develop
  the codebase.
- The **Qwen3.8-27B** model and its authors (the Qwen team, Alibaba Group) —
  the language model that powered the development sessions.
- The theoretical foundations:
  - D. D. Lee and H. S. Seung, "Algorithms for Non-negative Matrix
    Factorization," *Neural Computation* 13(9), 2001 — NMF.
  - J. M. Blauert, *Spatial Hearing: The Psychophysics of Human Sound
    Localization*, MIT Press, 1997 — ILD/ICPD as localization cues.
  - V. Pulkki, "Virtual sound source positioning using vector base amplitude
    panning," *J. Audio Eng. Soc.* 49(10), 2001 — the equal-power pan law
    used by the rendering stage.

## License

MIT — see [LICENSE](LICENSE).
