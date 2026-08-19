# UVD-SFD: CFD-consistent noise + ISM-style score difference

The first refinement phase exposes one ablation switch:

```text
--guidance-mode ism | uvd-sfd
```

`ism` is the original AnimPortrait3D null-prompt DDIM-inversion objective.
`uvd-sfd` is the CFD-consistent alternative. It keeps the same sampled
timestep and the same annealed interval schedule as the ISM ablation, but it
does not run DDIM inversion. Instead, the canonical UVD/CFD noise draw is used
directly at both endpoints of the interval.

## Objective

For UVD-SFD, one canonical noise tensor `xi` constructs both noisy latents:

```text
x_t = alpha_t * x + sigma_t * xi
x_s = alpha_s * x + sigma_s * xi

g = epsilon_cfg(x_t, t, text) - epsilon(x_s, s, null)
```

Here `scheduler.add_noise` supplies the scheduler's `alpha`/`sigma`
coefficients. The text endpoint uses the existing negative-prompt CFG rule;
the lower-noise endpoint uses the null embedding. No ISM SNR weight and no
model-predicted DDIM jump are applied to this UVD-SFD direction. The raw `ism`
mode remains unchanged and still uses its weighted null-prompt DDIM-inversion
target.

UVD-SFD draws a fresh semantic
`(layer, u, v, d)` Gaussian volume once per optimizer step. Full-image,
regional-crop, multi-view, and gradient-accumulation calls in that optimizer
step reuse the volume, so observations of the same animated surface share a
noise component. The next optimizer step receives a fresh draw, preserving
the Monte-Carlo nature of ISM.

Each latent footprint deduplicates its canonical cells and combines them with
`1/sqrt(N)` normalization. Under-resolved footprints blend the canonical
component with independent fallback noise using square-root weights. This
keeps unit marginal variance. Correspondence reliability affects only this
coupling; it never multiplies or clips the final UVD-SFD gradient.

The UVD mode deliberately has no private CFG scale, timestep schedule,
gradient clipping, learning-rate multiplier, color projection, reference
penalty, topology freeze, or SDEdit optimizer reset. Its two intentional
changes relative to raw ISM are the CFD-consistent endpoint construction and
the direct conditional-minus-null score difference.

UVD mode requires `ism_variant: animportrait3d` so it reuses the reference
interval schedule and prompt layout. Checkpoints store the score-difference
objective version as well as the canonical noise/private-RNG state. Resuming
an older UVD objective, the other guidance mode, changed atlas settings, or a
missing UVD noise state is rejected.

## Orthogonal SDEdit ablation

The second phase is selected independently:

```text
--sdedit-mode independent | flame-surface
```

`independent` preserves ordinary AnimPortrait3D SDEdit. `flame-surface`
jointly denoises matched views using canonical surface K/V memory. Its source
image and surface correspondence are both rendered from the frozen
first-phase snapshot. Checkpoints store and validate the SDEdit method
signature so a resumed run cannot silently switch ablations.

The two switches form four collision-free runs:

| First phase | SDEdit phase | Output suffix |
|---|---|---|
| `ism` | `independent` | `mouth` / `full` |
| `uvd-sfd` | `independent` | `*_uvd_sfd` |
| `ism` | `flame-surface` | `*_surface_sdedit` |
| `uvd-sfd` | `flame-surface` | `*_uvd_sfd_surface_sdedit` |

For a full-stage-only ablation, pass the exact same `--mouth-ply` and
`--mouth-params` to every full run; otherwise the default path intentionally
selects the matching mouth-stage combination and changes the initialization.

## Examples

First-phase smoke tests (no SDEdit):

```powershell
python train_mouth.py --reconstruction <stage1_dir> --guidance-mode ism --sdedit-mode independent --max-steps 10 --output outputs/smoke/mouth_ism --gpu 0
python train_mouth.py --reconstruction <stage1_dir> --guidance-mode uvd-sfd --sdedit-mode independent --max-steps 10 --output outputs/smoke/mouth_uvd_sfd --gpu 0
```

Production mouth and full UVD-SFD runs:

```powershell
python train_mouth.py --reconstruction <stage1_dir> --guidance-mode uvd-sfd --sdedit-mode flame-surface --surface-views 4 --gpu 0
python train_full.py --reconstruction <stage1_dir> --guidance-mode uvd-sfd --sdedit-mode flame-surface --surface-views 4 --prompt "<identity-specific prompt>" --gpu 0
```

Use a fresh output directory for every run. Checkpoints from the former
UVD-consistent-DDIM-inversion objective are intentionally rejected by the new
objective signature; restart from a verified PLY if it is still useful as an
initialization.
