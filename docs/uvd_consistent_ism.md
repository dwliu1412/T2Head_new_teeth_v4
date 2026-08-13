# UVD-consistent ISM

The first refinement phase exposes one ablation switch:

```text
--guidance-mode ism | uvd-sfd
```

`ism` is the original AnimPortrait3D null-prompt DDIM-inversion objective.
`uvd-sfd` is retained as the historical experiment/output label, but now
means **UVD-consistent ISM**. Both modes use the same timestep schedule,
ControlNet/UNet predictions, CFG scale, ISM weight, regional weights,
optimizer, and densification. The only algorithmic difference is the Gaussian
noise supplied to the same ISM function.

## Objective

For both modes the gradient is

```text
sqrt((1 - alpha_t) / alpha_t) * (epsilon_cfg(x_t, text) - epsilon_inv)
```

where `epsilon_inv` is obtained by the AnimPortrait3D null-prompt DDIM
inversion. The removed implementation instead used
`epsilon_cfg - epsilon_uvd`; that was a probability-flow update rather than
ISM and is no longer present.

Raw ISM draws ordinary iid noise. UVD-consistent ISM draws a fresh semantic
`(layer, u, v, d)` Gaussian volume once per optimizer step. Full-image,
regional-crop, multi-view, and gradient-accumulation calls in that optimizer
step reuse the volume, so observations of the same animated surface share a
noise component. The next optimizer step receives a fresh draw, preserving
the Monte-Carlo nature of ISM.

Each latent footprint deduplicates its canonical cells and combines them with
`1/sqrt(N)` normalization. Under-resolved footprints blend the canonical
component with independent fallback noise using square-root weights. This
keeps unit marginal variance. Correspondence reliability affects only this
coupling; it never multiplies or clips the final ISM gradient.

The UVD mode deliberately has no private CFG scale, timestep schedule,
gradient clipping, learning-rate multiplier, color projection, reference
penalty, topology freeze, or SDEdit optimizer reset. This makes the guidance
ablation attributable to canonical noise coupling.

UVD mode requires `ism_variant: animportrait3d`; a custom config cannot
silently fall back to the older interval objective. Checkpoints store both the
guidance-method signature and the canonical noise/private-RNG state. Resuming
with the other guidance mode, changed atlas settings, or a missing UVD noise
state is rejected.

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
python train_mouth.py --reconstruction <stage1_dir> --guidance-mode uvd-sfd --sdedit-mode independent --max-steps 10 --output outputs/smoke/mouth_uvd_ism --gpu 0
```

Production mouth and full UVD-consistent runs:

```powershell
python train_mouth.py --reconstruction <stage1_dir> --guidance-mode uvd-sfd --sdedit-mode flame-surface --surface-views 4 --gpu 0
python train_full.py --reconstruction <stage1_dir> --guidance-mode uvd-sfd --sdedit-mode flame-surface --surface-views 4 --prompt "<identity-specific prompt>" --gpu 0
```

Use a fresh output directory for every run. Old probability-flow UVD-SFD
checkpoints are intentionally rejected; restart from their verified PLY if it
is still useful as an initialization.
