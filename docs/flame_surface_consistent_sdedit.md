# FLAME surface-consistent SDEdit

This repository keeps the original AnimPortrait3D SDEdit teacher as the
default and adds an opt-in multi-view branch for high-frequency appearance
refinement.  The branch is selected independently from the first-phase
`ism|uvd-sfd` ablation:

```text
first phase:  --guidance-mode ism | uvd-sfd
SDEdit phase: --sdedit-mode independent | flame-surface
```

## Why the branch is surface-addressed

Camera-neighbour K/V sharing does not establish that two image tokens describe
the same point of the avatar.  The new path renders, for every selected view,
one unambiguous FLAME correspondence per pixel and assigns the key

```text
(classifier-free-guidance branch, semantic layer, canonical UV texel)
```

The five semantic memories are fixed and mutually exclusive:

1. face/non-oral surface;
2. lips;
3. upper teeth;
4. lower teeth;
5. oral cavity.

Upper and lower teeth intentionally overlap in numeric UV space, so the layer
ID is a required part of the key.  A query never searches memory belonging to
another layer or texel.

For each U-Net self-attention layer, visible tokens are first reduced to one
K/V slot per `(view, surface key)`.  A target query attends only to observation
slots carrying that same key; `exclude_self` controls whether its own view's
slot participates.  Positive and negative CFG batches use separate memories.
The original within-view self-attention output is retained and the surface
result is blended in during the late denoising steps; tokens without enough
matching views fall back exactly to the original output. Text cross-attention
is unchanged.

## Visibility and depth gate

Correspondence is produced from the same frozen first-phase Gaussian snapshot
that supplies the SDEdit source image, using the same live pose, camera, and
crop as the RGB/control inputs. The snapshot includes the discrete FLAME
`face_idx` binding as well as UVD, covariance, opacity, and colour, so later
UV face rebinding cannot move either its geometry or semantic layer. It never
comes from the gradually changing trainable avatar. For each layer the renderer records canonical UV moments,
alpha, expected depth, and its front-to-back contribution while the full scene
remains present. A token is admitted only
when all of the following hold:

- UV is finite and inside the canonical atlas;
- the semantic layer is valid;
- alpha and full-scene contribution pass their thresholds;
- UV variance is below the seam/mixture threshold;
- expected depth is within the configured tolerance of the nearest candidate;
- the winning layer is sufficiently dominant;
- at least `min_views` distinct views observe the same surface key.

UV, layer, and depth use the same nearest-neighbour crop/resample. Visibility
is smoothly resized and then multiplied by the categorical validity gate.

## Running it

Mouth refinement:

```powershell
python train_mouth.py `
  --guidance-mode ism `
  --sdedit-mode flame-surface `
  --surface-views 4
```

Full refinement (using the matching mouth output by default):

```powershell
python train_full.py `
  --prompt "<identity-specific full prompt>" `
  --guidance-mode ism `
  --sdedit-mode flame-surface `
  --surface-views 4
```

The default output names are `mouth_surface_sdedit` and
`full_surface_sdedit`; CFD-consistent UVD-SFD combinations additionally
contain the `uvd_sfd` suffix.  This prevents an experimental result from overwriting or
being mistaken for the independent baseline.

The CLI raises `data.batch_size` only for the surface branch. Its sampler first
draws row zero with the exact legacy `B=1` camera distribution, then adds
evenly spaced companions from the same calibrated elevation ring. Before the
SDEdit boundary the system keeps only that anchor, preserving both the
single-view objective and its camera distribution. At the boundary it consumes
the complete same-pose multi-view group in one joint denoising trajectory.

## Configuration

Both reconstruction configs contain the disabled-by-default block
`system.sdedit.surface_memory`.  The main controls are:

- `views`: joint camera count;
- `atlas_resolution`: UV texel quantization used as the hard address;
- `min_views` and `max_memory_views`: distinct source-view gates;
- `strength`, `start_progress`, `end_progress`: late-denoise blend schedule;
- `processor_patterns`: self-attention layers to wrap (the default targets the
  last two U-Net up blocks);
- `alpha_threshold`, `contribution_threshold`, `variance_threshold`,
  `depth_tolerance`, and `dominance_ratio`: renderer-side correspondence gate.

Setting `mode: independent` and `surface_memory.enabled: false` installs no
attention processor, renders no correspondence buffers, and keeps the legacy
batch size and denoising call path.

Checkpoints lock the SDEdit method after its first optimizer update. An exact
checkpoint at the untouched ISM/SDEdit boundary may still be resumed twice,
once with each SDEdit mode, so both ablations can share an identical
first-phase avatar. Switching modes after any SDEdit update is rejected.

## Current scope

One training batch shares a FLAME expression/jaw pose and varies calibrated
cameras.  Because the address is canonical, the attention implementation can
also consume observations from different expressions; enabling that extension
requires both a multi-pose sampler and per-row FLAME geometry/render support.
The current batch builder and system intentionally use one shared expression
and jaw pose for every camera, so cross-expression memory is not enabled by
this branch. The current `face` layer is the non-oral FLAME-bound complement,
not a skin-only segmentation.
