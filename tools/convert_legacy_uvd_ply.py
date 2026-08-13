"""Validate and rewrite a canonical UVD Gaussian PLY.

Legacy face-frame PLY conversion was removed with the legacy binding path.
"""

import argparse
from pathlib import Path

import torch

from gaussiansplatting.scene.gaussian_flame_face import GaussianFlameUVModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spatial-lr-scale", type=float, default=4.0)
    parser.add_argument("--flame-scale", type=float, default=-10.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must differ.")

    model = GaussianFlameUVModel(0, device=args.device)
    model.initialize_flame_state(args.spatial_lr_scale, args.flame_scale)
    model.load_ply(str(args.input))
    inside, finite = model._uv_inside_faces(model.get_uv, model._face_idx, 2e-4)
    if not (inside & finite).all():
        raise RuntimeError("PLY contains invalid UV/face bindings.")
    if not torch.isfinite(model.get_covariance()).all():
        raise RuntimeError("Conversion produced a non-finite covariance.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_ply(str(args.output))
    print(f"rewrote {model.num_gs} canonical UVD Gaussians to {args.output}")


if __name__ == "__main__":
    main()
