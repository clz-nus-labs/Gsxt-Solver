from __future__ import annotations

import argparse
import json
from pathlib import Path

from .solver import Solver


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSXT structured image inference.")
    parser.add_argument("image")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--model-dir",
        help="Model bundle downloaded by gsxt-models; defaults to training outputs.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--python-executable")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--target-order", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["standard", "debug"],
        default="standard",
        help="standard returns the stable public schema; debug returns full diagnostics.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    if args.model_dir:
        solver = Solver.from_bundle(
            project_root,
            args.model_dir,
            python_executable=args.python_executable,
            use_gpu=not args.cpu,
        )
    else:
        solver = Solver.from_project(
            project_root,
            python_executable=args.python_executable,
            use_gpu=not args.cpu,
        )
    result = solver.solve(
        args.image,
        mode=args.mode,
        output_dir=args.output_dir,
        threshold=args.threshold,
        target_order=args.target_order,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.mode == "standard" and result.get("success") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
