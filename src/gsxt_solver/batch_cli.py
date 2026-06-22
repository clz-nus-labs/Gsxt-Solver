from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .config import ModelPaths
from .solver import Solver


def test_number(path: Path) -> int:
    match = re.fullmatch(r"test(\d+)", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 10**9


def compact_result(image: Path, result: dict[str, Any]) -> dict[str, Any]:
    task_spec = result.get("task_spec") or {}
    return {
        "image": image.name,
        "status": "ok",
        "task_type": result.get("task_type") or task_spec.get("modality"),
        "target_order": (
            result.get("resolved_target_order")
            or task_spec.get("target_order")
            or ""
        ),
        "item_count": len(result.get("items") or []),
        "output_dir": result.get("runtime", {}).get("output_dir"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bundled GSXT test fixtures.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--fixtures", default="tests/fixtures")
    parser.add_argument("--output-dir", default="runs/test-suite")
    parser.add_argument("--python-executable")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    fixtures = Path(args.fixtures).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    images = sorted(fixtures.glob("test*.png"), key=test_number)
    if not images:
        raise FileNotFoundError(f"No test*.png fixtures found under: {fixtures}")

    models = ModelPaths.from_bundle(args.model_dir, project_root=project_root)
    solver = Solver.from_project(
        project_root,
        models=models,
        python_executable=args.python_executable,
        use_gpu=not args.cpu,
    )

    summaries: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        print(f"[{index:02d}/{len(images):02d}] {image.name}", flush=True)
        try:
            result = solver.solve(
                image,
                output_dir=output_root / image.stem,
                threshold=args.threshold,
            )
            summaries.append(compact_result(image, result))
        except Exception as exc:
            summaries.append(
                {
                    "image": image.name,
                    "status": "error",
                    "error": str(exc),
                }
            )

    payload = {
        "fixture_count": len(images),
        "success_count": sum(row["status"] == "ok" for row in summaries),
        "failure_count": sum(row["status"] == "error" for row in summaries),
        "results": summaries,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_path)
    if payload["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
