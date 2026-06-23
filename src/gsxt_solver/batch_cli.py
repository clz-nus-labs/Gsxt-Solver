from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .result import format_error_result
from .solver import Solver


def test_number(path: Path) -> int:
    match = re.fullmatch(r"test(\d+)", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 10**9


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

    solver = Solver.from_bundle(
        project_root,
        args.model_dir,
        python_executable=args.python_executable,
        use_gpu=not args.cpu,
    )

    summaries: list[dict] = []
    for index, image in enumerate(images, start=1):
        print(f"[{index:02d}/{len(images):02d}] {image.name}", flush=True)
        try:
            result = solver.predict(
                image,
                output_dir=output_root / image.stem,
                save_result=True,
                save_visual=True,
                threshold=args.threshold,
            )
            summaries.append(result)
        except Exception as exc:
            summaries.append(format_error_result(image=image, error=exc))

    payload = {
        "fixture_count": len(images),
        "success_count": sum(row.get("success") is True for row in summaries),
        "failure_count": sum(row.get("success") is not True for row in summaries),
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
