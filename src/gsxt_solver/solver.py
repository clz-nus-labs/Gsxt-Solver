from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .config import ModelPaths


class SolverError(RuntimeError):
    pass


class Solver:
    def __init__(
        self,
        *,
        project_root: str | Path,
        models: ModelPaths,
        python_executable: str | Path | None = None,
        use_gpu: bool = True,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.models = models
        self.python_executable = str(python_executable or sys.executable)
        self.use_gpu = use_gpu
        self.backend = (
            self.project_root
            / "Scripts"
            / "Gsxt"
            / "demos"
            / "dynamic_mixed_infer.py"
        )

    @classmethod
    def from_project(
        cls,
        project_root: str | Path,
        *,
        models: ModelPaths | None = None,
        python_executable: str | Path | None = None,
        use_gpu: bool = True,
    ) -> "Solver":
        root = Path(project_root).resolve()
        return cls(
            project_root=root,
            models=models or ModelPaths.from_project(root),
            python_executable=python_executable,
            use_gpu=use_gpu,
        )

    def validate(self) -> None:
        if not self.backend.exists():
            raise FileNotFoundError(f"Inference backend not found: {self.backend}")
        for repository in ("PaddleDetection", "PaddleOCR"):
            path = (
                self.project_root
                / "Scripts"
                / "Gsxt"
                / "third_party"
                / repository
            )
            if not path.exists():
                raise FileNotFoundError(f"Required source dependency not found: {path}")
        self.models.validate()

    def solve(
        self,
        image: str | Path,
        *,
        output_dir: str | Path | None = None,
        threshold: float = 0.3,
        target_order: str = "",
        timeout: int = 300,
    ) -> dict[str, Any]:
        self.validate()
        image_path = Path(image).resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if output_dir is None:
            run_dir = Path(tempfile.mkdtemp(prefix="gsxt-solver-"))
        else:
            run_dir = Path(output_dir).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)

        command = [
            self.python_executable,
            str(self.backend),
            "--image",
            str(image_path),
            "--output-dir",
            str(run_dir),
            "--threshold",
            str(threshold),
            "--det-config",
            str(self.models.det_config),
            "--det-weights",
            str(self.models.det_weights),
            "--det-dataset",
            str(self.models.det_dataset),
            "--rec-config",
            str(self.models.rec_config),
            "--rec-weights",
            str(self.models.rec_weights),
            "--icon-weights",
            str(self.models.icon_weights),
            "--icon-labels",
            str(self.models.icon_labels),
        ]
        if target_order:
            command.extend(["--target-order", target_order])
        if not self.use_gpu:
            command.append("--cpu")

        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise SolverError(
                "Inference failed.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        result_path = run_dir / "result.json"
        if not result_path.exists():
            raise SolverError(
                f"Inference completed without result.json.\nstdout:\n{completed.stdout}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["runtime"] = {
            "output_dir": str(run_dir),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if output_dir is None:
            result["runtime"]["temporary_output"] = True
        return result
