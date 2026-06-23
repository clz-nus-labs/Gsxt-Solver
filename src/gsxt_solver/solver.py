from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from .config import ModelPaths
from .result import format_error_result, format_standard_result


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

    @classmethod
    def from_bundle(
        cls,
        project_root: str | Path,
        model_dir: str | Path,
        *,
        python_executable: str | Path | None = None,
        use_gpu: bool = True,
    ) -> "Solver":
        """Create a solver from a model directory assembled by ``gsxt-models``."""

        root = Path(project_root).resolve()
        return cls(
            project_root=root,
            models=ModelPaths.from_bundle(model_dir, project_root=root),
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

    def _run_backend(
        self,
        image: str | Path,
        *,
        output_dir: str | Path | None = None,
        threshold: float = 0.3,
        target_order: str = "",
        timeout: int = 300,
    ) -> tuple[dict[str, Any], Path, Path]:
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
        return result, image_path, result_path

    def _resolve_output_dir(
        self,
        image: str | Path,
        *,
        mode: Literal["standard", "debug"],
        output_dir: str | Path | None,
    ) -> Path:
        if output_dir is not None:
            return Path(output_dir).resolve()
        image_path = Path(image)
        suffix = "-debug" if mode == "debug" else ""
        return self.project_root / "runs" / f"{image_path.stem}{suffix}"

    @staticmethod
    def _save_outputs(
        result: dict[str, Any],
        *,
        image_path: Path,
        work_dir: str | Path,
        output_dir: Path,
        save_result: bool,
        save_visual: bool,
    ) -> None:
        if not save_result and not save_visual:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        if save_result:
            (output_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if save_visual:
            source_visual = Path(work_dir) / "visual" / image_path.name
            if source_visual.exists():
                visual_dir = output_dir / "visual"
                visual_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_visual, visual_dir / image_path.name)

    def predict(
        self,
        image: str | Path,
        *,
        output_dir: str | Path | None = None,
        save_result: bool = False,
        save_visual: bool = False,
        threshold: float = 0.3,
        target_order: str = "",
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Run inference and return the stable, probability-free public result."""

        try:
            with tempfile.TemporaryDirectory(prefix="gsxt-solver-standard-") as work_dir:
                debug_result, image_path, _ = self._run_backend(
                    image,
                    output_dir=work_dir,
                    threshold=threshold,
                    target_order=target_order,
                    timeout=timeout,
                )
                result = format_standard_result(debug_result, image=image_path)

                if save_result or save_visual:
                    self._save_outputs(
                        result,
                        image_path=image_path,
                        work_dir=work_dir,
                        output_dir=self._resolve_output_dir(
                            image_path,
                            mode="standard",
                            output_dir=output_dir,
                        ),
                        save_result=save_result,
                        save_visual=save_visual,
                    )
        except Exception as error:
            result = format_error_result(image=image, error=error)
            if save_result:
                public_dir = self._resolve_output_dir(
                    image,
                    mode="standard",
                    output_dir=output_dir,
                )
                public_dir.mkdir(parents=True, exist_ok=True)
                (public_dir / "result.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        return result

    def debug(
        self,
        image: str | Path,
        *,
        output_dir: str | Path | None = None,
        save_result: bool = True,
        save_visual: bool = True,
        threshold: float = 0.3,
        target_order: str = "",
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Run inference and return the full diagnostic backend payload."""

        with tempfile.TemporaryDirectory(prefix="gsxt-solver-debug-") as work_dir:
            result, image_path, _ = self._run_backend(
                image,
                output_dir=work_dir,
                threshold=threshold,
                target_order=target_order,
                timeout=timeout,
            )
            if save_result or save_visual:
                public_dir = self._resolve_output_dir(
                    image_path,
                    mode="debug",
                    output_dir=output_dir,
                )
                result["runtime"]["output_dir"] = str(public_dir)
                result["runtime"].pop("temporary_output", None)
                self._save_outputs(
                    result,
                    image_path=image_path,
                    work_dir=work_dir,
                    output_dir=public_dir,
                    save_result=save_result,
                    save_visual=save_visual,
                )
            else:
                result["runtime"]["output_dir"] = None
                result["runtime"]["temporary_output"] = True
        return result

    def solve(
        self,
        image: str | Path,
        *,
        mode: Literal["standard", "debug"] = "standard",
        output_dir: str | Path | None = None,
        save_result: bool | None = None,
        save_visual: bool | None = None,
        threshold: float = 0.3,
        target_order: str = "",
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Run in standard mode by default, or return diagnostics with mode='debug'."""

        if mode not in {"standard", "debug"}:
            raise ValueError("mode must be 'standard' or 'debug'")
        method = self.predict if mode == "standard" else self.debug
        default_save = mode == "debug"
        return method(
            image,
            output_dir=output_dir,
            save_result=default_save if save_result is None else save_result,
            save_visual=default_save if save_visual is None else save_visual,
            threshold=threshold,
            target_order=target_order,
            timeout=timeout,
        )
