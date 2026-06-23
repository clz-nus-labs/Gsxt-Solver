from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from gsxt_solver.solver import Solver


class SolverOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image = self.root / "sample.png"
        self.image.write_bytes(b"image")
        self.solver = Solver.__new__(Solver)
        self.solver.project_root = self.root

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def fake_backend(self, image, *, output_dir=None, **_kwargs):
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        image_path = Path(image).resolve()
        visual_dir = run_dir / "visual"
        visual_dir.mkdir(parents=True, exist_ok=True)
        (visual_dir / image_path.name).write_bytes(b"visual")
        result_path = run_dir / "result.json"
        result_path.write_text("{}", encoding="utf-8")
        result = {
            "task_type": "icon",
            "task_spec": {"action": "explicit_order"},
            "items": [
                {
                    "kind": "icon",
                    "label": "star",
                    "center": [20, 30],
                    "bbox": [10, 15, 30, 45],
                }
            ],
            "runtime": {"output_dir": str(run_dir)},
        }
        return result, image_path, result_path

    def test_standard_mode_does_not_save_by_default(self) -> None:
        output_dir = self.root / "standard"
        with patch.object(self.solver, "_run_backend", side_effect=self.fake_backend):
            result = self.solver.predict(self.image, output_dir=output_dir)

        self.assertTrue(result["success"])
        self.assertFalse(output_dir.exists())

    def test_standard_mode_can_save_selected_outputs(self) -> None:
        output_dir = self.root / "standard-saved"
        with patch.object(self.solver, "_run_backend", side_effect=self.fake_backend):
            self.solver.predict(
                self.image,
                output_dir=output_dir,
                save_result=True,
                save_visual=False,
            )

        self.assertTrue((output_dir / "result.json").exists())
        self.assertFalse((output_dir / "visual").exists())

    def test_standard_mode_uses_default_output_directory_when_saving(self) -> None:
        with patch.object(self.solver, "_run_backend", side_effect=self.fake_backend):
            self.solver.predict(
                self.image,
                save_result=True,
                save_visual=True,
            )

        output_dir = self.root / "runs" / "sample"
        self.assertTrue((output_dir / "result.json").exists())
        self.assertTrue((output_dir / "visual" / self.image.name).exists())

    def test_debug_mode_saves_result_and_visual_by_default(self) -> None:
        output_dir = self.root / "debug"
        with patch.object(self.solver, "_run_backend", side_effect=self.fake_backend):
            result = self.solver.debug(self.image, output_dir=output_dir)

        self.assertEqual(result["runtime"]["output_dir"], str(output_dir.resolve()))
        self.assertTrue((output_dir / "result.json").exists())
        self.assertTrue((output_dir / "visual" / self.image.name).exists())

    def test_debug_mode_uses_default_output_directory(self) -> None:
        with patch.object(self.solver, "_run_backend", side_effect=self.fake_backend):
            self.solver.debug(self.image)

        output_dir = self.root / "runs" / "sample-debug"
        self.assertTrue((output_dir / "result.json").exists())
        self.assertTrue((output_dir / "visual" / self.image.name).exists())

    def test_debug_mode_can_disable_all_files(self) -> None:
        output_dir = self.root / "debug-disabled"
        with patch.object(self.solver, "_run_backend", side_effect=self.fake_backend):
            result = self.solver.debug(
                self.image,
                output_dir=output_dir,
                save_result=False,
                save_visual=False,
            )

        self.assertFalse(output_dir.exists())
        self.assertIsNone(result["runtime"]["output_dir"])
        self.assertTrue(result["runtime"]["temporary_output"])

    def test_solve_uses_mode_specific_defaults(self) -> None:
        standard_dir = self.root / "solve-standard"
        debug_dir = self.root / "solve-debug"
        with patch.object(self.solver, "_run_backend", side_effect=self.fake_backend):
            self.solver.solve(
                self.image,
                mode="standard",
                output_dir=standard_dir,
            )
            self.solver.solve(
                self.image,
                mode="debug",
                output_dir=debug_dir,
            )

        self.assertFalse(standard_dir.exists())
        self.assertTrue((debug_dir / "result.json").exists())
        self.assertTrue((debug_dir / "visual" / self.image.name).exists())


if __name__ == "__main__":
    unittest.main()
