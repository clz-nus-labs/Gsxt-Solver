from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RuntimeConstraintTests(unittest.TestCase):
    def test_numpy_and_opencv_variants_are_constrained(self) -> None:
        constraints = (
            ROOT / "Scripts" / "Gsxt" / "runtime-constraints.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("numpy<2", constraints)
        self.assertIn("opencv-python<=4.6.0", constraints)
        self.assertIn("opencv-contrib-python<4.12", constraints)
        self.assertIn("opencv-python-headless<4.12", constraints)

    def test_setup_scripts_apply_runtime_constraints(self) -> None:
        for name in (
            "setup_paddledetection_repo.ps1",
            "setup_paddleocr_repo.ps1",
        ):
            script = (
                ROOT / "Scripts" / "Gsxt" / "training" / name
            ).read_text(encoding="utf-8")
            self.assertIn("runtime-constraints.txt", script)
            self.assertIn(" -c ", script)


if __name__ == "__main__":
    unittest.main()
