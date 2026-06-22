from __future__ import annotations

import unittest

from gsxt_solver.result import format_error_result, format_standard_result


class StandardResultTests(unittest.TestCase):
    def test_removes_diagnostic_scores(self) -> None:
        debug_result = {
            "task_type": "char",
            "task_spec": {
                "action": "semantic_order",
                "modality": "char",
                "confidence": 0.91,
                "evidence": {"instruction_score": 0.88},
            },
            "items": [
                {
                    "kind": "char",
                    "text": "古",
                    "center": [120, 80],
                    "bbox": [100, 50, 140, 110],
                    "det_score": 0.7,
                    "rec_score": 0.9,
                    "candidate_scores": {"古": 0.9, "右": 0.1},
                }
            ],
        }

        result = format_standard_result(debug_result, image="example.png")

        self.assertEqual(result["schema_version"], "1.0")
        self.assertEqual(result["task"], {"action": "semantic_order", "type": "char"})
        self.assertEqual(result["result"]["sequence"], ["古"])
        self.assertEqual(result["result"]["points"], [{"x": 120, "y": 80}])
        self.assertNotIn("det_score", str(result))
        self.assertNotIn("rec_score", str(result))
        self.assertNotIn("confidence", str(result))

    def test_formats_icon_item(self) -> None:
        result = format_standard_result(
            {
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
            },
            image="icon.png",
        )

        self.assertEqual(
            result["result"]["items"][0],
            {
                "index": 1,
                "type": "icon",
                "value": "star",
                "center": {"x": 20, "y": 30},
                "bbox": {"left": 10, "top": 15, "right": 30, "bottom": 45},
            },
        )

    def test_formats_error_without_traceback(self) -> None:
        result = format_error_result(
            image="broken.png",
            error=ValueError("invalid image\ninternal detail"),
        )

        self.assertEqual(
            result,
            {
                "schema_version": "1.0",
                "success": False,
                "image": "broken.png",
                "error": {
                    "type": "ValueError",
                    "message": "invalid image",
                },
            },
        )

    def test_empty_detection_is_a_valid_standard_result(self) -> None:
        result = format_standard_result(
            {
                "task_type": "mixed",
                "task_spec": {"action": "detect_only"},
                "items": [],
            },
            image="empty.png",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["result"]["count"], 0)
        self.assertEqual(result["result"]["sequence"], [])
        self.assertEqual(result["result"]["points"], [])


if __name__ == "__main__":
    unittest.main()
