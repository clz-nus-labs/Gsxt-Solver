from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPaths:
    det_weights: Path
    det_config: Path
    det_dataset: Path
    rec_weights: Path
    rec_config: Path
    icon_weights: Path
    icon_labels: Path

    @classmethod
    def from_project(cls, project_root: str | Path) -> "ModelPaths":
        root = Path(project_root).resolve()
        gsxt = root / "Scripts" / "Gsxt"
        return cls(
            det_weights=gsxt
            / "output"
            / "training"
            / "paddledet_external_mixed"
            / "best_model.pdparams",
            det_config=gsxt
            / "third_party"
            / "PaddleDetection"
            / "configs"
            / "picodet"
            / "picodet_s_320_coco_lcnet.yml",
            det_dataset=gsxt / "data" / "datasets" / "external_mixed_paddledet",
            rec_weights=gsxt
            / "output"
            / "training"
            / "chinese_char_rec_ppocrv4_domain_finetune"
            / "best_accuracy.pdparams",
            rec_config=gsxt
            / "output"
            / "training"
            / "chinese_char_rec_ppocrv4_domain_finetune"
            / "config.yml",
            icon_weights=gsxt
            / "output"
            / "training"
            / "icon_cls_geetest_plus_synthetic_mobilenet_v3_large"
            / "best_accuracy.pdparams",
            icon_labels=gsxt
            / "output"
            / "training"
            / "icon_cls_geetest_plus_synthetic_mobilenet_v3_large"
            / "label_list.txt",
        )

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str | Path,
        *,
        project_root: str | Path,
    ) -> "ModelPaths":
        bundle = Path(bundle_dir).resolve()
        project = Path(project_root).resolve()
        det_config = (
            project
            / "Scripts"
            / "Gsxt"
            / "third_party"
            / "PaddleDetection"
            / "configs"
            / "picodet"
            / "picodet_s_320_coco_lcnet.yml"
        )
        return cls(
            det_weights=bundle / "det" / "best_model.pdparams",
            det_config=det_config,
            det_dataset=bundle / "det" / "dataset",
            rec_weights=bundle / "rec" / "best_accuracy.pdparams",
            rec_config=bundle / "rec" / "config.yml",
            icon_weights=bundle / "icon" / "best_accuracy.pdparams",
            icon_labels=bundle / "icon" / "label_list.txt",
        )

    def validate(self) -> None:
        missing = [
            f"{name}: {path}"
            for name, path in self.__dict__.items()
            if not Path(path).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing model/runtime files:\n" + "\n".join(missing)
            )
