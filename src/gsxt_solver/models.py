from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from importlib.resources import files
from pathlib import Path


MINIMAL_DETECTION_DATASET = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "char"},
        {"id": 2, "name": "icon"},
    ],
}


def load_manifest() -> dict:
    manifest_path = files("gsxt_solver").joinpath("assets/models.json")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_models(destination: str | Path, release_base_url: str) -> Path:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    for asset in manifest["assets"]:
        target = destination / asset["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target).lower() == asset["sha256"].lower():
            continue
        url = release_base_url.rstrip("/") + "/" + asset["release_name"]
        temporary = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha256(temporary).lower() != asset["sha256"].lower():
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {asset['release_name']}")
        temporary.replace(target)
    dataset_dir = destination / "det" / "dataset"
    (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "val.json").write_text(
        json.dumps(MINIMAL_DETECTION_DATASET, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GSXT model release assets.")
    parser.add_argument("--destination", default="models")
    parser.add_argument(
        "--release-base-url",
        required=True,
        help="Example: https://github.com/OWNER/REPO/releases/download/models-v0.1.0",
    )
    args = parser.parse_args()
    print(download_models(args.destination, args.release_base_url))


if __name__ == "__main__":
    main()
