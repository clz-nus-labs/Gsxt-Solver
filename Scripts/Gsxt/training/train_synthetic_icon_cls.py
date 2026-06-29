from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader, Dataset
from paddle.vision.models import mobilenet_v3_large, mobilenet_v3_small, resnet18

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "synthetic_icon_cls"
DEFAULT_OUTPUT = PROJECT_ROOT / "Scripts" / "Gsxt" / "output" / "training" / "synthetic_icon_cls"
DEFAULT_BACKBONE_OUTPUT = PROJECT_ROOT / "Scripts" / "Gsxt" / "output" / "training" / "synthetic_icon_backbone_cls"


@dataclass
class TrainState:
    epoch: int
    best_acc: float
    best_epoch: int
    stale_epochs: int


class IconDataset(Dataset):
    def __init__(
        self,
        list_path: Path,
        label_to_id: dict[str, int],
        image_size: int,
        train: bool,
        augment: str = "default",
    ) -> None:
        self.rows: list[tuple[Path, int]] = []
        self.image_size = image_size
        self.train = train
        self.augment = augment
        for line in list_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            path_text, label = line.split("\t", 1)
            self.rows.append((Path(path_text), label_to_id[label]))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        path, label = self.rows[index]
        img = Image.open(path).convert("RGB")
        if self.train and self.augment != "none" and random.random() < (0.12 if self.augment == "mild" else 0.25):
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.train and self.augment != "none" and random.random() < (0.25 if self.augment == "mild" else 0.55):
            max_angle = 8 if self.augment == "mild" else 18
            angle = random.uniform(-max_angle, max_angle)
            img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0))
        if self.train and self.augment != "none" and random.random() < (0.15 if self.augment == "mild" else 0.35):
            arr_tmp = np.asarray(img).astype("float32")
            if self.augment == "mild":
                factor = random.uniform(0.9, 1.1)
                bias = random.uniform(-8, 8)
            else:
                factor = random.uniform(0.75, 1.25)
                bias = random.uniform(-20, 20)
            arr_tmp = np.clip(arr_tmp * factor + bias, 0, 255).astype("uint8")
            img = Image.fromarray(arr_tmp)
        img.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        x = (self.image_size - img.width) // 2
        y = (self.image_size - img.height) // 2
        canvas.paste(img, (x, y))
        arr = np.asarray(canvas).astype("float32") / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype="float32")) / np.array(
            [0.229, 0.224, 0.225], dtype="float32"
        )
        arr = arr.transpose(2, 0, 1)
        return arr, np.array(label, dtype="int64")


class ConvBNAct(nn.Layer):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, 3, stride=stride, padding=1, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Layer):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels)
        self.conv2 = nn.Conv2D(channels, channels, 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(channels)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class SmallIconCNN(nn.Layer):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 48),
            ConvBNAct(48, 64, stride=2),
            ResidualBlock(64),
            ConvBNAct(64, 128, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBNAct(128, 256, stride=2),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.AdaptiveAvgPool2D(1),
        )
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.features(x)
        x = paddle.flatten(x, start_axis=1)
        x = self.dropout(x)
        return self.classifier(x)


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Layer:
    if model_name == "small_cnn":
        return SmallIconCNN(num_classes=num_classes)
    if model_name == "mobilenet_v3_large":
        return mobilenet_v3_large(pretrained=pretrained, num_classes=num_classes)
    if model_name == "mobilenet_v3_small":
        return mobilenet_v3_small(pretrained=pretrained, num_classes=num_classes)
    if model_name == "resnet18":
        return resnet18(pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def class_weights(dataset: IconDataset, num_classes: int) -> paddle.Tensor:
    counts = np.zeros(num_classes, dtype="float32")
    for _, label in dataset.rows:
        counts[label] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (counts * num_classes)
    return paddle.to_tensor(weights.astype("float32"))


def freeze_backbone(model: nn.Layer) -> tuple[int, int]:
    frozen = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        is_classifier = name.startswith("classifier")
        parameter.stop_gradient = not is_classifier
        if is_classifier:
            trainable += int(np.prod(parameter.shape))
        else:
            frozen += int(np.prod(parameter.shape))
    return frozen, trainable


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
    )


@paddle.no_grad()
def evaluate(model: nn.Layer, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    losses: list[float] = []
    iterator = tqdm(loader, desc="eval", leave=False) if tqdm is not None else loader
    for images, labels in iterator:
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(axis=1)
        correct += int((pred == labels).astype("int64").sum().numpy())
        total += int(labels.shape[0])
        losses.append(float(loss.numpy()))
    acc = correct / max(1, total)
    return acc, float(np.mean(losses)) if losses else 0.0


def save_checkpoint(
    output_dir: Path,
    name: str,
    model: nn.Layer,
    optimizer: paddle.optimizer.Optimizer,
    state: TrainState,
) -> None:
    paddle.save(model.state_dict(), str(output_dir / f"{name}.pdparams"))
    paddle.save(optimizer.state_dict(), str(output_dir / f"{name}.pdopt"))
    (output_dir / f"{name}.json").write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(
    output_dir: Path,
    model: nn.Layer,
    optimizer: paddle.optimizer.Optimizer,
) -> TrainState:
    params_path = output_dir / "latest.pdparams"
    opt_path = output_dir / "latest.pdopt"
    state_path = output_dir / "latest.json"
    if not params_path.exists() or not opt_path.exists() or not state_path.exists():
        raise FileNotFoundError("Missing latest checkpoint files for resume.")
    model.set_state_dict(paddle.load(str(params_path)))
    optimizer.set_state_dict(paddle.load(str(opt_path)))
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return TrainState(**payload)


def load_initial_weights(model: nn.Layer, checkpoint: Path) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint}")
    source_state = paddle.load(str(checkpoint))
    target_state = model.state_dict()
    compatible = {}
    skipped: list[str] = []
    for name, value in source_state.items():
        if name not in target_state:
            skipped.append(name)
            continue
        if list(value.shape) != list(target_state[name].shape):
            skipped.append(name)
            continue
        compatible[name] = value
    if not compatible:
        raise ValueError(f"No compatible tensors found in checkpoint: {checkpoint}")
    target_state.update(compatible)
    model.set_state_dict(target_state)
    print(
        f"init_checkpoint={checkpoint.resolve()} "
        f"loaded_tensors={len(compatible)} skipped_tensors={len(skipped)}"
    )
    if skipped:
        print("skipped_tensor_examples=" + ", ".join(skipped[:8]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--model",
        choices=["small_cnn", "mobilenet_v3_large", "mobilenet_v3_small", "resnet18"],
        default="small_cnn",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0006)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--augment",
        choices=["none", "mild", "default"],
        default="default",
        help="Training augmentation strength. Conservative fine-tuning usually works best with mild or none.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights. Useful for conservative fine-tuning from an already strong model.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze all parameters except classifier.*. Useful for a first conservative adaptation pass.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Load compatible model weights before training. Use for fine-tuning from an existing icon model.",
    )
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.set_device(args.device)

    dataset_dir = Path(args.dataset)
    if args.output:
        output_dir = Path(args.output)
    elif args.model == "small_cnn":
        output_dir = DEFAULT_OUTPUT
    else:
        output_dir = DEFAULT_BACKBONE_OUTPUT / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [line.strip() for line in (dataset_dir / "label_list.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    label_to_id = {label: index for index, label in enumerate(labels)}
    (output_dir / "label_list.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    train_ds = IconDataset(dataset_dir / "train.txt", label_to_id, args.image_size, train=True, augment=args.augment)
    val_ds = IconDataset(dataset_dir / "val.txt", label_to_id, args.image_size, train=False, augment="none")
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, workers=0)

    model = build_model(args.model, num_classes=len(labels), pretrained=args.pretrained)
    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.lr, T_max=args.epochs)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=1.0e-4,
    )
    weight_tensor = class_weights(train_ds, len(labels))

    state = TrainState(epoch=0, best_acc=0.0, best_epoch=0, stale_epochs=0)
    if args.resume:
        state = load_checkpoint(output_dir, model, optimizer)
        print(f"resume_from_epoch={state.epoch}, best_acc={state.best_acc:.6f}")
    elif args.init_checkpoint:
        load_initial_weights(model, Path(args.init_checkpoint))

    if args.freeze_backbone:
        frozen_count, trainable_count = freeze_backbone(model)
        print(f"freeze_backbone=True frozen_params={frozen_count} trainable_params={trainable_count}")

    print(f"model={args.model} pretrained={args.pretrained}")
    print(f"classes={len(labels)} train={len(train_ds)} val={len(val_ds)}")
    print(f"augment={args.augment} class_weights={not args.no_class_weights}")
    print(f"output={output_dir.resolve()}")

    for epoch in range(state.epoch + 1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        iterator = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch") if tqdm is not None else train_loader
        for images, labels_tensor in iterator:
            logits = model(images)
            loss = F.cross_entropy(logits, labels_tensor, weight=None if args.no_class_weights else weight_tensor)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            losses.append(float(loss.numpy()))
            if tqdm is not None:
                iterator.set_postfix(loss=f"{np.mean(losses):.4f}")

        val_acc, val_loss = evaluate(model, val_loader)
        lr_scheduler.step()
        train_loss = float(np.mean(losses)) if losses else 0.0
        improved = val_acc > state.best_acc + args.min_delta
        if improved:
            state.best_acc = val_acc
            state.best_epoch = epoch
            state.stale_epochs = 0
        else:
            state.stale_epochs += 1
        state.epoch = epoch

        save_checkpoint(output_dir, "latest", model, optimizer, state)
        if improved:
            save_checkpoint(output_dir, "best_accuracy", model, optimizer, state)

        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.6f} "
            f"best_acc={state.best_acc:.6f} best_epoch={state.best_epoch} stale={state.stale_epochs}/{args.patience}"
        )

        if state.stale_epochs >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={state.best_epoch} best_acc={state.best_acc:.6f}")
            break


if __name__ == "__main__":
    main()
