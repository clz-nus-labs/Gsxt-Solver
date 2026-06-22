from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for lightweight use.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "Scripts" / "Gsxt" / "synthetic" / "output"
DEFAULT_BACKGROUNDS = PROJECT_ROOT / "Scripts" / "Gsxt" / "synthetic" / "backgrounds"

CHARS = list(
    "古罗马春夏秋冬东南西北中山水火木金土日月星云风雨电花鸟鱼虫甲乙丙丁"
    "天地人上下左右前后大小多少明暗红蓝绿黄紫白黑江河湖海林森田石"
    "车船门窗书笔刀剑牛羊犬猫龙凤竹梅兰荷城桥塔钟鼓"
)
ICONS = [
    "apple",
    "burger",
    "star",
    "bolt",
    "shield",
    "circle",
    "heart",
    "triangle",
    "square",
    "diamond",
    "moon",
    "sun",
    "cloud",
    "umbrella",
    "flag",
    "cross",
    "target",
    "coin",
    "speaker",
    "exclamation",
    "ring",
    "gear",
    "arrow",
    "check",
    "question",
    "play",
    "pause",
    "plus",
    "minus",
    "droplet",
    "leaf",
    "location",
    "home",
    "key",
    "lock",
    "bell",
    "camera",
    "book",
    "clock",
    "crown",
]
ICON_DISPLAY = {
    "apple": "苹果",
    "burger": "汉堡",
    "star": "星",
    "bolt": "闪电",
    "shield": "盾",
    "circle": "圆",
    "heart": "心",
    "triangle": "三角",
    "square": "方块",
    "diamond": "菱形",
    "moon": "月亮",
    "sun": "太阳",
    "cloud": "云朵",
    "umbrella": "伞",
    "flag": "旗",
    "cross": "十字",
    "target": "靶心",
    "coin": "圆章",
    "speaker": "喇叭",
    "exclamation": "叹号",
    "ring": "圆环",
    "gear": "齿轮",
    "arrow": "箭头",
    "check": "对勾",
    "question": "问号",
    "play": "播放",
    "pause": "暂停",
    "plus": "加号",
    "minus": "减号",
    "droplet": "水滴",
    "leaf": "叶子",
    "location": "定位",
    "home": "房子",
    "key": "钥匙",
    "lock": "锁",
    "bell": "铃铛",
    "camera": "相机",
    "book": "书本",
    "clock": "时钟",
    "crown": "皇冠",
}
PROMPT_PREFIXES = ["请依次点击：", "请按顺序选择：", "目标顺序："]


@dataclass
class ObjectLabel:
    label_type: str
    label: str
    bbox: list[int]
    score: float = 1.0


def find_font() -> str:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/STSONG.TTF"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError("No Chinese font found in C:/Windows/Fonts")


def load_chars(mode: str, limit: int, seed: int) -> list[str]:
    base_chars = list(dict.fromkeys(CHARS))
    if mode == "base":
        return base_chars

    dict_path = PROJECT_ROOT / "Scripts" / "Gsxt" / "third_party" / "PaddleOCR" / "ppocr" / "utils" / "ppocr_keys_v1.txt"
    chars: list[str] = []
    if dict_path.exists():
        for line in dict_path.read_text(encoding="utf-8").splitlines():
            for char in line.strip():
                if "\u4e00" <= char <= "\u9fff":
                    chars.append(char)
    if not chars:
        # Common CJK Unified Ideographs range fallback. This is broad; labels are
        # still filtered by font rendering and model dictionary during training.
        chars = [chr(code) for code in range(0x4E00, 0x9FA6)]
    chars = list(dict.fromkeys(base_chars + chars))
    if limit > 0 and len(chars) > limit:
        rng = random.Random(seed)
        head = base_chars[:]
        tail = [char for char in chars if char not in set(head)]
        rng.shuffle(tail)
        chars = head + tail[: max(0, limit - len(head))]
    return chars


def rand_color(lo: int = 20, hi: int = 245) -> tuple[int, int, int]:
    return tuple(random.randint(lo, hi) for _ in range(3))


def make_background(width: int, height: int) -> Image.Image:
    base_a = rand_color(50, 180)
    base_b = rand_color(80, 230)
    img = Image.new("RGB", (width, height), base_a)
    pix = img.load()
    wave_period = random.uniform(25, 80)
    for y in range(height):
        t = y / max(1, height - 1)
        for x in range(width):
            wave = 0.08 * math.sin((x + y) / wave_period)
            k = min(1.0, max(0.0, t + wave))
            pix[x, y] = tuple(int(base_a[i] * (1 - k) + base_b[i] * k) for i in range(3))

    draw = ImageDraw.Draw(img, "RGBA")
    for _ in range(random.randint(30, 80)):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(2, 18)
        color = (*rand_color(80, 255), random.randint(18, 60))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    if random.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.2)))
    return img


def list_backgrounds(background_dir: Path | None) -> list[Path]:
    if background_dir is None or not background_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [path for path in background_dir.rglob("*") if path.suffix.lower() in exts]


def make_photo_like_background(width: int, height: int) -> Image.Image:
    horizon = random.randint(int(height * 0.35), int(height * 0.62))
    sky_a = random.choice([(130, 180, 230), (240, 170, 80), (170, 190, 230), (80, 130, 190)])
    sky_b = random.choice([(245, 235, 210), (255, 210, 110), (120, 170, 220), (210, 180, 220)])
    land_a = random.choice([(40, 120, 60), (165, 125, 45), (60, 95, 70), (120, 160, 80)])
    land_b = random.choice([(20, 70, 45), (210, 165, 70), (90, 120, 65), (45, 90, 60)])
    img = Image.new("RGB", (width, height), sky_a)
    pix = img.load()
    for y in range(height):
        if y < horizon:
            t = y / max(1, horizon)
            base1, base2 = sky_a, sky_b
        else:
            t = (y - horizon) / max(1, height - horizon)
            base1, base2 = land_a, land_b
        for x in range(width):
            wave = 0.08 * math.sin(x / random.uniform(35, 95) + y / random.uniform(50, 120))
            k = min(1.0, max(0.0, t + wave))
            pix[x, y] = tuple(int(base1[i] * (1 - k) + base2[i] * k) for i in range(3))

    draw = ImageDraw.Draw(img, "RGBA")
    for _ in range(random.randint(2, 5)):
        y = random.randint(max(0, horizon - 70), min(height - 1, horizon + 40))
        color = (*rand_color(35, 120), random.randint(80, 150))
        pts = [(0, y + random.randint(-12, 12))]
        for x in range(0, width + 80, 80):
            pts.append((x, y + random.randint(-35, 35)))
        pts.extend([(width, height), (0, height)])
        draw.polygon(pts, fill=color)
    for _ in range(random.randint(4, 9)):
        y = random.randint(horizon, height)
        cx = random.randint(0, width)
        rx = random.randint(max(80, width // 4), max(120, width))
        ry = random.randint(20, 90)
        draw.arc(
            [cx - rx, y - ry, cx + rx, y + ry],
            190,
            345,
            fill=(*rand_color(80, 230), random.randint(45, 100)),
            width=random.randint(2, 5),
        )
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 0.8)))


def make_background_from_image(width: int, height: int, background_paths: list[Path]) -> Image.Image:
    path = random.choice(background_paths)
    img = Image.open(path).convert("RGB")
    scale = max(width / img.width, height / img.height)
    new_size = (max(width, int(img.width * scale)), max(height, int(img.height * scale)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    x = random.randint(0, img.width - width)
    y = random.randint(0, img.height - height)
    img = img.crop((x, y, x + width, y + height))
    if random.random() < 0.45:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.2)))
    return img


def choose_background(width: int, height: int, background_paths: list[Path]) -> Image.Image:
    if background_paths and random.random() < 0.75:
        return make_background_from_image(width, height, background_paths)
    if random.random() < 0.55:
        return make_photo_like_background(width, height)
    return make_background(width, height)


def alpha_bbox(layer: Image.Image) -> list[int] | None:
    bbox = layer.getbbox()
    if bbox is None:
        return None
    return [int(v) for v in bbox]


def bbox_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def expanded(box: list[int], margin: int) -> list[int]:
    return [box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin]


def intersection_area(a: list[int], b: list[int]) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def paste_layer(base: Image.Image, layer: Image.Image, x: int, y: int) -> list[int]:
    base.alpha_composite(layer, (x, y))
    bbox = alpha_bbox(layer)
    if bbox is None:
        return [x, y, x, y]
    return [x + bbox[0], y + bbox[1], x + bbox[2], y + bbox[3]]


def shear_layer(layer: Image.Image) -> Image.Image:
    sx = random.uniform(-0.22, 0.22)
    sy = random.uniform(-0.12, 0.12)
    w, h = layer.size
    new_w = int(w + abs(sx) * h) + 8
    new_h = int(h + abs(sy) * w) + 8
    return layer.transform(
        (new_w, new_h),
        Image.Transform.AFFINE,
        (1, sx, 4 if sx < 0 else 0, sy, 1, 4 if sy < 0 else 0),
        resample=Image.Resampling.BICUBIC,
    )


def wave_distort(layer: Image.Image) -> Image.Image:
    src = layer
    w, h = src.size
    amp = random.uniform(1.5, 5.0)
    period = random.uniform(28, 70)
    dst = Image.new("RGBA", (w + int(amp * 2) + 4, h), (0, 0, 0, 0))
    for y in range(h):
        offset = int(math.sin(y / period * math.tau) * amp)
        crop = src.crop((0, y, w, y + 1))
        dst.alpha_composite(crop, (offset + int(amp) + 2, y))
    return dst


def add_cutout(layer: Image.Image) -> Image.Image:
    if random.random() > 0.55:
        return layer
    mask = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask, "RGBA")
    for _ in range(random.randint(1, 4)):
        x1 = random.randint(0, max(0, layer.width - 4))
        y1 = random.randint(0, max(0, layer.height - 4))
        x2 = min(layer.width, x1 + random.randint(10, max(12, layer.width // 3)))
        y2 = min(layer.height, y1 + random.randint(3, max(5, layer.height // 7)))
        draw.rounded_rectangle([x1, y1, x2, y2], radius=random.randint(1, 5), fill=(0, 0, 0, random.randint(55, 120)))
    base = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    alpha = layer.getchannel("A")
    alpha = Image.composite(Image.new("L", layer.size, 0), alpha, mask.getchannel("A"))
    base.paste(layer, (0, 0), alpha)
    return base


def add_pixel_speckles(layer: Image.Image) -> Image.Image:
    noisy = layer.copy()
    draw = ImageDraw.Draw(noisy, "RGBA")
    count = random.randint(180, 520)
    for _ in range(count):
        x = random.randint(0, max(0, noisy.width - 1))
        y = random.randint(0, max(0, noisy.height - 1))
        r = random.choice([1, 1, 1, 2, 3])
        color = (*rand_color(0, 255), random.randint(70, 210))
        if random.random() < 0.75:
            draw.point((x, y), fill=color)
        else:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return noisy


def add_local_blur(layer: Image.Image) -> Image.Image:
    if random.random() > 0.75:
        return layer
    blurred = layer.filter(ImageFilter.GaussianBlur(random.uniform(1.0, 2.4)))
    mask = Image.new("L", layer.size, 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(random.randint(1, 3)):
        x = random.randint(0, max(0, layer.width - 20))
        y = random.randint(0, max(0, layer.height - 20))
        w = random.randint(18, max(24, layer.width // 2))
        h = random.randint(12, max(18, layer.height // 3))
        draw.ellipse([x, y, min(layer.width, x + w), min(layer.height, y + h)], fill=random.randint(90, 180))
    return Image.composite(blurred, layer, mask)


def add_translucent_occluders(layer: Image.Image) -> Image.Image:
    if random.random() > 0.7:
        return layer
    overlay = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for _ in range(random.randint(1, 5)):
        x1 = random.randint(0, max(0, layer.width - 5))
        y1 = random.randint(0, max(0, layer.height - 5))
        x2 = min(layer.width, x1 + random.randint(10, max(15, layer.width // 2)))
        y2 = min(layer.height, y1 + random.randint(8, max(12, layer.height // 4)))
        color = (*rand_color(30, 230), random.randint(35, 105))
        if random.random() < 0.5:
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=color)
    return Image.alpha_composite(layer, overlay)


def draw_noisy_char(char: str, font_path: str) -> Image.Image:
    size = random.randint(58, 112)
    font = ImageFont.truetype(font_path, size)
    margin = int(size * 0.7)
    layer = Image.new("RGBA", (size + margin * 2, size + margin * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    outline_only = random.random() < 0.28
    fill = rand_color(30, 255)
    fill_alpha = random.randint(210, 255)
    if outline_only:
        fill = rand_color(20, 255)
        fill_alpha = random.randint(20, 80)
    stroke = rand_color(0, 220)
    shadow = rand_color(0, 150)
    stroke_width = random.randint(2, 10)
    if random.random() < 0.65:
        draw.text(
            (margin + random.randint(4, 10), margin + random.randint(4, 10)),
            char,
            font=font,
            fill=(*shadow, random.randint(80, 150)),
            stroke_width=max(1, stroke_width - 1),
            stroke_fill=(*shadow, random.randint(80, 150)),
        )
    draw.text(
        (margin, margin),
        char,
        font=font,
        fill=(*fill, fill_alpha),
        stroke_width=stroke_width,
        stroke_fill=(*stroke, random.randint(170, 255)),
    )

    if random.random() < 0.9:
        noise = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        nd = ImageDraw.Draw(noise)
        for _ in range(random.randint(36, 96)):
            x1 = random.randint(0, layer.width)
            y1 = random.randint(0, layer.height)
            x2 = x1 + random.randint(-42, 42)
            y2 = y1 + random.randint(-42, 42)
            nd.line(
                [x1, y1, x2, y2],
                fill=(*rand_color(), random.randint(45, 170)),
                width=random.randint(1, 4),
            )
        layer = Image.alpha_composite(layer, noise)

    if random.random() < 0.95:
        layer = add_pixel_speckles(layer)
    layer = add_translucent_occluders(layer)
    layer = add_local_blur(layer)
    if random.random() < 0.75:
        layer = shear_layer(layer)
    if random.random() < 0.7:
        layer = wave_distort(layer)
    layer = add_cutout(layer)
    angle = random.uniform(-45, 45)
    layer = layer.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    if random.random() < 0.65:
        layer = layer.filter(ImageFilter.GaussianBlur(random.uniform(0.35, 1.45)))
    return layer


def draw_icon(label: str) -> Image.Image:
    size = random.randint(64, 118)
    layer = Image.new("RGBA", (size + 24, size + 24), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    outline_style = random.random() < 0.35
    c1 = (*rand_color(40, 255), 0 if outline_style else 235)
    c2 = (*rand_color(0, 180), 230)
    s = size
    o = 12

    if label == "apple":
        draw.ellipse([o + 12, o + 20, o + s - 10, o + s], fill=c1, outline=c2, width=5)
        draw.ellipse([o + s * 0.48, o + 2, o + s * 0.8, o + 30], fill=(*rand_color(80, 220), 230))
        draw.line([o + s * 0.5, o + 18, o + s * 0.42, o + 2], fill=c2, width=5)
    elif label == "burger":
        draw.rounded_rectangle([o + 4, o + 18, o + s - 4, o + 45], radius=18, fill=c1, outline=c2, width=4)
        draw.rectangle([o + 8, o + 45, o + s - 8, o + 58], fill=(*rand_color(160, 255), 235))
        draw.rectangle([o + 8, o + 58, o + s - 8, o + 74], fill=(*rand_color(30, 180), 235))
        draw.rounded_rectangle([o + 4, o + 74, o + s - 4, o + 94], radius=12, fill=c1, outline=c2, width=4)
    elif label == "star":
        pts = []
        cx, cy = o + s / 2, o + s / 2
        for i in range(10):
            r = s * (0.45 if i % 2 == 0 else 0.2)
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        draw.polygon(pts, fill=c1, outline=c2)
    elif label == "bolt":
        pts = [(o + s * 0.58, o), (o + s * 0.25, o + s * 0.55), (o + s * 0.52, o + s * 0.55), (o + s * 0.35, o + s), (o + s * 0.78, o + s * 0.38), (o + s * 0.52, o + s * 0.38)]
        draw.polygon(pts, fill=c1, outline=c2)
    elif label == "shield":
        pts = [(o + s * 0.5, o), (o + s * 0.9, o + s * 0.18), (o + s * 0.8, o + s * 0.72), (o + s * 0.5, o + s), (o + s * 0.2, o + s * 0.72), (o + s * 0.1, o + s * 0.18)]
        draw.polygon(pts, fill=c1, outline=c2)
    elif label == "heart":
        pts = [(o + s * 0.5, o + s * 0.9), (o + s * 0.1, o + s * 0.45), (o + s * 0.2, o + s * 0.18), (o + s * 0.48, o + s * 0.25), (o + s * 0.52, o + s * 0.25), (o + s * 0.8, o + s * 0.18), (o + s * 0.9, o + s * 0.45)]
        draw.polygon(pts, fill=c1, outline=c2)
    elif label == "triangle":
        draw.polygon([(o + s * 0.5, o), (o, o + s), (o + s, o + s)], fill=c1, outline=c2)
    elif label == "square":
        draw.rounded_rectangle([o + 5, o + 5, o + s - 5, o + s - 5], radius=10, fill=c1, outline=c2, width=5)
    elif label == "diamond":
        draw.polygon([(o + s * 0.5, o), (o + s, o + s * 0.5), (o + s * 0.5, o + s), (o, o + s * 0.5)], fill=c1, outline=c2)
    elif label == "moon":
        draw.ellipse([o, o, o + s, o + s], fill=c1, outline=c2, width=4)
        draw.ellipse([o + s * 0.28, o - 2, o + s * 1.08, o + s * 0.9], fill=(0, 0, 0, 0), outline=None)
    elif label == "sun":
        cx, cy = o + s / 2, o + s / 2
        for i in range(12):
            a = i * math.tau / 12
            draw.line([cx, cy, cx + math.cos(a) * s * 0.55, cy + math.sin(a) * s * 0.55], fill=c2, width=5)
        draw.ellipse([o + s * 0.2, o + s * 0.2, o + s * 0.8, o + s * 0.8], fill=c1, outline=c2, width=4)
    elif label == "cloud":
        draw.ellipse([o, o + s * 0.35, o + s * 0.45, o + s * 0.85], fill=c1, outline=c2, width=4)
        draw.ellipse([o + s * 0.25, o + s * 0.15, o + s * 0.75, o + s * 0.75], fill=c1, outline=c2, width=4)
        draw.ellipse([o + s * 0.55, o + s * 0.35, o + s, o + s * 0.85], fill=c1, outline=c2, width=4)
    elif label == "umbrella":
        draw.pieslice([o, o, o + s, o + s], 180, 360, fill=c1, outline=c2, width=4)
        draw.line([o + s * 0.5, o + s * 0.5, o + s * 0.5, o + s], fill=c2, width=5)
        draw.arc([o + s * 0.35, o + s * 0.75, o + s * 0.68, o + s * 1.1], 0, 180, fill=c2, width=5)
    elif label == "flag":
        draw.line([o + s * 0.2, o, o + s * 0.2, o + s], fill=c2, width=6)
        draw.polygon([(o + s * 0.22, o + 8), (o + s * 0.9, o + s * 0.2), (o + s * 0.22, o + s * 0.45)], fill=c1, outline=c2)
    elif label == "cross":
        draw.rounded_rectangle([o + s * 0.4, o, o + s * 0.6, o + s], radius=6, fill=c1)
        draw.rounded_rectangle([o, o + s * 0.4, o + s, o + s * 0.6], radius=6, fill=c1)
        draw.rounded_rectangle([o + s * 0.4, o, o + s * 0.6, o + s], radius=6, outline=c2, width=4)
        draw.rounded_rectangle([o, o + s * 0.4, o + s, o + s * 0.6], radius=6, outline=c2, width=4)
    elif label == "target":
        for scale in [1.0, 0.72, 0.42, 0.16]:
            pad = s * (1 - scale) / 2
            draw.ellipse([o + pad, o + pad, o + s - pad, o + s - pad], outline=c2, width=5)
        draw.line([o + s * 0.5, o + s * 0.15, o + s * 0.5, o + s * 0.85], fill=c1 if not outline_style else c2, width=4)
        draw.line([o + s * 0.15, o + s * 0.5, o + s * 0.85, o + s * 0.5], fill=c1 if not outline_style else c2, width=4)
    elif label == "coin":
        draw.ellipse([o, o, o + s, o + s], fill=c1, outline=c2, width=7)
        pts = []
        cx, cy = o + s / 2, o + s / 2
        for i in range(8):
            r = s * (0.27 if i % 2 == 0 else 0.12)
            a = -math.pi / 2 + i * math.pi / 4
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        draw.polygon(pts, fill=(*rand_color(80, 255), 210), outline=c2)
    elif label == "speaker":
        draw.polygon([(o + s * 0.1, o + s * 0.38), (o + s * 0.32, o + s * 0.38), (o + s * 0.55, o + s * 0.18), (o + s * 0.55, o + s * 0.82), (o + s * 0.32, o + s * 0.62), (o + s * 0.1, o + s * 0.62)], fill=c1, outline=c2)
        for i in range(3):
            draw.arc([o + s * (0.52 + i * 0.06), o + s * (0.25 - i * 0.07), o + s * (0.9 + i * 0.08), o + s * (0.75 + i * 0.07)], -45, 45, fill=c2, width=5)
    elif label == "exclamation":
        draw.rounded_rectangle([o + s * 0.42, o, o + s * 0.62, o + s * 0.68], radius=8, fill=c1 if not outline_style else None, outline=c2, width=5)
        draw.ellipse([o + s * 0.38, o + s * 0.78, o + s * 0.66, o + s], fill=c1 if not outline_style else None, outline=c2, width=5)
    elif label == "ring":
        draw.ellipse([o, o, o + s, o + s], outline=c2, width=random.randint(8, 14))
        if random.random() < 0.5:
            draw.ellipse([o + s * 0.22, o + s * 0.22, o + s * 0.78, o + s * 0.78], outline=(*rand_color(40, 255), 210), width=5)
    elif label == "gear":
        cx, cy = o + s / 2, o + s / 2
        pts = []
        for i in range(20):
            r = s * (0.46 if i % 2 == 0 else 0.35)
            a = i * math.tau / 20
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        draw.polygon(pts, fill=c1, outline=c2)
        draw.ellipse([o + s * 0.32, o + s * 0.32, o + s * 0.68, o + s * 0.68], fill=(0, 0, 0, 0), outline=c2, width=5)
    elif label == "arrow":
        draw.line([o + s * 0.1, o + s * 0.5, o + s * 0.82, o + s * 0.5], fill=c2, width=10)
        draw.polygon([(o + s * 0.62, o + s * 0.22), (o + s * 0.95, o + s * 0.5), (o + s * 0.62, o + s * 0.78)], fill=c1 if not outline_style else c2, outline=c2)
    elif label == "check":
        draw.line([o + s * 0.15, o + s * 0.55, o + s * 0.38, o + s * 0.78, o + s * 0.88, o + s * 0.22], fill=c2, width=12)
    elif label == "question":
        draw.arc([o + s * 0.2, o, o + s * 0.82, o + s * 0.62], 190, 45, fill=c2, width=9)
        draw.line([o + s * 0.55, o + s * 0.58, o + s * 0.48, o + s * 0.72], fill=c2, width=9)
        draw.ellipse([o + s * 0.42, o + s * 0.82, o + s * 0.58, o + s * 0.98], fill=c2)
    elif label == "play":
        draw.polygon([(o + s * 0.25, o + s * 0.12), (o + s * 0.25, o + s * 0.88), (o + s * 0.88, o + s * 0.5)], fill=c1, outline=c2)
    elif label == "pause":
        draw.rounded_rectangle([o + s * 0.22, o + s * 0.12, o + s * 0.4, o + s * 0.88], radius=5, fill=c1, outline=c2, width=4)
        draw.rounded_rectangle([o + s * 0.6, o + s * 0.12, o + s * 0.78, o + s * 0.88], radius=5, fill=c1, outline=c2, width=4)
    elif label == "plus":
        draw.line([o + s * 0.5, o + s * 0.15, o + s * 0.5, o + s * 0.85], fill=c2, width=12)
        draw.line([o + s * 0.15, o + s * 0.5, o + s * 0.85, o + s * 0.5], fill=c2, width=12)
    elif label == "minus":
        draw.line([o + s * 0.15, o + s * 0.5, o + s * 0.85, o + s * 0.5], fill=c2, width=13)
    elif label == "droplet":
        pts = [(o + s * 0.5, o), (o + s * 0.85, o + s * 0.48), (o + s * 0.68, o + s * 0.92), (o + s * 0.32, o + s * 0.92), (o + s * 0.15, o + s * 0.48)]
        draw.polygon(pts, fill=c1, outline=c2)
    elif label == "leaf":
        draw.ellipse([o + s * 0.12, o + s * 0.12, o + s * 0.9, o + s * 0.78], fill=c1, outline=c2, width=5)
        draw.line([o + s * 0.2, o + s * 0.75, o + s * 0.82, o + s * 0.22], fill=c2, width=4)
    elif label == "location":
        draw.ellipse([o + s * 0.25, o, o + s * 0.75, o + s * 0.5], fill=c1, outline=c2, width=5)
        draw.polygon([(o + s * 0.5, o + s), (o + s * 0.28, o + s * 0.42), (o + s * 0.72, o + s * 0.42)], fill=c1, outline=c2)
        draw.ellipse([o + s * 0.42, o + s * 0.18, o + s * 0.58, o + s * 0.34], fill=c2)
    elif label == "home":
        draw.polygon([(o + s * 0.1, o + s * 0.45), (o + s * 0.5, o + s * 0.08), (o + s * 0.9, o + s * 0.45)], fill=c1, outline=c2)
        draw.rectangle([o + s * 0.22, o + s * 0.45, o + s * 0.78, o + s * 0.9], fill=c1, outline=c2, width=5)
    elif label == "key":
        draw.ellipse([o + s * 0.08, o + s * 0.28, o + s * 0.42, o + s * 0.62], fill=c1, outline=c2, width=5)
        draw.line([o + s * 0.4, o + s * 0.45, o + s * 0.92, o + s * 0.45], fill=c2, width=8)
        draw.line([o + s * 0.76, o + s * 0.45, o + s * 0.76, o + s * 0.62], fill=c2, width=6)
    elif label == "lock":
        draw.rounded_rectangle([o + s * 0.18, o + s * 0.42, o + s * 0.82, o + s * 0.92], radius=8, fill=c1, outline=c2, width=5)
        draw.arc([o + s * 0.28, o + s * 0.05, o + s * 0.72, o + s * 0.58], 180, 360, fill=c2, width=7)
    elif label == "bell":
        draw.pieslice([o + s * 0.2, o + s * 0.12, o + s * 0.8, o + s * 0.9], 180, 360, fill=c1, outline=c2, width=5)
        draw.rectangle([o + s * 0.22, o + s * 0.48, o + s * 0.78, o + s * 0.82], fill=c1, outline=c2, width=5)
        draw.ellipse([o + s * 0.42, o + s * 0.82, o + s * 0.58, o + s * 0.98], fill=c2)
    elif label == "camera":
        draw.rounded_rectangle([o + s * 0.1, o + s * 0.25, o + s * 0.9, o + s * 0.82], radius=10, fill=c1, outline=c2, width=5)
        draw.ellipse([o + s * 0.36, o + s * 0.38, o + s * 0.64, o + s * 0.66], fill=(0, 0, 0, 0), outline=c2, width=5)
    elif label == "book":
        draw.rounded_rectangle([o + s * 0.12, o + s * 0.12, o + s * 0.88, o + s * 0.9], radius=7, fill=c1, outline=c2, width=5)
        draw.line([o + s * 0.5, o + s * 0.16, o + s * 0.5, o + s * 0.88], fill=c2, width=4)
    elif label == "clock":
        draw.ellipse([o + s * 0.08, o + s * 0.08, o + s * 0.92, o + s * 0.92], fill=c1, outline=c2, width=6)
        draw.line([o + s * 0.5, o + s * 0.5, o + s * 0.5, o + s * 0.24], fill=c2, width=5)
        draw.line([o + s * 0.5, o + s * 0.5, o + s * 0.72, o + s * 0.58], fill=c2, width=5)
    elif label == "crown":
        draw.polygon([(o + s * 0.08, o + s * 0.8), (o + s * 0.18, o + s * 0.25), (o + s * 0.38, o + s * 0.58), (o + s * 0.5, o + s * 0.18), (o + s * 0.62, o + s * 0.58), (o + s * 0.82, o + s * 0.25), (o + s * 0.92, o + s * 0.8)], fill=c1, outline=c2)
    else:
        draw.ellipse([o, o, o + s, o + s], fill=c1, outline=c2, width=6)

    if random.random() < 0.55:
        layer = add_pixel_speckles(layer)
    if random.random() < 0.4:
        layer = add_translucent_occluders(layer)
    layer = layer.rotate(random.uniform(-35, 35), resample=Image.Resampling.BICUBIC, expand=True)
    if random.random() < 0.45:
        layer = layer.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.0)))
    return layer


def build_layout_slots(width: int, height: int, count: int) -> list[list[int]]:
    top = 82
    margin_x = 28
    margin_y = 18
    cols = 3 if count > 2 else count
    rows = math.ceil(count / cols)
    usable_w = width - margin_x * 2
    usable_h = height - top - margin_y
    slot_w = usable_w / cols
    slot_h = usable_h / rows
    slots: list[list[int]] = []
    for row in range(rows):
        for col in range(cols):
            x1 = int(margin_x + col * slot_w)
            y1 = int(top + row * slot_h)
            x2 = int(margin_x + (col + 1) * slot_w)
            y2 = int(top + (row + 1) * slot_h)
            slots.append([x1, y1, x2, y2])
    random.shuffle(slots)
    return slots[:count]


def fit_layer_to_slot(layer: Image.Image, slot: list[int]) -> Image.Image:
    max_w = max(48, slot[2] - slot[0] - 28)
    max_h = max(48, slot[3] - slot[1] - 24)
    if layer.width <= max_w and layer.height <= max_h:
        return layer
    fitted = layer.copy()
    fitted.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return fitted


def place_in_slot(layer: Image.Image, slot: list[int]) -> tuple[int, int]:
    layer = fit_layer_to_slot(layer, slot)
    slot_w = slot[2] - slot[0]
    slot_h = slot[3] - slot[1]
    jitter_x = max(0, min(14, (slot_w - layer.width) // 3))
    jitter_y = max(0, min(12, (slot_h - layer.height) // 3))
    x = slot[0] + max(0, (slot_w - layer.width) // 2) + random.randint(-jitter_x, jitter_x)
    y = slot[1] + max(0, (slot_h - layer.height) // 2) + random.randint(-jitter_y, jitter_y)
    return max(4, x), max(60, y)


def place_non_overlapping(width: int, height: int, layer: Image.Image, used: list[list[int]]) -> tuple[int, int]:
    actual = alpha_bbox(layer) or [0, 0, layer.width, layer.height]
    min_gap = random.randint(18, 34)
    for _ in range(180):
        x = random.randint(8, max(8, width - layer.width - 8))
        y = random.randint(54, max(54, height - layer.height - 8))
        box = [x + actual[0], y + actual[1], x + actual[2], y + actual[3]]
        padded = expanded(box, min_gap)
        ok = True
        for old in used:
            hard_inter = intersection_area(box, old)
            soft_inter = intersection_area(padded, old)
            if hard_inter > 0.02 * min(bbox_area(box), bbox_area(old)) or soft_inter > 0:
                ok = False
                break
        if ok:
            return x, y
    return random.randint(8, max(8, width - layer.width - 8)), random.randint(54, max(54, height - layer.height - 8))


def draw_prompt(img: Image.Image, font_path: str, sequence: list[str]) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    prefix = random.choice(PROMPT_PREFIXES)
    text = prefix + "、".join(sequence)
    font_size = 26
    while font_size >= 17:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        if bbox[2] - bbox[0] <= img.width - 34:
            break
        font_size -= 1
    draw.text(
        (18, random.randint(12, 24)),
        text,
        font=font,
        fill=(248, 248, 248, 245),
        stroke_width=1,
        stroke_fill=(20, 20, 20, 190),
    )


def generate_one(index: int, out_dir: Path, font_path: str, width: int, height: int, background_paths: list[Path], chars: list[str]) -> tuple[dict, list[tuple[Path, str]], Image.Image]:
    img = choose_background(width, height, background_paths).convert("RGBA")
    labels: list[ObjectLabel] = []
    rec_rows: list[tuple[Path, str]] = []

    specs: list[tuple[str, str, str]] = []
    for char in random.sample(chars, random.randint(1, 4)):
        specs.append(("char", char, char))
    for icon in random.sample(ICONS, random.randint(1, 3)):
        specs.append(("icon", icon, ICON_DISPLAY[icon]))
    random.shuffle(specs)

    draw_prompt(img, font_path, [display for _, _, display in specs])
    slots = build_layout_slots(width, height, len(specs))

    for order, (label_type, label, display) in enumerate(specs):
        if label_type == "char":
            layer = draw_noisy_char(label, font_path)
        else:
            layer = draw_icon(label)
        layer = fit_layer_to_slot(layer, slots[order])
        x, y = place_in_slot(layer, slots[order])
        bbox = paste_layer(img, layer, x, y)
        labels.append(ObjectLabel(label_type, label, bbox))

        if label_type == "char":
            crop = img.crop(tuple(bbox)).convert("RGB")
            crop_path = out_dir / "char_rec" / f"sample_{index:05d}_{len(rec_rows):02d}.png"
            crop.save(crop_path)
            rec_rows.append((crop_path, label))

    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 0.5)))

    image_path = out_dir / "images" / f"sample_{index:05d}.png"
    img.convert("RGB").save(image_path)

    image_info = {
        "id": index,
        "file_name": image_path.name,
        "width": width,
        "height": height,
        "objects": [label.__dict__ for label in labels],
        "target_sequence": [
            {"label_type": label_type, "label": label, "display": display}
            for label_type, label, display in specs
        ],
    }
    return image_info, rec_rows, img.convert("RGB")


def save_coco(samples: list[dict], out_dir: Path) -> None:
    categories = [
        {"id": 1, "name": "char", "supercategory": "target"},
        {"id": 2, "name": "icon", "supercategory": "target"},
    ]
    images = []
    annotations = []
    ann_id = 1
    for sample in samples:
        images.append(
            {
                "id": sample["id"],
                "file_name": sample["file_name"],
                "width": sample["width"],
                "height": sample["height"],
            }
        )
        for obj in sample["objects"]:
            x1, y1, x2, y2 = obj["bbox"]
            w, h = x2 - x1, y2 - y1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": sample["id"],
                    "category_id": 1 if obj["label_type"] == "char" else 2,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "label": obj["label"],
                    "label_type": obj["label_type"],
                }
            )
            ann_id += 1
    coco = {"images": images, "annotations": annotations, "categories": categories}
    (out_dir / "labels_coco.json").write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


def make_preview(images: list[Image.Image], out_dir: Path, cols: int = 4) -> None:
    if not images:
        return
    thumb_w, thumb_h = 320, 220
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (35, 35, 35))
    for i, img in enumerate(images):
        thumb = img.copy()
        thumb.thumbnail((thumb_w - 12, thumb_h - 12))
        x = (i % cols) * thumb_w + 6
        y = (i // cols) * thumb_h + 6
        canvas.paste(thumb, (x, y))
    canvas.save(out_dir / "preview_grid.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--background-dir", default=str(DEFAULT_BACKGROUNDS))
    parser.add_argument("--char-mode", choices=["base", "dict"], default="dict")
    parser.add_argument("--char-limit", type=int, default=1200)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output)
    for sub in ["images", "char_rec"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    font_path = find_font()
    background_paths = list_backgrounds(Path(args.background_dir))
    chars = load_chars(args.char_mode, args.char_limit, args.seed)
    samples: list[dict] = []
    rec_rows: list[tuple[Path, str]] = []
    preview_images: list[Image.Image] = []
    samples_jsonl_path = out_dir / "samples.jsonl"
    rec_label_path = out_dir / "char_rec_labels.txt"
    samples_jsonl_path.write_text("", encoding="utf-8")
    rec_label_path.write_text("", encoding="utf-8")

    iterator = range(1, args.count + 1)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="synthetic", unit="img")

    for index in iterator:
        sample, rows, img = generate_one(index, out_dir, font_path, args.width, args.height, background_paths, chars)
        samples.append(sample)
        rec_rows.extend(rows)
        with samples_jsonl_path.open("a", encoding="utf-8") as fin:
            fin.write(json.dumps(sample, ensure_ascii=False) + "\n")
        if rows:
            with rec_label_path.open("a", encoding="utf-8") as fin:
                for path, label in rows:
                    fin.write(f"{path.resolve().as_posix()}\t{label}\n")
        if len(preview_images) < 16:
            preview_images.append(img)

    save_coco(samples, out_dir)
    make_preview(preview_images, out_dir)

    print(f"generated_images={len(samples)}")
    print(f"generated_char_crops={len(rec_rows)}")
    print(f"output={out_dir.resolve()}")
    print(f"preview={out_dir.resolve() / 'preview_grid.png'}")
    print(f"coco={out_dir.resolve() / 'labels_coco.json'}")
    print(f"rec_labels={rec_label_path.resolve()}")
    print(f"backgrounds={len(background_paths)}")
    print(f"chars={len(chars)}")


if __name__ == "__main__":
    main()
