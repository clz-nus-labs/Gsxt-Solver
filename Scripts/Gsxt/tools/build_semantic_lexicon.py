from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for pattern in ("*.txt", "*.dict", "*.csv", "*.tsv", "*.json", "*.jsonl", "*.zip"):
        yield from sorted(path.rglob(pattern))


def iter_zip_lines(path: Path) -> Iterable[str]:
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if not name.lower().endswith((".txt", ".dict", ".csv", ".tsv", ".json", ".jsonl")):
                continue
            with zf.open(name) as handle:
                for raw_line in handle:
                    yield raw_line.decode("utf-8", errors="ignore")


def iter_plain_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        yield from handle


def iter_lines(path: Path) -> Iterable[str]:
    if path.suffix.lower() == ".zip":
        yield from iter_zip_lines(path)
    else:
        yield from iter_plain_lines(path)


def json_text_values(obj) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from json_text_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from json_text_values(value)


def extract_text(line: str, parse_json: bool) -> str:
    if parse_json:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return line
        return " ".join(json_text_values(obj))
    return line


def clean_phrase(token: str) -> str:
    return "".join(char for char in token.strip() if "\u4e00" <= char <= "\u9fff")


def add_word_list_line(counter: Counter[str], line: str, min_len: int, max_len: int, weight: int) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    # THUOCL / jieba userdict: word freq tag. CSV/TSV: word in first field.
    token = re.split(r"[\t,，\s]+", stripped, maxsplit=1)[0]
    phrase = clean_phrase(token)
    if min_len <= len(phrase) <= max_len:
        counter[phrase] += weight
        return True
    return False


def add_corpus_line(
    counter: Counter[str],
    line: str,
    min_len: int,
    max_len: int,
    max_sequence_len: int,
    use_jieba: bool,
) -> None:
    text = extract_text(line, parse_json=line.lstrip().startswith(("{", "[")))
    for seq in CHINESE_RE.findall(text):
        if len(seq) < min_len:
            continue
        seq = seq[:max_sequence_len]
        if use_jieba:
            try:
                import jieba  # type: ignore

                for word in jieba.cut(seq, HMM=True):
                    phrase = clean_phrase(word)
                    if min_len <= len(phrase) <= max_len:
                        counter[phrase] += 3
            except ImportError:
                pass
        for n in range(min_len, max_len + 1):
            if len(seq) < n:
                continue
            for index in range(0, len(seq) - n + 1):
                counter[seq[index : index + n]] += 1


def build_lexicon(args: argparse.Namespace) -> Counter[str]:
    counter: Counter[str] = Counter()
    files: list[Path] = []
    for source in args.source:
        files.extend(iter_files(Path(source)))

    iterator = files
    if tqdm is not None:
        iterator = tqdm(files, desc="semantic sources")

    for file_path in iterator:
        is_word_list = file_path.suffix.lower() in {".dict", ".csv", ".tsv"} or "thuocl" in file_path.name.lower()
        for line in iter_lines(file_path):
            if is_word_list and add_word_list_line(
                counter,
                line,
                min_len=args.min_len,
                max_len=args.max_len,
                weight=args.word_list_weight,
            ):
                continue
            add_corpus_line(
                counter,
                line,
                min_len=args.min_len,
                max_len=args.max_len,
                max_sequence_len=args.max_sequence_len,
                use_jieba=args.use_jieba,
            )
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight Chinese phrase lexicon for semantic ordering.")
    parser.add_argument("--source", action="append", required=True, help="Corpus/lexicon file, directory, or zip.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=200000)
    parser.add_argument("--max-sequence-len", type=int, default=120)
    parser.add_argument("--word-list-weight", type=int, default=1000)
    parser.add_argument("--use-jieba", action="store_true", help="Also segment corpus lines with jieba if installed.")
    args = parser.parse_args()

    counter = build_lexicon(args)
    rows = [
        (phrase, count)
        for phrase, count in counter.most_common()
        if count >= args.min_count and args.min_len <= len(phrase) <= args.max_len
    ][: args.top_k]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for phrase, count in rows:
            handle.write(f"{phrase}\t{count}\n")

    print(f"saved={output} phrases={len(rows)}")


if __name__ == "__main__":
    main()
