from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


EPOCH_RE = re.compile(r"epoch:\s*\[(\d+)/(\d+)\]")
BEST_EPOCH_RE = re.compile(r"best metric,.*best_epoch:\s*(\d+)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a training command and stop it when PaddleOCR best_epoch is stale.")
    parser.add_argument("--patience", type=int, required=True, help="Stop after this many epochs without a new best epoch.")
    parser.add_argument("--log-file", default="", help="Optional UTF-8 log file for the streamed output.")
    parser.add_argument("--cwd", default="", help="Working directory for the child process.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("Missing child command.")

    log_handle = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)

    current_epoch = 0
    best_epoch = 0
    stopped_by_early_stop = False

    process = subprocess.Popen(
        command,
        cwd=args.cwd or None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=os.environ.copy(),
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if log_handle:
                log_handle.write(line)

            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            best_match = BEST_EPOCH_RE.search(line)
            if best_match:
                best_epoch = int(best_match.group(1))
                stale_epochs = current_epoch - best_epoch
                if args.patience > 0 and current_epoch > 0 and stale_epochs >= args.patience:
                    message = (
                        f"\nEarly stopping: current_epoch={current_epoch}, "
                        f"best_epoch={best_epoch}, stale_epochs={stale_epochs}, patience={args.patience}\n"
                    )
                    print(message, end="")
                    if log_handle:
                        log_handle.write(message)
                    stopped_by_early_stop = True
                    process.terminate()
                    break

        return_code = process.wait(timeout=30)
    except KeyboardInterrupt:
        process.terminate()
        return process.wait()
    except subprocess.TimeoutExpired:
        process.kill()
        return_code = process.wait()
    finally:
        if log_handle:
            log_handle.close()

    if stopped_by_early_stop:
        return 0
    return return_code


if __name__ == "__main__":
    sys.exit(main())
