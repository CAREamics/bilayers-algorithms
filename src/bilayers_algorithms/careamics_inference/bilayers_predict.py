#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path


def whole_number(text: str) -> int:
    """Convert "64" or "64.0" to an int; reject anything else."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not value.is_integer():
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}")
    return int(value)


def optional_whole_number(text: str) -> int | None:
    text = text.strip()
    return whole_number(text) if text else None


def int_list(tokens: Sequence[str]) -> list[int] | None:
    """Accept ["256", "256"] and ["256 256"] alike; nothing given gives None."""
    values = " ".join(tokens).split()
    return [whole_number(value) for value in values] if values else None


def text_or_none(text: str) -> str | None:
    return text.strip() or None


def resolve_checkpoint(path: Path) -> Path:
    """Return the .ckpt file for a checkpoint file or a folder holding exactly one."""
    if path.is_file():
        if path.suffix == ".zip":
            raise SystemExit(f"BioImage Model Zoo models are not supported yet: {path}")
        return path
    if not path.is_dir():
        raise SystemExit(f"Checkpoint path does not exist: {path}")
    candidates = sorted(path.glob("*.ckpt"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        found = ", ".join(sorted(item.name for item in path.iterdir())) or "nothing"
        raise SystemExit(f"No .ckpt checkpoint found in {path}; found: {found}")
    names = ", ".join(candidate.name for candidate in candidates)
    raise SystemExit(f"Several .ckpt checkpoints found in {path}: {names}")


def build_predict_command(
    args: argparse.Namespace, checkpoint: Path, work_dir: Path
) -> list[str]:
    command = [
        "predict.py",
        "--ckpt_path",
        str(checkpoint),
        "--data",
        str(args.data),
        "--output_path",
        str(work_dir),
        "--data_type",
        "tiff",
        "--write_type",
        "tiff",
    ]
    if args.batch_size is not None:
        command += ["--batch_size", str(args.batch_size)]
    if args.tile_size is not None:
        command += ["--tile_size", *map(str, args.tile_size)]
    if args.tile_overlap is not None:
        command += ["--tile_overlap", *map(str, args.tile_overlap)]
    if args.axes is not None:
        command += ["--axes", args.axes]
    return command


def restored_name(path: Path, prefix: str = "") -> str:
    return f"{prefix}{path.stem}_restored{path.suffix}"


def flatten_outputs(src: Path, dst: Path) -> list[Path]:
    """Move all files under src flat into dst as <stem>_restored<extension>."""
    files = sorted(path for path in src.rglob("*") if path.is_file())
    if not files:
        raise SystemExit(f"Prediction produced no output files in {src}")
    plain_names = [restored_name(path) for path in files]
    clashing = {name for name, count in Counter(plain_names).items() if count > 1}
    names = [
        restored_name(
            path, "".join(f"{part}_" for part in path.relative_to(src).parent.parts)
        )
        if name in clashing
        else name
        for path, name in zip(files, plain_names, strict=True)
    ]
    if len(set(names)) != len(names):
        raise SystemExit(f"Prediction outputs in {src} have conflicting file names")
    dst.mkdir(parents=True, exist_ok=True)
    moved = []
    for path, name in zip(files, names, strict=True):
        target = dst / name
        target.unlink(missing_ok=True)
        shutil.move(path, target)
        moved.append(target)
    return moved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CAREamics prediction and write flat *_restored files."
    )
    parser.add_argument(
        "--ckpt_path",
        type=text_or_none,
        help="Path to a .ckpt file or a folder containing exactly one",
    )
    parser.add_argument(
        "--data", type=text_or_none, help="Path to a TIFF file or folder of TIFFs"
    )
    parser.add_argument(
        "--output_path", type=text_or_none, help="Folder for the restored images"
    )
    parser.add_argument("--batch_size", type=optional_whole_number, default=None)
    parser.add_argument(
        "--tile_size", nargs="*", default=[], help="2D or 3D, e.g. 256 256"
    )
    parser.add_argument(
        "--tile_overlap", nargs="*", default=[], help="2D or 3D, e.g. 48 48"
    )
    parser.add_argument("--axes", type=text_or_none, default=None)
    args = parser.parse_args(argv)

    for option in ("ckpt_path", "data", "output_path"):
        value = getattr(args, option)
        if value is None:
            parser.error(f"--{option} is required and must not be empty")
        setattr(args, option, Path(value))
    try:
        args.tile_size = int_list(args.tile_size)
        args.tile_overlap = int_list(args.tile_overlap)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint = resolve_checkpoint(args.ckpt_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        try:
            subprocess.run(
                build_predict_command(args, checkpoint, work_dir), check=True
            )
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"predict.py failed with exit code {error.returncode}"
            ) from error
        flatten_outputs(work_dir / "predictions", args.output_path)


if __name__ == "__main__":
    main()
