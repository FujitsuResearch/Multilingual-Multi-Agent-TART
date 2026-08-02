#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = ""  # Please write offical GAIA-MAPS repo here before start downloading
REPO_TYPE = "dataset"

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
DEFAULT_DATA_DIR = REPO_ROOT / "gaia_data" / "high_mid_lang_data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GAIA/MAPS data from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {REPO_ID}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Directory where data will be saved. Default: {DEFAULT_DATA_DIR}",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=REPO_TYPE,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            allow_patterns=[
                "README.md",
                ".gitattributes",
                "data/russian/gaia/**",
                "data/*/gaia_v2_lilt/**",
            ],
        )
    except Exception as exc:
        print(f"GAIA/MAPS download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    data_dir = output_dir / "data"
    original_data_dir = output_dir / "original_data"
    if data_dir.exists():
        if original_data_dir.exists():
            print(
                f"Cannot rename {data_dir} because {original_data_dir} already exists.",
                file=sys.stderr,
            )
            sys.exit(1)
        data_dir.rename(original_data_dir)

    for language_dir in original_data_dir.iterdir():
        if not language_dir.is_dir():
            continue

        old_gaia_dir = language_dir / "gaia"
        gaia_v2_lilt_dir = language_dir / "gaia_v2_lilt"
        renamed_gaia_dir = language_dir / "gaia"

        if language_dir.name == "russian" and old_gaia_dir.is_dir():
            if gaia_v2_lilt_dir.is_dir():
                shutil.rmtree(gaia_v2_lilt_dir)
        else:
            if old_gaia_dir.is_dir():
                shutil.rmtree(old_gaia_dir)

            if gaia_v2_lilt_dir.is_dir():
                gaia_v2_lilt_dir.rename(renamed_gaia_dir)

        target_json = renamed_gaia_dir / f"{language_dir.name}.json"
        for source_name in ("gaia-v2-lilt.json", "gaia.json"):
            source_json = renamed_gaia_dir / source_name
            if source_json.is_file() and not target_json.exists():
                source_json.rename(target_json)

    print(f"Downloaded GAIA/MAPS data into: {output_dir}")


if __name__ == "__main__":
    main()
