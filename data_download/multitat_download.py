#!/usr/bin/env python3
# =============================================================================
# multitat_eval/download.py — fetch the RAW MULTITAT test data into ./data/.
#
# Pure standard library (urllib); no curl/git needed. Downloads the per-language
# dev.<lang>.json files from the MULTITAT repo so scorer.py can look up gold by
# task id (source.qid). See scorer.py's header for what the id means.
#
# Usage:
#   python download.py                       # all 11 languages -> ./data/
#   python download.py --langs en zh         # a subset
# =============================================================================
import argparse
import json
import os
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_HERE), "mutitat_eval", "data")
BASE_URL = "https://raw.githubusercontent.com/zhxlia/MULTITAT/main/dataset"
LANGS = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]


def main():
    ap = argparse.ArgumentParser(description="Download MULTITAT dev.<lang>.json into ./data/")
    ap.add_argument("--langs", nargs="*", default=LANGS, help="ISO codes (default: all 11)")
    ap.add_argument("--base-url", default=BASE_URL)
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    ok = 0
    for lang in args.langs:
        url = f"{args.base_url}/dev.{lang}.json"
        dst = os.path.join(DATA_DIR, f"dev.{lang}.json")
        try:
            with urllib.request.urlopen(url) as r:
                data = r.read()
            n = len(json.loads(data))          # validate it parses
            with open(dst, "wb") as f:
                f.write(data)
            print(f"  dev.{lang}.json  ({n} items)  <- {url}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  dev.{lang}.json  FAILED: {e}", file=sys.stderr)
    print(f"downloaded {ok}/{len(args.langs)} language files into {DATA_DIR}")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
    
# python data_download/multitat_download.py 
