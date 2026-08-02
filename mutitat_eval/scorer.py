#!/usr/bin/env python3
# =============================================================================
# multitat_eval/scorer.py — self-contained MULTITAT scorer (TAT-QA EM + F1).
#
# Drop this whole `multitat_eval/` directory into any project. It has no
# third-party dependencies (pure Python standard library) and no ties to any
# host project: `download.py` fetches the data next to this file, and this module
# both exposes a small API (score one task at a time) and a CLI (score a file).
#
# DEFAULTS FOLLOW MULTITAT'S ORIGINAL SETTINGS:
#   • metric  = Exact-Match + numeracy token-F1 (the DROP metric TAT-QA uses;
#               for arithmetic & count answers F1 is set equal to EM).
#   • extract = MULTITAT's own `extract_answer`: the text after the LAST
#               ':' / '：' / ':\n' in the model output (its prompts end "…Answer: X").
#   • scale   = none (MULTITAT gold has no scale field; units live in the answer).
#
# -----------------------------------------------------------------------------
# WHAT IS THE "id" (task id)?   — for anyone who re-downloads the data
# -----------------------------------------------------------------------------
# MULTITAT ships one file per language: dataset/dev.<lang>.json
#   (11 langs: bn de en es fr ja ru sw te th zh), 250 items each.
#   Source: https://github.com/zhxlia/MULTITAT  (dataset/dev.<lang>.json)
#   Run `python download.py` (in this directory) to fetch them into ./data/.
#
# Each item's id is  source.qid  (a TAT-QA-dev UUID), e.g.:
#   { "source": { "qid": "acde36c6-…", "answer_from": "table|hybrid|text",
#                 "answer_type": "arithmetic|span|count" },
#     "text": {...}, "table": {...}, "question": "...", "answer": <num|str|list> }
#
# A qid is unique WITHIN a language but the SAME qid repeats across ALL 11
# languages (same question translated): numeric answers are identical across
# languages, text/span answers are localized. So always score a run against the
# gold of the SAME language the outputs were produced for — pass that language
# (`lang=` / `--lang`), or a single-language gold file.
# -----------------------------------------------------------------------------
import argparse
import glob
import json
import os
import re
import string
from itertools import permutations

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")          # populated by download.py

# ── DROP-style normalization + EM/F1 (the TAT-QA metric), self-contained ─────
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _clean_numeric_token(text: str) -> str:
    token = str(text).strip().strip("`")
    if token.endswith("%") and _is_number(token[:-1]):
        return token[:-1]
    return token


def _normalize(answer: str) -> str:
    # Split on spaces only (matches TAT-QA's `tokenize()`), NOT hyphens: splitting
    # on '-' too (as the original DROP eval script does, but TAT-QA deliberately
    # doesn't) strips the sign off lone negative numbers like "-19.17" -> "19.17",
    # which would silently accept sign-flipped wrong answers as correct.
    tokens = []
    for tok in re.split(r" ", str(answer).lower()):
        tok = _clean_numeric_token(tok)
        tok = tok if _is_number(tok) else "".join(c for c in tok if c not in _PUNCT)
        # TAT-QA's get_answer_str() rounds numeric answers to 2 decimals (applied to
        # both gold and prediction) before comparing; without this, an arithmetically
        # correct but unrounded prediction (e.g. -19.168591224018478) fails to match
        # gold's conventional 2dp precision (-19.17).
        tok = str(round(float(tok), 2)) if _is_number(tok) else tok
        tok = " ".join(_ARTICLES.sub(" ", tok).split())
        if tok.strip():
            tokens.append(tok)
    return " ".join(tokens).strip()


def _answer_to_bags(answer):
    spans = answer if isinstance(answer, (list, tuple)) else [answer]
    norm, bags = [], []
    for s in spans:
        n = _normalize(s)
        norm.append(n)
        bags.append(set(n.split()))
    return norm, bags


def _f1(pred_bag: set, gold_bag: set) -> float:
    inter = len(gold_bag & pred_bag)
    prec = 1.0 if not pred_bag else inter / len(pred_bag)
    rec = 1.0 if not gold_bag else inter / len(gold_bag)
    return 0.0 if prec == 0 and rec == 0 else (2 * prec * rec) / (prec + rec)


def _numbers_match(gold_bag: set, pred_bag: set) -> bool:
    g = {w for w in gold_bag if _is_number(w)}
    p = {w for w in pred_bag if _is_number(w)}
    return (not g) or bool(g & p)


def _best_alignment(pred_bags, gold_bags) -> float:
    n = max(len(pred_bags), len(gold_bags))
    if n == 0:
        return 1.0
    pb = pred_bags + [set()] * (n - len(pred_bags))
    gb = gold_bags + [set()] * (n - len(gold_bags))
    best = 0.0
    for perm in permutations(range(n)):
        total = sum(_f1(pb[pi], gb[gi]) for gi, pi in enumerate(perm) if _numbers_match(gb[gi], pb[pi]))
        best = max(best, total / n)
    return best


def em_f1(prediction, gold) -> tuple:
    """DROP Exact-Match + token-F1 for a (possibly multi-span) prediction/gold."""
    pred_spans, pred_bags = _answer_to_bags(prediction)
    gold_spans, gold_bags = _answer_to_bags(gold)
    em = 1.0 if (len(pred_spans) == len(gold_spans) and set(pred_spans) == set(gold_spans)) else 0.0
    return em, round(_best_alignment(pred_bags, gold_bags), 2)


# ── answer extraction from raw model output ──────────────────────────────────
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d)")
_MULTITAT_TAGS = (": ", "：", ":\n")   # inference_CoT.extract_answer delimiters


def extract_prediction(output: str, mode: str = "multitat"):
    """Pull the answer from raw LLM text.
      multitat (default) : text after the last ':'/'：'/':\\n' (MULTITAT native)
      answer-tag         : the last <answer>[...]</answer>, list-split
      raw                : the whole output string
    """
    text = (output or "").strip()
    if mode == "raw":
        return text
    if mode == "answer-tag":
        tags = _ANSWER_TAG.findall(text)
        if tags:
            c = tags[-1].strip()
            inner = c[1:-1] if c.startswith("[") and c.endswith("]") else c
            parts = [p.strip() for p in _THOUSANDS.sub("", inner).split(",") if p.strip()]
            return parts if len(parts) > 1 else (parts[0] if parts else "")
        return text
    # multitat
    for tag in _MULTITAT_TAGS:
        if tag in text:
            return text.rsplit(tag)[-1].strip()
    return text


def _gold_answer(item):
    a = item.get("answer")
    return [str(x) for x in a] if isinstance(a, list) else str(a)


def _maybe_split_prediction_for_gold(prediction, gold_answer):
    """Split string predictions only when gold is a multi-span/list answer.

    Single-answer golds and non-string predictions are intentionally left
    unchanged, so numeric answers such as "1,234" keep their existing behavior.
    """
    if not isinstance(gold_answer, list) or not isinstance(prediction, str):
        return prediction

    text = prediction.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    # Protect thousands separators before splitting on answer separators.
    text = _THOUSANDS.sub("", text)
    if ";" in text:
        parts = [p.strip() for p in text.split(";") if p.strip()]
    elif "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
    else:
        return prediction

    return parts if len(parts) == len(gold_answer) else prediction


def _percent_variants(pred, gold):
    gold_str = " ".join(gold) if isinstance(gold, list) else str(gold)
    if "%" not in gold_str:
        return [pred]
    def add(p):
        if isinstance(p, list):
            return [x if x.strip().endswith("%") else x + "%" for x in p]
        return p if str(p).strip().endswith("%") else str(p) + "%"
    return [pred, add(pred)]


def score_prediction(prediction, gold_answer, answer_type):
    """EM/F1 for an already-extracted prediction vs gold. F1=EM for arithmetic/count."""
    best_em, best_f1 = 0.0, 0.0
    for cand in _percent_variants(prediction, gold_answer):
        em, f1 = em_f1(cand, gold_answer)
        if (em, f1) > (best_em, best_f1):
            best_em, best_f1 = em, f1
    if answer_type in ("arithmetic", "count"):
        best_f1 = best_em
    return best_em, best_f1


# ── Scorer: loads gold once, scores tasks by id ──────────────────────────────
class MultiTATScorer:
    """Load MULTITAT gold once, then score per task (by id) or in bulk.

    scorer = MultiTATScorer(lang="en")            # single language
    scorer = MultiTATScorer()                      # all languages in data/
    r = scorer.score_task(qid, output, lang="en")  # -> {em, f1, answer_type, ...}
    """

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        lang: str = None,
        extract: str = "multitat",
        fallback_langs=None,
    ):
        self.extract = extract
        self.fallback_langs = list(fallback_langs or [])
        if lang:
            langs = [lang] + [lg for lg in self.fallback_langs if lg != lang]
            files = [os.path.join(data_dir, f"dev.{lg}.json") for lg in langs]
        else:
            files = sorted(glob.glob(os.path.join(data_dir, "dev.*.json")))
        files = [f for f in files if os.path.exists(f)]
        if not files:
            raise FileNotFoundError(f"No dev.*.json in {data_dir} (run download.py). lang={lang}")
        self._by_lang_qid, self._by_qid, self._dup = {}, {}, set()
        for path in files:
            m = re.search(r"dev\.([a-z]+)\.json$", os.path.basename(path))
            lg = m.group(1) if m else "?"
            for item in json.load(open(path, encoding="utf-8")):
                qid = item.get("source", {}).get("qid")
                if qid is None:
                    continue
                self._by_lang_qid[(lg, qid)] = item
                if qid in self._by_qid:
                    self._dup.add(qid)
                self._by_qid[qid] = item

    def _lookup(self, qid, lang):
        if lang is not None:
            return self._by_lang_qid.get((lang, qid))
        if qid in self._dup:
            raise ValueError(f"id {qid} exists in multiple languages; pass lang=")
        return self._by_qid.get(qid)

    def _score_item(self, item, qid, output, lang: str = None, extract: str = None) -> dict:
        at = item.get("source", {}).get("answer_type")
        af = item.get("source", {}).get("answer_from")
        gold = _gold_answer(item)
        pred = extract_prediction(output, mode=extract or self.extract)
        pred = _maybe_split_prediction_for_gold(pred, gold)
        em, f1 = score_prediction(pred, gold, at)
        return {"qid": qid, "lang": lang, "matched_lang": lang, "found": True,
                "em": em, "f1": f1, "answer_type": at, "answer_from": af,
                "prediction": pred, "gold": gold}

    def score_task(self, qid, output, lang: str = None, extract: str = None) -> dict:
        """Score ONE task. Returns a dict with em/f1 and metadata (or None gold if
        the id is unknown)."""
        candidate_langs = [lang]
        candidate_langs.extend(
            lg for lg in self.fallback_langs if lg and lg not in candidate_langs
        )

        best = None
        for candidate_lang in candidate_langs:
            item = self._lookup(qid, candidate_lang)
            if item is None:
                continue
            score = self._score_item(item, qid, output, candidate_lang, extract)
            if best is None or (score["em"], score["f1"]) > (best["em"], best["f1"]):
                best = score
            if score["em"] == 1.0:
                break

        if best is None:
            return {"qid": qid, "lang": lang, "matched_lang": None, "found": False,
                    "em": None, "f1": None, "answer_type": None, "answer_from": None}

        best["lang"] = lang
        return best

    def score_predictions(self, predictions, extract: str = None) -> dict:
        """Score an iterable of {id, output[, lang]} dicts; return aggregates."""
        agg = {"n": 0, "em": 0.0, "f1": 0.0}
        by_type, by_from = {}, {}
        misses = 0
        for p in predictions:
            r = self.score_task(p.get("id"), p.get("output", ""), p.get("lang"), extract)
            if not r["found"]:
                misses += 1
                continue
            agg["n"] += 1
            agg["em"] += r["em"]
            agg["f1"] += r["f1"]
            for d, k in ((by_type, r["answer_type"]), (by_from, r["answer_from"])):
                s = d.setdefault(k, {"n": 0, "em": 0.0, "f1": 0.0})
                s["n"] += 1
                s["em"] += r["em"]
                s["f1"] += r["f1"]

        def pct(s):
            return {"n": s["n"], "EM": round(100 * s["em"] / s["n"], 2) if s["n"] else None,
                    "F1": round(100 * s["f1"] / s["n"], 2) if s["n"] else None}
        return {"overall": pct(agg), "unmatched_ids": misses,
                "by_answer_type": {k: pct(v) for k, v in sorted(by_type.items(), key=lambda x: str(x[0]))},
                "by_answer_from": {k: pct(v) for k, v in sorted(by_from.items(), key=lambda x: str(x[0]))}}


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Score model outputs against MULTITAT with TAT-QA EM/F1. "
                    "Defaults to MULTITAT's original settings. See file header for the 'id'.")
    ap.add_argument("--predictions", required=True, help="JSONL of {id, output[, lang]}")
    ap.add_argument("--lang", help="score against this single language's gold (dev.<lang>.json)")
    ap.add_argument("--data-dir", default=DATA_DIR, help=f"dir with dev.*.json (default: {DATA_DIR})")
    ap.add_argument("--extract", choices=["multitat", "answer-tag", "raw"], default="multitat",
                    help="answer extraction: 'multitat' last-colon (default, native); "
                         "'answer-tag' <answer>[...]</answer>; 'raw' whole string.")
    ap.add_argument("--out", help="write full results JSON here")
    args = ap.parse_args()

    scorer = MultiTATScorer(data_dir=args.data_dir, lang=args.lang, extract=args.extract)
    preds = [json.loads(l) for l in open(args.predictions, encoding="utf-8") if l.strip()]
    res = scorer.score_predictions(preds)

    o = res["overall"]
    print(f"scored {o['n']} predictions ({res['unmatched_ids']} ids not found)  [extract={args.extract}]")
    print(f"  OVERALL   EM={o['EM']}  F1={o['F1']}")
    for axis in ("by_answer_type", "by_answer_from"):
        print(f"  {axis}:")
        for k, v in res[axis].items():
            print(f"    {str(k):<12} n={v['n']:<4} EM={v['EM']:<6} F1={v['F1']}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
