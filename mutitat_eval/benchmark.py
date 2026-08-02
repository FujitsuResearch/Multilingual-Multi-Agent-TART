#!/usr/bin/env python3
# =============================================================================
# mutitat_eval/benchmark.py — runs the OWL workforce over MULTITAT tasks.
#
# Loads MULTITAT gold (dev.<lang>.json, populated by download.py), builds one
# Task per item (question + table + paragraphs, flattened into the prompt),
# drives it through an OwlGaiaWorkforce, and writes predictions in the exact
# {"id", "output", "lang"} JSONL format scorer.py expects. scorer.py itself is
# imported read-only (for scoring, not modified) both to decide per-task retry
# success and to compute the final EM/F1 summary — so the predictions file this
# writes is always independently re-scorable via:
#   python mutitat_eval/scorer.py --predictions <file> --lang <lang>
# =============================================================================
import sys
import os

# Make `camel`/`utils` (the OWL agent code, one level up) importable regardless
# of the caller's cwd — same pattern utils/maps.py already uses.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from tqdm import tqdm

import camel.toolkits  # noqa: F401  (import order matters: resolves a
# circular import in camel.agents that otherwise breaks `camel.tasks` /
# `camel.societies.workforce` when they're the first camel.* import — same
# reason every run_*_workforce_*.py in this repo imports camel.toolkits first)
from camel.tasks import Task
from camel.societies.workforce import Workforce

from scorer import MultiTATScorer  # sibling file in mutitat_eval/

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = _HERE / "data"

ANSWER_FORMAT_INSTRUCTIONS = """
Do not use the web search. Now, answer the question based on the context, table and semantic representation.
""".strip()


class MultiTATBenchmark:
    r"""Runs the OWL workforce over MULTITAT (translated TAT-QA table-QA)
    tasks and writes predictions in the format `mutitat_eval/scorer.py`
    consumes.

    Args:
        data_dir (str, optional): Directory containing `dev.<lang>.json`
            gold files (default: `mutitat_eval/data`, populated by
            `download.py`).
        save_to (str): JSONL path predictions are appended to, one
            `{"id", "output", "lang"}` record per task.
        processes (int, optional): Unused placeholder kept for parity with
            `GAIABenchmark`. (default: :obj:`1`)
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        save_to: str = "",
        processes: int = 1,
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.save_to = save_to
        self.processes = processes
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._results: List[Dict[str, Any]] = []

    def load(self, lang: str = "en") -> "MultiTATBenchmark":
        r"""Load `dev.<lang>.json` into memory."""
        path = self.data_dir / f"dev.{lang}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python mutitat_eval/download.py "
                f"--langs {lang}` first."
            )
        with open(path, "r", encoding="utf-8") as f:
            self._data[lang] = json.load(f)
        return self

    def _load_tasks(
        self,
        lang: str = "en",
        randomize: bool = False,
        subset: Optional[int] = None,
        idx: Optional[List[int]] = None,
        answer_type: Optional[str] = None,
        answer_from: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.load(lang=lang)
        datas = self._data[lang]

        if answer_type:
            datas = [
                d for d in datas if d.get("source", {}).get("answer_type") == answer_type
            ]
        if answer_from:
            datas = [
                d for d in datas if d.get("source", {}).get("answer_from") == answer_from
            ]

        if randomize:
            random.shuffle(datas)
        if subset:
            datas = datas[:subset]
        if idx is not None and len(idx) != 0:
            datas = [datas[i] for i in idx]

        return datas

    def _format_context(self, item: Dict[str, Any]) -> str:
        r"""Flatten an item's paragraphs + table into prompt text."""
        parts = []

        paragraphs = (item.get("text") or {}).get("paragraph") or []
        if paragraphs:
            parts.append("Context:\n" + "\n".join(paragraphs))

        table = item.get("table") or {}
        rows = table.get("content") or []
        if rows:
            md_rows = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
            header_sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
            md_table = "\n".join([md_rows[0], header_sep] + md_rows[1:])
            title = table.get("title") or table.get("caption") or ""
            parts.append(f"Table:{(' ' + title) if title else ''}\n{md_table}")

        return "\n\n".join(parts)

    def _create_task(self, item: Dict[str, Any]) -> Task:
        qid = str(item["source"]["qid"])
        context = self._format_context(item)
        content = (
            f"{context}\n\nQuestion: {item['question']}\n\n{ANSWER_FORMAT_INSTRUCTIONS}"
        )
        return Task(id=qid, content=content)

    def _load_existing_predictions(self, path: str) -> List[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _append_prediction(self, path: str, record: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _trace_path(save_to: str) -> str:
        r"""Sibling of the compact JSONL holding the rich GAIA-format traces:
        `<...>.jsonl` -> `<...>.trace.json` (parallels how the runner derives
        `<...>.metrics.json`)."""
        if save_to.endswith(".jsonl"):
            return save_to[: -len(".jsonl")] + ".trace.json"
        return save_to + ".trace.json"

    def _load_existing_traces(self, path: str) -> List[Dict[str, Any]]:
        r"""Load a prior trace file (JSON array) so a resumed run preserves
        earlier tasks through the whole-file rewrite `_save_traces` does."""
        if not path or not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load existing traces from {path}: {e}")
            return []

    def _save_traces(self, path: str, results: List[Dict[str, Any]]) -> None:
        r"""Whole-file rewrite of the rich traces, mirroring
        `GAIABenchmark._save_results_to_file` (`utils/maps.py:167`):
        human-readable, and each per-task write leaves a valid partial."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    def run_workforce_with_retry(
        self,
        workforce: Workforce,
        lang: str = "en",
        max_tries: int = 3,
        max_replanning_tries: int = 1,
        randomize: bool = False,
        subset: Optional[int] = None,
        idx: Optional[List[int]] = None,
        answer_type: Optional[str] = None,
        answer_from: Optional[str] = None,
        save_result: bool = False,
    ) -> Dict[str, Any]:
        r"""Run the workforce over MULTITAT tasks with retry.

        Args:
            workforce (Workforce): The workforce to use for task processing.
            lang (str): MULTITAT language code (e.g. "en", "zh").
            max_tries (int): Maximum number of retries per task.
            max_replanning_tries (int): Maximum replanning tries per attempt.
            randomize (bool): Whether to shuffle task order.
            subset (Optional[int]): Number of tasks to run.
            idx (Optional[List[int]]): Specific task indices to run.
            answer_type (Optional[str]): Filter by MULTITAT `answer_type`
                (`arithmetic`/`span`/`count`).
            answer_from (Optional[str]): Filter by MULTITAT `answer_from`
                (`table`/`text`/`hybrid`).
            save_result (bool): Whether to append predictions to
                `self.save_to` (JSONL) and resume from it if it exists.

        Returns:
            Dict[str, Any]: The `MultiTATScorer.score_predictions` summary.
        """
        tasks = self._load_tasks(lang, randomize, subset, idx, answer_type, answer_from)
        logger.info(f"Number of MULTITAT[{lang}] tasks: {len(tasks)}")

        fallback_langs = ["en"] if lang != "en" else []
        scorer = MultiTATScorer(
            data_dir=str(self.data_dir),
            lang=lang,
            fallback_langs=fallback_langs,
        )

        # Compact JSONL drives task-skipping; the rich trace file is seeded
        # from disk so the whole-file rewrite in the loop preserves earlier
        # tasks on resume.
        existing = self._load_existing_predictions(self.save_to) if save_result else []
        done_ids = {r["id"] for r in existing}
        self._results = (
            self._load_existing_traces(self._trace_path(self.save_to))
            if save_result
            else []
        )

        for item in tqdm(tasks, desc=f"Running MULTITAT[{lang}]"):
            qid = str(item["source"]["qid"])
            if qid in done_ids:
                logger.success(f"Task {qid} already completed, skipping.")
                continue

            camel_task = self._create_task(item)

            success = False
            tries = 0
            raw_answer = None
            trajectory_with_retry: List[Dict[str, Any]] = []

            while not success and tries < max_tries:
                tries += 1
                logger.info(f"Attempt {tries}/{max_tries} for task {qid}")
                try:
                    if workforce.is_running():
                        workforce.stop()
                    processed_task = workforce.process_task(
                        camel_task, max_replanning_tries=max_replanning_tries
                    )
                    try:
                        raw_answer = workforce.get_workforce_final_answer(processed_task)
                    except Exception as e:
                        logger.error(f"Error extracting final answer for {qid}: {e}")
                        raw_answer = None

                    score = scorer.score_task(qid, raw_answer or "", lang=lang)
                    success = bool(score["found"] and score["em"] == 1.0)
                    trajectory_with_retry.append(
                        {
                            "attempts": tries,
                            "model_answer": raw_answer,
                            "ground_truth": item.get("answer"),
                            "success": success,
                            "em": score["em"],
                            "f1": score["f1"],
                            "trajectory": workforce.get_overall_task_solve_trajectory(),
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing task {qid} (attempt {tries}): {e}")

            # EM/F1 of the last scored attempt (0.0 if every attempt errored
            # out before reaching the scorer).
            last_scored = trajectory_with_retry[-1] if trajectory_with_retry else {}
            em = last_scored.get("em", 0.0)
            f1 = last_scored.get("f1", 0.0)

            record = {
                "id": qid,
                "output": raw_answer or "",
                "lang": lang,
                "em": em,
                "f1": f1,
                "score": bool(success),
            }
            self._results.append(
                {
                    "task_id": qid,
                    "question": item["question"],
                    "answer_type": item.get("source", {}).get("answer_type"),
                    "answer_from": item.get("source", {}).get("answer_from"),
                    "model_answer": raw_answer,
                    "ground_truth": item.get("answer"),
                    "score": bool(success),
                    "em": em,
                    "f1": f1,
                    "attempts": tries,
                    "trajectory": trajectory_with_retry,
                }
            )

            if save_result:
                self._append_prediction(self.save_to, record)
                self._save_traces(self._trace_path(self.save_to), self._results)

        return self._generate_summary(lang)

    def _generate_summary(self, lang: str) -> Dict[str, Any]:
        r"""Score every prediction written to `self.save_to` so far (covers
        both this run and any resumed-from-disk completions)."""
        fallback_langs = ["en"] if lang != "en" else []
        scorer = MultiTATScorer(
            data_dir=str(self.data_dir),
            lang=lang,
            fallback_langs=fallback_langs,
        )
        predictions = (
            self._load_existing_predictions(self.save_to)
            if self.save_to
            else [
                {"id": r["task_id"], "output": r.get("model_answer") or "", "lang": lang}
                for r in self._results
            ]
        )
        return scorer.score_predictions(predictions)
