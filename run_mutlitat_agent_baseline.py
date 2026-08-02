#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Make root OWL modules and mutitat_eval's sibling imports resolvable whether
# this file is run from the repo root or from owl/.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "mutitat_eval"))

RESULTS_DIR = _HERE / "results" / "MultiTAT" / "baseline"
OTHER_METRIC_DIR = RESULTS_DIR / "other_metric_files"
DEFAULT_CONFIG_PATH = _HERE / "mutitat_eval" / "config_multitat.yaml"


def construct_agent_list() -> List[Dict[str, Any]]:
    from camel.models import ModelFactory
    from camel.toolkits import CodeExecutionToolkit, FunctionTool
    from camel.types import ModelPlatformType, ModelType
    from utils import OwlWorkforceChatAgent

    reasoning_model = ModelFactory.create(
       model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
               model_type="azure/Mistral-Large-3",
               api_key="anything",
               url="http://localhost:4567"
               )

    code_runner_toolkit = CodeExecutionToolkit(sandbox="subprocess", verbose=True)

    reasoning_coding_agent = OwlWorkforceChatAgent(
        (
            "You are a local reasoning and coding assistant. Solve the assigned "
            "task using only the task content and local files explicitly provided "
            "in the prompt. If the task involves JSON or JSONL data, write and "
            "execute Python code to load, inspect, filter, aggregate, and verify "
            "the data. If arithmetic, table reasoning, counting, comparison, or "
            "data transformation is needed, execute code rather than estimating. "
            "Return the answer concisely and do not use web search."
        ),
        model=reasoning_model,
        tools=[
            FunctionTool(code_runner_toolkit.execute_code),
        ],
    )

    return [
        {
            "name": "Reasoning Coding Agent",
            "description": (
                "A local-only reasoning and coding worker that can execute Python "
                "to read and analyze JSON/JSONL data, tables, and text included "
                "in the task. It has no internet, browser, search, multimodal, or "
                "document-processing tools."
            ),
            "agent": reasoning_coding_agent,
        }
    ]


def construct_workforce():
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType
    from utils import OwlGaiaWorkforce

    # coordinator assigned only to avoid error in framework but not used.
    coordinator_agent_kwargs = {
        "model": ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                model_type="azure/Mistral-Large-3",
                api_key="anything",
                url="http://localhost:4567"
                )
    }

    task_agent_kwargs = {
        "model": ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                model_type="azure/Mistral-Large-3",
                api_key="anything",
                url="http://localhost:4567"
                )
    }

    answerer_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="azure/Mistral-Large-3",
            api_key="anything",
            url="http://localhost:4567"
            )
    }

    workforce = OwlGaiaWorkforce(
        "MultiTAT Local Reasoning Workforce",
        task_agent_kwargs=task_agent_kwargs,
        coordinator_agent_kwargs=coordinator_agent_kwargs,
        answerer_agent_kwargs=answerer_agent_kwargs,
    )

    for agent_dict in construct_agent_list():
        workforce.add_single_agent_worker(
            agent_dict["description"],
            worker=agent_dict["agent"],
        )

    return workforce


def evaluate_on_multitat(param: Dict[str, Any]) -> None:
    from mutitat_eval.benchmark import DATA_DIR, MultiTATBenchmark
    from mutitat_eval.scorer import MultiTATScorer

    class TraceDirectoryMultiTATBenchmark(MultiTATBenchmark):
        @staticmethod
        def _trace_path(save_to: str) -> str:
            name = Path(save_to).name
            if name.endswith(".jsonl"):
                name = name[: -len(".jsonl")] + ".trace.json"
            else:
                name = name + ".trace.json"
            return str(RESULTS_DIR / "traces" / name)

    lang = param["lang"]
    max_tries = int(param.get("max_tries", 1))
    max_replanning_tries = int(param.get("max_replanning_tries", 1))
    subset = param.get("subset")
    idx = param.get("idx")
    answer_type = param.get("answer_type")
    answer_from = param.get("answer_from")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OTHER_METRIC_DIR.mkdir(parents=True, exist_ok=True)
    save_path = str(
        OTHER_METRIC_DIR / f"{lang}_pass{max_tries}_local_reasoning.jsonl"
    )

    benchmark = TraceDirectoryMultiTATBenchmark(save_to=save_path)
    workforce = construct_workforce()

    result = benchmark.run_workforce_with_retry(
        workforce,
        lang=lang,
        max_tries=max_tries,
        max_replanning_tries=max_replanning_tries,
        subset=subset,
        idx=idx,
        answer_type=answer_type,
        answer_from=answer_from,
        save_result=True,
    )

    metrics_path = save_path[: -len(".jsonl")] + ".metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    overall = result["overall"]
    logger.success(
        f"MULTITAT[{lang}] EM={overall['EM']} F1={overall['F1']} "
        f"(n={overall['n']}, unmatched={result['unmatched_ids']})"
    )
    logger.info(f"Predictions: {save_path}")
    logger.info(f"Metrics:     {metrics_path}")
    logger.info(f"Traces:      {benchmark._trace_path(save_path)}")

    predictions = [
        json.loads(line)
        for line in open(save_path, encoding="utf-8")
        if line.strip()
    ]
    standalone_result = MultiTATScorer(
        data_dir=str(DATA_DIR),
        lang=lang,
        fallback_langs=["en"] if lang != "en" else [],
    ).score_predictions(predictions)
    if standalone_result != result:
        logger.warning(
            "In-process summary diverged from standalone scorer.py rescoring "
            "of the same predictions file."
        )


def _load_config(setup: str, config_path: Path) -> Dict[str, Any]:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    selected_config = next(
        (cfg for cfg in config_data["configs"] if cfg["name"] == setup),
        None,
    )
    if not selected_config:
        raise ValueError(f"Configuration named '{setup}' not found in {config_path}")
    return selected_config


def _build_param_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.setup:
        param = _load_config(args.setup, args.config)
    else:
        param = {
            "name": f"multitat_{args.lang or 'en'}",
            "DOTENV_FILE": args.dotenv_file,
            "lang": args.lang or "en",
            "max_tries": args.max_tries,
            "subset": args.subset,
        }

    if args.lang:
        param["lang"] = args.lang
    if args.max_tries is not None:
        param["max_tries"] = args.max_tries
    if args.max_replanning_tries is not None:
        param["max_replanning_tries"] = args.max_replanning_tries
    if args.subset is not None:
        param["subset"] = args.subset
    if args.idx:
        param["idx"] = [int(i) for i in args.idx.split(",") if i.strip()]
    if args.answer_type:
        param["answer_type"] = args.answer_type
    if args.answer_from:
        param["answer_from"] = args.answer_from
    if args.dotenv_file:
        param["DOTENV_FILE"] = args.dotenv_file

    return param


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dotenv-file", default="../.env")
    parser.add_argument("--lang")
    parser.add_argument("--max-tries", type=int)
    parser.add_argument("--max-replanning-tries", type=int)
    parser.add_argument("--subset", type=int)
    parser.add_argument("--idx")
    parser.add_argument("--answer-type")
    parser.add_argument("--answer-from")
    args = parser.parse_args()

    selected_config = _build_param_from_args(args)

    dotenv_file = selected_config.get("DOTENV_FILE")
    if dotenv_file:
        dotenv_path = (_HERE / dotenv_file).resolve()
        logger.info(f"DOTENV_FILE from config: {dotenv_path}")
        load_dotenv(dotenv_path=str(dotenv_path), override=True)

    evaluate_on_multitat(selected_config)


if __name__ == "__main__":
    main()
