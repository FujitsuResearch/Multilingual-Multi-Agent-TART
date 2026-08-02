#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from litellm import completion

SUPPORT_FILE_QUERY_SUFFIX = (
    ' Here are the necessary table files: gaia_data/low_resource_lang_data/support_files/{file_name}, '
    'for processing excel file, you can write python code and leverage excel '
    'toolkit to process the file step-by-step and get the information.'
)

# Optional quick-test defaults (set these for script-only runs)
DEFAULT_TASK_ID: Optional[str] = "t_001"
DEFAULT_QUERY: Optional[str] = (
     "Nzukọ ezinụlọ m ga-eme n&#39;izu a, e kenyere m poteto a gwepịara egwepịa ka m buru. Ndị bịara ya gụnyere nne na nna m lụrụ di, ejima m nwoke na ezinụlọ ya, nwanne nne m nwanyị na ezinụlọ ya, nne nne m na nwanne ya nwoke, ada nwanne ya nwoke, na ezinụlọ nwa ya nwanyị. Ndị toro eto niile ma e wezụga mụ onwe m alụọla di ma ọ bụ nwunye ọzọ, mana nna nna m ochie na nwunye nne m nwụrụ n&#39;afọ gara aga. Ndị di na nwunye niile dị ndụ na-abịa. Nwanne m nwoke nwere ụmụ abụọ ka bụ ụmụaka, nwanne nne m nwanyị nwere otu nwa dị afọ isii, nwa nwanne nne m nwanyị nwekwara ụmụ atọ na-erubeghị afọ 12. Echere m na onye ọ bụla toro eto ga-eri ihe dị ka poteto a gwakọtara agwakọta 1.5, nwatakịrị ọ bụla ga-eri ihe dị ka ọkara poteto a gwakọtara agwakọta, ma e wezụga ụmụnne nwanne m nke abụọ anaghị eri carbohydrates. Nduku nkezi dị ihe dị ka ọkara paụnd, a na-erekwa nduku na akpa paụnd ise. Akpa poteto ole ka m chọrọ? Nye naanị nọmba ahụ."
)


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def extract_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty model output_correct it.")

    # direct JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # fenced block
    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if not p:
                continue
            try:
                obj = json.loads(p)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

    # substring between first { and last }
    i = raw.find("{")
    j = raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        candidate = raw[i : j + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Model output is not valid JSON object.")


def call_parser_model(
    model: str,
    system_prompt: str,
    query: str,
    max_tokens: int = 1200,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    response = completion(
        model=model,
        allowed_openai_params=['reasoning_effort'],
        reasoning_effort="medium",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )

    text = response.choices[0].message.content or ""
    return extract_json(text)


def read_tasks(
    input_json: Optional[Path], task_id: Optional[str], query: Optional[str]
) -> List[Dict[str, str]]:
    if input_json is not None:
        data = json.loads(input_json.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("--input_json must be a JSON list of objects.")
        tasks: List[Dict[str, str]] = []
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                continue
            tid = str(row.get("task_id", f"task_{i}"))
            q = row.get("query") or row.get("question") or row.get("Question")
            if not q:
                continue
            file_name = str(row.get("file_name") or "").strip()
            query_text = str(q)
            if file_name:
                query_text += SUPPORT_FILE_QUERY_SUFFIX.format(file_name=file_name)
            tasks.append({"task_id": tid, "query": query_text})
        return tasks

    if not task_id or not query:
        raise ValueError("Provide either --input_json OR both --task_id and --query.")
    return [{"task_id": task_id, "query": query}]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run semantic parser prompt via LiteLLM and save JSON outputs."
    )
    parser.add_argument(
        "--prompt_path", default="tart_generator/tart_prompt.txt", help="Path to parser prompt file."
    )
    parser.add_argument("--env_file", default=".env", help="Path to .env file.")
    parser.add_argument(
        "--model", default="azure/gpt-5-mini", help="LiteLLM model name."
    )
    parser.add_argument(
        "--input_json", default=None, help="Optional JSON list input file."
    )
    parser.add_argument("--task_id", default=None, help="Single task id.")
    parser.add_argument("--query", default=None, help="Single query text.")
    parser.add_argument(
        "--output_json", default="tart_data/test/hi.json", help="Output file path."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    load_dotenv(dotenv_path=args.env_file)
    
    
    azure_key = os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    azure_base = os.getenv("AZURE_API_BASE") or os.getenv("AZURE_OPENAI_BASE_URL")
    azure_version = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
    if not azure_key or not azure_base:
        raise RuntimeError(
            "Azure OpenAI env vars not found. Need AZURE_API_KEY/AZURE_OPENAI_API_KEY "
            "and AZURE_API_BASE/AZURE_OPENAI_BASE_URL."
        )
    
    os.environ["AZURE_API_KEY"] = azure_key
    os.environ["AZURE_API_BASE"] = azure_base
    os.environ["AZURE_API_VERSION"] = azure_version

    prompt = load_prompt(Path(args.prompt_path))

    # Allow in-script defaults when CLI task/query are not provided.
    task_id = args.task_id if args.task_id is not None else DEFAULT_TASK_ID
    query = args.query if args.query is not None else DEFAULT_QUERY

    tasks = read_tasks(
        Path(args.input_json) if args.input_json else None,
        task_id,
        query,
    )
    if args.limit is not None:
        tasks = tasks[: args.limit]

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outputs: List[Dict[str, Any]] = []
    processed_task_ids = set()
    if out_path.exists():
        existing_outputs = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(existing_outputs, list):
            raise ValueError(f"Existing output file must contain a JSON list: {out_path}")
        outputs = existing_outputs
        processed_task_ids = {
            str(row["task_id"])
            for row in outputs
            if isinstance(row, dict) and row.get("task_id") is not None
        }

    for t in tasks:
        task_id = t["task_id"]
        q = t["query"]
        if task_id in processed_task_ids:
            print(f"[skip] {task_id}")
            continue

        try:
            semantic = call_parser_model(
                model=args.model,
                system_prompt=prompt,
                query=q,
            )
            outputs.append(
                {
                    "task_id": task_id,
                    "query": q,
                    "semantic": semantic,
                    "status": "ok",
                }
            )
            print(f"[ok] {task_id}")
        except Exception as e:
            outputs.append(
                {
                    "task_id": task_id,
                    "query": q,
                    "semantic": None,
                    "status": "error",
                    "error": str(e),
                }
            )
            print(f"[error] {task_id}: {e}")

        processed_task_ids.add(task_id)
        out_path.write_text(
            json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved {len(outputs)} records to {out_path}")

    print(f"Saved {len(outputs)} records to {out_path}")


if __name__ == "__main__":
    main()
