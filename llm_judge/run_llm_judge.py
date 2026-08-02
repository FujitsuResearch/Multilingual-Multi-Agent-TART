"""
Run LLM as a Judge to categorize agent plans into allowed categories as specified below.

"""


import argparse
import json
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")


SCRIPT_DIR = Path(__file__).resolve().parents[1] 
INPUT_PATH = SCRIPT_DIR / "agent_plans" / "gaia_agent" / "gpt-5-mini" / "gpt5_mini_baseline_plans.json"
PROMPT_PATH = SCRIPT_DIR / "llm_judge" / "llm_judge_prompt.txt"
EXAMPLES_PATH = SCRIPT_DIR / "llm_judge" / "examples.txt"
OUTPUT_PATH = SCRIPT_DIR / "llm_judge" / "llm_judge_output.json"


def load_text(path):
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, records):
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_existing_results(path):
    if not path.exists():
        return []
    return load_json(path)


def sample_key(sample):
    return (
        str(sample.get("language", "")).lower(),
        str(sample.get("task_id", "")),
    )


def render_prompt(prompt_template, examples, sample):
    values = {
        "examples": examples,
        "target_language_query": sample.get("target_language_query", ""),
        "english_reference_query": sample.get("english_question", ""),
        "agent_plan": sample.get("agent_plan", ""),
        "ground_truth_answer": sample.get("ground_truth_answer", ""),
    }

    rendered = prompt_template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered


def filter_samples(samples, language=None, task_id=None, limit=None, existing_keys=None):
    filtered = samples

    if language:
        filtered = [
            sample
            for sample in filtered
            if str(sample.get("language", "")).lower() == language.lower()
        ]

    if task_id:
        filtered = [
            sample
            for sample in filtered
            if str(sample.get("task_id", "")) == task_id
        ]

    if existing_keys:
        filtered = [
            sample
            for sample in filtered
            if sample_key(sample) not in existing_keys
        ]

    if limit is not None:
        filtered = filtered[:limit]

    return filtered


def call_claude(prompt, args):
    BASE_URL = "http://localhost:5643/v1"
    API_KEY = "anything"
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Return the JSON evaluation for this case."},
        ],
        max_tokens=args.max_tokens,
        )
    text = response.choices[0].message.content or ""
    return text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Azure Claude Sonnet judge on Mistral false samples."
    )
    parser.add_argument("--language", help="Run only this language code, for example ar or ky.")
    parser.add_argument("--limit", type=int, help="Maximum number of samples to run.")
    parser.add_argument("--task-id", help="Run only this task id.")
    parser.add_argument("--input", default=str(INPUT_PATH), help="Input JSON file.")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSON file.")
    parser.add_argument(
        "--model",
        default=os.getenv("AZURE_CLAUDE_MODEL", "bedrock/us.anthropic.claude-opus-4-8"),
        help="LiteLLM model name for the Azure Claude deployment.",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()

    prompt_template = load_text(PROMPT_PATH)
    examples = load_text(EXAMPLES_PATH)
    output_path = Path(args.output)
    results = load_existing_results(output_path)
    existing_keys = {sample_key(result) for result in results}
    samples = filter_samples(
        load_json(Path(args.input)),
        language=args.language,
        task_id=args.task_id,
        limit=args.limit,
        existing_keys=existing_keys,
    )

    for index, sample in enumerate(samples, start=1):
        prompt = render_prompt(prompt_template, examples, sample)
        model_response = call_claude(prompt, args)
        result = dict(sample)
        result["model_response"] = model_response
        results.append(result)
        save_json(output_path, results)
        print(f"Saved {index}/{len(samples)}: {sample.get('task_id')} ({sample.get('language')})")

    print(f"Finished. Added {len(samples)} new samples. Total saved: {len(results)} to {output_path}")


if __name__ == "__main__":
    main()
