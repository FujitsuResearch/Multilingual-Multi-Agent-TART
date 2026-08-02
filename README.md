# TART

This repository contains code for running baseline and TART-assisted agent experiments on GAIA/MAPS and MultiTAT. TART representations must be generated before running TART-assisted agent files. The `tart_data/` files included in this repository are reference/sample files only and are not a complete replacement for running the TART generator.

## Project Structure

- `run_gaia_agent_baseline.py` - GAIA baseline agent runner.
- `run_gaia_agent_with_tart.py` - GAIA TART-assisted agent runner.
- `run_mutlitat_agent_baseline.py` - MultiTAT baseline agent runner.
- `run_mutlitat_agent_with_tart.py` - MultiTAT TART-assisted agent runner.
- `data_download/` - dataset download scripts.
- `tart_generator/` - TART representation generator.
- `tart_data/` - reference/sample TART outputs. Generate full TART data before full experiments.
- `gaia_data/` - GAIA/MAPS data. Low-resource data is provided; high/mid GAIA/MAPS data should be downloaded first.
- `mutitat_eval/` - MultiTAT data, scorer, benchmark, and config.
- `gaia_config_ymls/` - GAIA baseline and TART run configs.
- `agent_plans/` - saved agent plans used by the LLM judge.
- `llm_judge/` - LLM-as-a-judge prompt, examples, and runner.
- `camel/` and `utils/` - local runtime and workforce utilities.

## Setup

Run all commands from the repository root:

```bash
cd /path/to/TART
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file with the model credentials required by your setup.

## Step 1: Download Data

### GAIA/MAPS

Low-resource GAIA data is provided in:

```text
gaia_data/low_resource_lang_data/
```

Download high/mid-resource GAIA/MAPS data before running GAIA files:

```bash
python data_download/gaia_download.py --repo-id <HF_DATASET_REPO_ID>
```

Arguments:

- `--repo-id` - Hugging Face dataset repository id for GAIA/MAPS.
- `--output-dir` - optional output directory. Default:
  ```text
  gaia_data/high_mid_lang_data/
  ```

The downloader saves data into:

```text
gaia_data/high_mid_lang_data/
```

and organizes it as:

```text
gaia_data/high_mid_lang_data/original_data/<language>/gaia/<language>.json
```

The script downloads `gaia_v2` for most languages, keeps original `gaia` for Russian, renames folders to `gaia`, and renames JSON files to match the language name.

### MultiTAT

Download MultiTAT data before running MultiTAT files:

```bash
python data_download/multitat_download.py
```

Arguments:

- `--langs` - optional list of language codes. Default downloads all supported languages: `bn de en es fr ja ru sw te th zh`.
- `--base-url` - optional source URL for the MultiTAT dataset files.

The data is saved into:

```text
mutitat_eval/data/
```

Example for selected languages:

```bash
python data_download/multitat_download.py --langs en de sw
```

## Step 2: Generate Full TART Data

Before running TART-assisted files, generate TART representations for the language and dataset you want to run. The included `tart_data/` content is for reference/sample use only.

General command:

```bash
python tart_generator/run_tart_generator.py \
  --input_json <input_tasks.json> \
  --output_json <output_tart_file.json> \
  --model azure/gpt-5-mini \
  --env_file .env
```

Arguments:

- `--prompt_path` - TART prompt file. Default: `tart_generator/tart_prompt.txt`.
- `--env_file` - environment file. Default: `.env`.
- `--model` - LiteLLM model name. Default: `azure/gpt-5-mini`.
- `--input_json` - JSON list of tasks. Each item should contain `task_id` and a question field such as `query`, `question`, or `Question`.
- `--task_id` - single task id, used only when not using `--input_json`.
- `--query` - single query text, used only when not using `--input_json`.
- `--output_json` - output TART JSON path.
- `--limit` - optional number of tasks to process.

Example for GAIA Hindi:

```bash
python tart_generator/run_tart_generator.py \
  --input_json gaia_data/high_mid_lang_data/original_data/hindi/gaia/hindi.json \
  --output_json tart_data/gpt-5-mini/hi.json \
  --model azure/gpt-5-mini \
  --env_file .env
```

Example for MultiTAT English:

```bash
python tart_generator/run_tart_generator.py \
  --input_json mutitat_eval/data/dev.en.json \
  --output_json tart_data/multitat/mistral/en.json \
  --model azure/gpt-5-mini \
  --env_file .env
```

For a smoke test:

```bash
python tart_generator/run_tart_generator.py \
  --input_json gaia_data/high_mid_lang_data/original_data/hindi/gaia/hindi.json \
  --output_json tart_data/gpt-5-mini/hi.json \
  --limit 1
```

## Step 3: Run GAIA

GAIA scripts route languages internally:

- High/mid-resource: `english`, `arabic`, `hindi`, `german`, `russian`
  - data directory: `gaia_data/high_mid_lang_data`
- Low-resource: `swahili`, `kygryz`, `yoruba`, `igbo`, `bengali`, `nyanja`
  - data directory: `gaia_data/low_resource_lang_data`

### GAIA Baseline

Run:

```bash
python run_gaia_agent_baseline.py --normal hindi
```

Argument:

- `--normal` - config name from `gaia_config_ymls/config.yaml`.

Results are saved under:

```text
results/gaia/baseline/
```

### GAIA With TART

Generate the TART file first, then run:

```bash
python run_gaia_agent_with_tart.py --parser parser_hindi
```

Argument:

- `--parser` - config name from `gaia_config_ymls/config_parser.yaml`.

Important config fields:

- `lang` - language name used to locate GAIA data.
- `parser_model` - selects which `tart_data/` subdirectory is used.
- `input_ablations` - removes selected TART fields for ablation.
- `input_operation_ablations` - removes operation fields for ablation.
- `DOTENV_FILE` - environment file path.

Results are saved under:

```text
results/gaia/tart/
```

or ablation-specific subdirectories.

## Step 4: Run MultiTAT

MultiTAT language codes are:

```text
bn de en es fr ja ru sw te th zh
```

### MultiTAT Baseline

Using config:

```bash
python run_mutlitat_agent_baseline.py --setup multitat_en --dotenv-file .env
```

Direct CLI run:

```bash
python run_mutlitat_agent_baseline.py \
  --lang en \
  --max-tries 1 \
  --max-replanning-tries 1 \
  --subset 10 \
  --dotenv-file .env
```

Arguments:

- `--setup` - config name from `mutitat_eval/config_multitat.yaml`.
- `--config` - optional config path. Default: `mutitat_eval/config_multitat.yaml`.
- `--dotenv-file` - environment file path.
- `--lang` - MultiTAT language code.
- `--max-tries` - retry count per task.
- `--max-replanning-tries` - replanning retry count.
- `--subset` - optional number of examples.
- `--idx` - optional comma-separated task indices.
- `--answer-type` - optional filter by answer type.
- `--answer-from` - optional filter by answer source.

Results are saved under:

```text
results/MultiTAT/baseline/
```

### MultiTAT With TART

Generate the TART file first, then run:

```bash
python run_mutlitat_agent_with_tart.py --setup multitat_en --dotenv-file .env
```

Direct CLI run with an explicit TART file:

```bash
python run_mutlitat_agent_with_tart.py \
  --lang en \
  --semantics-path tart_data/multitat/mistral/en.json \
  --max-tries 1 \
  --max-replanning-tries 1 \
  --subset 10 \
  --dotenv-file .env
```

Additional argument:

- `--semantics-path` - path to generated TART JSON. If omitted, the default is:
  ```text
  tart_data/multitat/mistral/<lang>.json
  ```

Results are saved under:

```text
results/MultiTAT/tart/
```

## Step 5: Run LLM as a Judge

The LLM judge categorizes saved agent plans using the prompt and examples in `llm_judge/`. This is useful for analyzing plan behavior after agent runs.

Default run:

```bash
python llm_judge/run_llm_judge.py
```

Small test:

```bash
python llm_judge/run_llm_judge.py --limit 1
```

With explicit input and output:

```bash
python llm_judge/run_llm_judge.py \
  --input agent_plans/gaia_agent/gpt-5-mini/gpt5_mini_baseline_plans.json \
  --output llm_judge/llm_judge_output.json \
  --limit 10
```

Arguments:

- `--language` - run only one language.
- `--limit` - maximum samples to judge.
- `--task-id` - run only one task id.
- `--input` - input agent-plan JSON file.
- `--output` - output JSON file. Default:
  ```text
  llm_judge/llm_judge_output.json
  ```
- `--model` - judge model name.
- `--max-tokens` - maximum output tokens. Default: `8192`.
- `--temperature` - temperature argument.

The current judge script uses a local OpenAI-compatible endpoint:

```text
http://localhost:5643/v1
```

Make sure that server is running before executing the judge.

## Recommended Execution Order

1. Install dependencies and create `.env`.
2. Download GAIA/MAPS high/mid-resource data:
   ```bash
   python data_download/gaia_download.py --repo-id <HF_DATASET_REPO_ID>
   ```
3. Download MultiTAT data:
   ```bash
   python data_download/multitat_download.py
   ```
4. Generate full TART representations for the target dataset/language.
5. Run baseline agent files.
6. Run TART-assisted agent files.
7. Run LLM-as-a-judge on saved agent plans if plan analysis is needed.

## Notes

- Run commands from the repository root.
- `results/`, `tmp/`, logs, caches, and `.env` are ignored by Git.
- The provided `tart_data/` files are reference/sample data only. Full TART data should be regenerated with `tart_generator/run_tart_generator.py` before full experiments.
- Update YAML config files if you add new languages or new model setups.
