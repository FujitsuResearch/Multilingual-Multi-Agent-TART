import sys

sys.path.append("../")
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.tools import BaseTool
import json
import os
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, Tuple, Callable

from tqdm import tqdm
from camel.benchmarks import BaseBenchmark
from camel.models import BaseModelBackend
from camel.tasks import Task
from camel.societies.workforce import Workforce
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from loguru import logger
from .common import extract_pattern, extract_dict_from_str
from .enhanced_role_playing import OwlGaiaRolePlaying, run_society
from .enhanced_workforce_parser import OwlGaiaWorkforce
from litellm import completion


def _load_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def _extract_json_obj(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    if "```" in raw:
        for block in raw.split("```"):
            b = block.strip()
            if b.startswith("json"):
                b = b[4:].strip()
            try:
                obj = json.loads(b)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

    i, j = raw.find("{"), raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            obj = json.loads(raw[i : j + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _should_remove_ambiguities(question: str) -> bool:
    return "here are the necessary" in (question or "").lower()


def run_semantic_parser_online(
    query: str,
    model: str,
    system_prompt: str,
    max_tokens: int = 1200,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    resp = completion(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    text = resp.choices[0].message.content or ""
    return _extract_json_obj(text)


class GAIABenchmark(BaseBenchmark):
    r"""GAIA Benchmark adapted from `"GAIA: a benchmark for General AI
    Assistants"
    <https://huggingface.co/datasets/gaia-benchmark/GAIA>`_.

    Args:
        data_dir (str): The directory to save the data.
        save_to (str): The file to save the results.
        processes (int, optional): The number of processes to use.
            (default: :obj:`1`)
    """

    def __init__(
        self,
        data_dir: str,
        save_to: str,
        processes: int = 1,
        parser_enabled: bool = True,
        parser_model: str = "azure/gpt-5-mini",
        semantic_ablation_fields: Optional[List[str]] = None,
    ):
        r"""Initialize the GAIA benchmark.

        Args:
            data_dir (str): The directory to save the data.
            save_to (str): The file to save the results.
            processes (int, optional): The number of processes to use for
                parallel processing. (default: :obj:`1`)
        """
        super().__init__("gaia", data_dir, save_to, processes)
        self.parser_enabled = parser_enabled
        self.parser_model = parser_model
        self.semantic_ablation_fields = set(semantic_ablation_fields or [])
        self._parser_prompt_cache = None
        self._semantic_cache = {}

    def _ablate_semantic_repr(self, semantic: Dict[str, Any]) -> Dict[str, Any]:
        if not self.semantic_ablation_fields or not isinstance(semantic, dict):
            return semantic

        return {
            key: value
            for key, value in semantic.items()
            if key not in self.semantic_ablation_fields
        }

    def download(self):
        r"""Download the GAIA dataset."""
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="/GAIA-MAPS",
            repo_type="dataset",
            local_dir=self.data_dir,
            local_dir_use_symlinks=True,
        )

    def _get_semantic_repr(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if not self.parser_enabled:
            return {}
        try:
            lang_value = str(task.get("lang") or task.get("Language")).lower()
            lang_map = {
                "english": "en",
                "arabic": "ar",
                "hebrew": "he",
                "hindi": "hi",
                "japanese": "ja",
                "korean": "ko",
                "russian": "ru",
                "german": "de",
                "spanish": "es",
                "italian": "it",
                "portuguese": "pt-br",
                "swahili": "sw",
                "bengali": "bn",
                "kygryz": "ky",
                "nyanja": "ny",
                "igbo": "ig",
                "yoruba": "yo",
            }

            lang_code = lang_map.get(lang_value, lang_value)
            
            model=self.parser_model
            if model=="azure/gpt-5-mini":
                dir_path = "tart_data/gpt-5-mini"
                print("gpt5-mini parser is selected")
            elif model== "bedrock/qwen.qwen3-vl-235b-a22b":
                dir_path = "tart_data/qwen_aws"
                print("qwen parser is selected")
            elif model== "azure/Mistral-Large-3":
                dir_path = "tart_data/mistral"
                print("mistral parser is selected")
            else:
                dir_path = "tart_data/claude_parser"
                print("claude parser is selected")
                
            # dir_path = "semantics_data/claude_parser"
            if lang_code not in self._semantic_cache:
                with open(f"{dir_path}/{lang_code}.json", "r", encoding="utf-8") as f:
                    semantics = json.load(f)
                    
                self._semantic_cache[lang_code] = {
                    row["task_id"]: self._ablate_semantic_repr(
                        row.get("semantic", {})
                    )
                    for row in semantics
                    if isinstance(row, dict) and row.get("task_id")
                    }
            
                    
            return self._semantic_cache[lang_code].get(task["task_id"], {})
        except Exception as e:
            logger.warning(f"Semantic parser failed: {e}")
            return {}
        
    def _add_final_answer_constraint(self, semantic: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(semantic, dict):
            return semantic

        if "final_answer_constraint" in semantic:
            return semantic
        
        if "answer_type" not in semantic:
            return semantic

        answer_type = str(semantic.get("answer_type") or "unknown").lower()

        answer_type_text = "answer" if answer_type == "unknown" else answer_type
        output_constraint = (
            f"return only the final {answer_type_text} "
            "in the format requested by the question"
        )

        semantic = dict(semantic)
        semantic["final_answer_constraint"] = {
            "answer_type": answer_type,
            "output_constraint": output_constraint,
            "include_provenance": False,
        }
        
        semantic.pop("answer_type", None)

        return semantic

    def _check_task_completed(self, task_id: str) -> bool:
        for data in self._results:
            if data["task_id"] == task_id:
                return True
        return False

    def dump_tasks(self, save_path: str, datas):
        constructed_data = []
        for idx, data in enumerate(datas):
            tmp_dict = {
                "idx": idx,
                "task_id": data["task_id"],
                "Question": data["Question"],
                "Level": data["Level"],
                "Final answer": data["Final answer"],
                "Annotation Metadata": data["Annotator Metadata"],
            }

            constructed_data.append(tmp_dict)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(constructed_data, f, indent=4)
        f.close()

        print(f"Successfully dumped tasks to {save_path}")

    def load(
        self,
        force_download=False,
        lang: str = "english",
        hyb: bool = False,
        llm: bool = False,
        google: bool = False,
    ):
        r"""Load the GAIA dataset.

        Args:
            force_download (bool, optional): Whether to
                force download the data.
        """
        if force_download:
            logger.info("Force downloading data.")
            self.download()

        
        load_path = f"original_data/{lang}/gaia"

        # Define validation and test directories

        valid_dir = self.data_dir / load_path
        test_dir = self.data_dir / load_path
        gaia_files = self.data_dir / "support_files"

        # Check if directories exist; if not, download the data
        if not valid_dir.is_dir() or not test_dir.is_dir():
            logger.info("Data not found. Downloading data.")
            exit(1)
            self.download()

        # Load metadata for both validation and test datasets
        for path, label in zip([valid_dir, test_dir], ["valid", "test"]):
            self._data[label] = []
            filename = (
                f"{lang[:2]}_en.json" if (hyb or llm or google) else f"{lang}.json"
            )
            with open(path / filename, "r") as f:
                data_list = json.load(f)
                for data in data_list:
                    if data["task_id"] == "0-0-0-0-0":
                        continue
                    if data["file_name"]:
                        data["file_name"] = gaia_files / data["file_name"]
                    self._data[label].append(data)
        return self

    def _load_results_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                _results = json.load(f)
            f.close()
            return _results
        except Exception as e:
            logger.warning(f"The file {file_path} does not exist.")
            return []

    def _save_results_to_file(self, results: List[Dict[str, Any]], file_path: str):
        # get base dir of file_path
        base_dir = os.path.dirname(file_path)
        os.makedirs(base_dir, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        f.close()

    @property
    def train(self):
        r"""Get the training set."""
        raise NotImplementedError("GAIA does not have a training set.")

    def _load_tasks(
        self,
        on: Literal["valid", "test"],
        level: Union[int, List[int], Literal["all"]],
        randomize: bool = False,
        subset: Optional[int] = None,
        idx: Optional[List[int]] = None,
        lang: str = "english",
        hyb: bool = False,
        llm: bool = False,
        google: bool = False,
    ) -> List[Dict[str, Any]]:
        r"""Load tasks from the dataset."""
        self.load(
            lang=lang, hyb=hyb, llm=llm, google=google
        )  # function modified to accept arguments
        if on not in ["valid", "test"]:
            raise ValueError(
                f"Invalid value for `on`: {on}, expected 'valid' or 'test'."
            )
        levels = (
            [1, 2, 3]
            if level == "all"
            else [level] if isinstance(level, int) else level
        )

        datas = [
            data
            for data in self._data[on]
            if str(data.get("Level")) in [str(level) for level in levels]
        ]
        print(len(datas))

        if randomize:
            random.shuffle(datas)
        if subset:
            datas = datas[:subset]

        if idx is not None:
            # pick only the tasks with the specified idx
            if len(idx) != 0:
                datas = [datas[i] for i in idx]

        return datas

    def get_formal_answer(self, question: str, text: str) -> str:

        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        )

        agent = ChatAgent(
            "You are a helpful assistant that can answer questions and provide final answers.",
            model=model,
        )

        prompt = f"""
I am solving a question:
<question>
{question}
</question>

Now, I have solved the question, the primary answer is as follows:
<answer>
{text}
</answer>

Now, I need you to determine the final answer. Do not try to solve the question, just pay attention to ONLY the format in which the answer is presented. DO NOT CHANGE THE MEANING OF THE PRIMARY ANSWER.
You should first analyze the answer format required by the question and then output the final answer that meets the format requirements. 
Here are the requirements for the final answer:
<requirements>
The final answer must be output exactly in the format specified by the question. Your final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. Numbers do not need to be written as words, but as digits.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
</requirements>

Please output with the final answer according to the requirements without any other text. If the primary answer is already a final answer with the correct format, just output the primary answer.
        """
        resp = agent.step(prompt)
        return resp.msgs[0].content

    def run_role_playing(
        self,
        user_role_name: str,
        assistant_role_name: str,
        user_agent_kwargs: dict,
        assistant_agent_kwargs: dict,
        on: Literal["train", "valid", "test"],
        level: Union[int, List[int], Literal["all"]],
        randomize: bool = False,
        subset: Optional[int] = None,
        idx: Optional[List[int]] = None,
        save_result: bool = False,
    ) -> Dict[str, Any]:

        # Validate inputs
        datas = self._load_tasks(on, level, randomize, subset, idx)
        logger.info(f"Number of tasks: {len(datas)}")

        self._results = []

        if save_result:
            self._results = self._load_results_from_file(self.save_to)

        # Process tasks
        for task in tqdm(datas, desc="Running"):
            if self._check_task_completed(task["task_id"]):
                logger.success(
                    f"The following task is already completed:\n task id: {task['task_id']}, question: {task['Question']}"
                )
                continue

            if_prepared_task, info = self._prepare_task(task)
            if not if_prepared_task:
                _result_info = {
                    "task_id": task["task_id"],
                    "question": task["Question"],
                    "level": task["Level"],
                    "model_answer": None,
                    "ground_truth": None,
                    "score": 0,
                    "history": None,
                }
                self._results.append(_result_info)
                continue
            try:
                logger.info(f"Task Question: {task['Question']}")
                logger.info(f"Required tools: {task['Annotator Metadata']['Tools']}")

                task_kwargs = {
                    "task_prompt": task["Question"],
                    "with_task_specify": False,
                }

                society = OwlGaiaRolePlaying(
                    **task_kwargs,
                    user_role_name=user_role_name,
                    user_agent_kwargs=user_agent_kwargs,
                    assistant_role_name=assistant_role_name,
                    assistant_agent_kwargs=assistant_agent_kwargs,
                )

                raw_answer, chat_history, token_info = run_society(society)
                try:
                    answer = extract_pattern(raw_answer, "final_answer")
                except Exception as e:
                    logger.error(
                        f"Error in extracting final answer from text {raw_answer}: {e}"
                    )
                    answer = None

                logger.info(
                    f"Model answer: {answer}, Ground truth: {task['Final answer']}"
                )

                _result_info = {
                    "task_id": task["task_id"],
                    "question": task["Question"]
                    + "Please decompose the task into several sub-tasks and find the answer step-by-step.",
                    "level": task["Level"],
                    "model_answer": answer,
                    "ground_truth": task["Final answer"],
                    "score": self.question_scorer(answer, task["Final answer"]),
                    "token_info": token_info,
                    "history": chat_history,
                }
                self._results.append(_result_info)

            except Exception as e:
                logger.error(f"Error in processing task: {e}")

            if save_result:
                self._save_results_to_file(self._results, self.save_to)

        return self._generate_summary()

    def run(
        self,
        agent: ChatAgent,
        on: Literal["valid", "test"],
        level: Union[int, List[int], Literal["all"]],
        max_tries: int = 3,
        randomize: bool = False,
        subset: Optional[int] = None,
        idx: Optional[List[int]] = None,
        save_result: bool = False,
    ) -> Dict[str, Any]:
        r"""Run the benchmark with a single agent."""

        datas = self._load_tasks(on, level, randomize, subset, idx)

        self._results = []

        if save_result:
            self._results = self._load_results_from_file(self.save_to)

        success = False
        tries = 0
        trajectory_with_retry: List[dict] = []

        for task in tqdm(datas, desc="Running"):
            if self._check_task_completed(task["task_id"]):
                logger.success(
                    f"The following task is already completed:\n task id: {task['task_id']}, question: {task['Question']}"
                )
                continue

            if_prepared_task, info = self._prepare_task(task)
            if not if_prepared_task:
                _result_info = {
                    "task_id": task["task_id"],
                    "question": task["Question"],
                    "level": task["Level"],
                    "model_answer": None,
                    "ground_truth": None,
                    "score": 0,
                    "history": None,
                }
                self._results.append(_result_info)
                continue

            success = False
            tries = 0
            trajectory_with_retry: List[dict] = []

            while not success and tries < max_tries:
                tries += 1
                logger.info(f"Attempt {tries}/{max_tries} for task {task['task_id']}")

                try:
                    logger.info(f"Task Question: {task['Question']}")
                    logger.info(
                        f"Required tools: {task['Annotator Metadata']['Tools']}"
                    )
                    agent.reset()

                    prompt = task["Question"]

                    resp = agent.step(prompt)
                    raw_answer = resp.msgs[0].content
                    answer = self.get_formal_answer(task["Question"], raw_answer)

                    logger.info(
                        f"Model answer: {answer}, Ground truth: {task['Final answer']}"
                    )

                    score = self.question_scorer(answer, task["Final answer"])
                    success = (
                        score == True
                    )  # Consider task successful if score is perfect
                    trajectory_dict = {
                        "attempts": tries,
                        "model_answer": answer,
                        "ground_truth": task["Final answer"],
                        "success": success,
                        "trajectory": agent.chat_history,
                    }
                    trajectory_with_retry.append(trajectory_dict)

                    if success or tries == max_tries:
                        _result_info = {
                            "task_id": task["task_id"],
                            "question": task["Question"],
                            "level": task["Level"],
                            "model_answer": answer,
                            "ground_truth": task["Final answer"],
                            "score": score,
                            "attempts": tries,
                            "trajectory": trajectory_with_retry,
                        }
                        self._results.append(_result_info)

                except Exception as e:
                    logger.error(f"Error in processing task: {e}")

            if save_result:
                self._save_results_to_file(self._results, self.save_to)

        return self._generate_summary()

    def run_workforce_with_retry(
        self,
        workforce: Workforce,
        on: Literal["valid", "test"],
        level: Union[int, List[int], Literal["all"]],
        max_tries: int = 3,
        max_replanning_tries: int = 2,
        randomize: bool = False,
        subset: Optional[int] = None,
        idx: Optional[List[int]] = None,
        save_result: bool = False,
        lang: str = "english",
        hyb: bool = False,
        llm: bool = False,
        google: bool = False,
        agent_option: bool = False,  # agent decide if translator tool call or not
        icl_enabled: bool = False,
    ) -> Dict[str, Any]:
        r"""Run the benchmark with retry mechanism.

        Args:
            workforce (Workforce): The workforce to use for task processing.
            max_tries (int): Maximum number of retries per task. Defaults to 3.
            on (Literal["valid", "test"]): Which dataset split to run on.
            level (Union[int, List[int], Literal["all"]]): Which difficulty levels to run.
            max_tries (int): Maximum number of retries per task. Defaults to 3.
            max_replanning_tries (int): Maximum number of replanning tries. Defaults to 2.
            randomize (bool): Whether to randomize task order. Defaults to False.
            subset (Optional[int]): Number of tasks to run. Defaults to None (all tasks).
            idx (Optional[List[int]]): Specific task indices to run. Defaults to None.
            save_result (bool): Whether to save results to file. Defaults to False.
        Returns:
            Dict[str, Any]: Summary of benchmark results.
        """
        tasks = self._load_tasks(
            on,
            level,
            randomize,
            subset,
            idx,
            lang=lang,
            hyb=hyb,
            llm=llm,
            google=google,
        )

        # for getting ground_truths
        metadata_path = Path(__file__).resolve().parents[1] / "gaia_data" / "metadata.jsonl"
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        self._results = []

        if save_result:
            self._results = self._load_results_from_file(self.save_to)


        for task in tqdm(tasks, desc=f"Running {on} set"):
            if self._check_task_completed(task["task_id"]):
                logger.success(
                    f"The following task is already completed:\n task id: {task['task_id']}, question: {task['Question']}"
                )
                continue

            success = False
            tries = 0
            trajectory_with_retry: List[dict] = []
            ground_truths = []

            # getting ground_truths
            task_id = task["task_id"]
            GT = False  # Ground truth
            for item in data:
                if item.get("task_id") == task_id:
                    ground_truths.append(task["Final answer"])
                    ground_truths.append(item["Final answer"])
                    GT = True
                    break

            if not GT:
                ground_truths = ["ID not matched"]

            unique_ground_truths = list(dict.fromkeys(ground_truths))
            # Ground_truth obtained
            logger.info(f"ground_truth: {unique_ground_truths}")

            while not success and tries < max_tries:
                tries += 1
                logger.info(f"Attempt {tries}/{max_tries} for task {task['task_id']}")

                try:
                    task["raw_question"] = task["Question"]
                    valid, error_msg = self._prepare_task(task)
                    if not valid:
                        logger.error(error_msg)
                        break
                    logger.info(f"Task Question: {task['Question']}")
                    task["lang"] = lang
                    task["semantic_repr"] = self._get_semantic_repr(task)
                    task["semantic_repr"] = self._add_final_answer_constraint(task["semantic_repr"])
                    camel_task = self._create_task(task, icl=icl_enabled)
                    if workforce.is_running():
                        workforce.stop()
                    processed_task = workforce.process_task(
                        camel_task, max_replanning_tries=max_replanning_tries
                    )

                    try:
                        answer = workforce.get_workforce_final_answer(processed_task)
                    except Exception as e:
                        logger.error(f"Error extracting final answer: {e}")
                        answer = None

                    logger.info(
                        f"Model answer: {answer}, Ground truth: {unique_ground_truths}"
                    )

                    scores = [
                        self.question_scorer(answer, gt) for gt in unique_ground_truths
                    ]
                    success = any(scores)
                    matched_gt = (
                        unique_ground_truths[scores.index(True)]
                        if success
                        else unique_ground_truths[0]
                    )
                    
                    logger.info(f"Score: {scores}")
                    
                    trajectory_dict = {
                        "attempts": tries,
                        "model_answer": answer,
                        "ground_truth": matched_gt,
                        "success": success,
                        "trajectory": workforce.get_overall_task_solve_trajectory(),
                    }
                    trajectory_with_retry.append(trajectory_dict)

                    if success or tries == max_tries:
                        _result_info = {
                            "task_id": task["task_id"],
                            "question": task["Question"],
                            "level": task["Level"],
                            "model_answer": answer,
                            "ground_truth": matched_gt,
                            "score": success,
                            "attempts": tries,
                            "trajectory": trajectory_with_retry,
                        }
                        self._results.append(_result_info)

                except Exception as e:
                    matched_gt = (
                        unique_ground_truths[0]
                        if unique_ground_truths
                        else "ID not matched"
                    )
                    logger.error(f"Error in processing task (attempt {tries}): {e}")
                    if tries == max_tries:
                        _result_info = {
                            "task_id": task["task_id"],
                            "question": task["Question"],
                            "level": task["Level"],
                            "model_answer": None,
                            "ground_truth": matched_gt,
                            "score": False,
                            "attempts": tries,
                            "trajectory": trajectory_with_retry,
                        }
                        self._results.append(_result_info)

            if save_result:
                self._save_results_to_file(self._results, self.save_to)

        return self._generate_summary()

    def _prepare_task(self, task: Dict[str, Any]) -> Tuple[bool, str]:
        r"""Prepare the task by validating and enriching its data."""
        if task["file_name"]:

            if isinstance(task["file_name"], Path):
                task["file_name"] = str(task["file_name"])

            file_path = Path(task["file_name"])
            if not file_path.exists():
                logger.info(f"Skipping task because file not found: {file_path}")
                return False, f"Skipping task because file not found: {file_path}"
            if file_path.suffix in [".pdf", ".docx", ".doc", ".txt"]:
                task[
                    "Question"
                ] += f" Here are the necessary document files: {file_path}"

            elif file_path.suffix in [".jpg", ".jpeg", ".png"]:
                task["Question"] += f" Here are the necessary image files: {file_path}"

            elif file_path.suffix in [".xlsx", "xls", ".csv"]:
                task[
                    "Question"
                ] += f" Here are the necessary table files: {file_path}, for processing excel file, you can write python code and leverage excel toolkit to process the file step-by-step and get the information."

            elif file_path.suffix in [".py"]:
                task["Question"] += f" Here are the necessary python files: {file_path}"

            else:
                task["Question"] += f" Here are the necessary files: {file_path}"

        return True, None

    def _create_task(self, task: Dict[str, Any], icl: bool = False) -> Task:
        r"""Create a Task from benchmark task dict with ICL metadata.

        Args:
            task (Dict[str, Any]): The task to create the message from.

        Returns:
            Task: The task created from the input.
        """

        raw_question = task.get("raw_question")
        lang_value = str(task.get("lang") or task.get("Language")).lower()
        lang_map = {
            "english": "en",
            "arabic": "ar",
            "hebrew": "he",
            "hindi": "hi",
            "japanese": "ja",
            "korean": "ko",
            "russian": "ru",
            "german": "de",
            "spanish": "es",
            "italian": "it",
            "portuguese": "pt-br",
            "swahili": "sw",
            "bengali": "bn",
            "kygryz": "ky",
            "nyanja": "ny",
            "igbo": "ig",
            "yoruba": "yo",
        }

        lang_code = lang_map.get(lang_value, lang_value)

        if icl:
            file_name = task.get("file_name")
            has_attachments = bool(file_name)

            attachment_exts = []
            attachment_kinds = []

            if file_name:
                p = Path(str(file_name))
                ext = p.suffix.lower().lstrip(".")
                if ext:
                    attachment_exts.append(ext)
                if ext in {"pdf", "doc", "docx", "txt"}:
                    attachment_kinds.append("document")
                elif ext in {"jpg", "jpeg", "png", "gif", "webp", "bmp"}:
                    attachment_kinds.append("image")
                elif ext in {"xlsx", "xls", "csv"}:
                    attachment_kinds.append("table")
                elif ext in {"mp3", "wav", "m4a", "flac", "ogg"}:
                    attachment_kinds.append("audio")
                elif ext in {"mp4", "mov", "avi", "mkv", "webm"}:
                    attachment_kinds.append("video")
                elif ext in {"zip", "tar", "gz", "7z"}:
                    attachment_kinds.append("archive")
                elif ext == "py":
                    attachment_kinds.append("code")
                else:
                    attachment_kinds.append("file")

            icl_meta = {
                "task_id": str(task.get("task_id", "")),
                "raw_question": raw_question,
                "lang": lang_code,
                "has_attachments": has_attachments,
                "attachment_exts": attachment_exts,
                "attachment_kinds": attachment_kinds,
                "file_name": str(task.get("file_name") or ""),
            }

            additional_info = json.dumps(icl_meta, ensure_ascii=False)

            return Task(
                id=str(task["task_id"]),
                content=task["Question"],  # full task text (possibly with file hints)
                semantic_repr=task.get("semantic_repr", {}) or {},
                additional_info=additional_info,  # ICL metadata channel
            )

        meta = {
            "task_id": str(task.get("task_id", "")),
            "raw_question": raw_question,
            "semantic_repr": task.get("semantic_repr", {}),
            # "lang": lang_code,
            "file_name": str(task.get("file_name") or ""),
        }
        additional_info = json.dumps(meta, ensure_ascii=False)

        return Task(
            id=str(task["task_id"]),
            content=task["Question"],
            semantic_repr=task.get("semantic_repr", {}) or {},
            additional_info=additional_info
        )

    def _generate_summary(self) -> Dict[str, Any]:
        r"""Generate and return a summary of the benchmark results."""
        correct = sum(result["score"] for result in self._results)
        return {
            "total": len(self._results),
            "correct": correct,
            "results": self._results,
            "accuracy": correct / len(self._results) if len(self._results) > 0 else 0,
        }

    def question_scorer(self, model_answer: str, ground_truth: str) -> bool:
        r"""Scorer for the GAIA benchmark.
        https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/
        scorer.py

        Args:
            model_answer (str): The model answer.
            ground_truth (str): The ground truth answer.

        Returns:
            bool: The score of the model
        """

        def is_float(element: Any) -> bool:
            try:
                float(element)
                return True
            except ValueError:
                return False

        if is_float(ground_truth):
            logger.info(f"Evaluating {model_answer} as a number.")
            normalized_answer = self.normalize_number_str(model_answer)
            return normalized_answer == float(ground_truth)

        elif any(char in ground_truth for char in [",", ";"]):
            logger.info(f"Evaluating {model_answer} as a comma separated list.")
            gt_elems = self.split_string(ground_truth)
            ma_elems = self.split_string(model_answer)

            if len(gt_elems) != len(ma_elems):
                logger.warning(
                    "Answer lists have different lengths, returning False.",
                    UserWarning,
                )
                return False

            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if is_float(gt_elem):
                    normalized_ma_elem = self.normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    ma_elem = self.normalize_str(ma_elem, remove_punct=False)
                    gt_elem = self.normalize_str(gt_elem, remove_punct=False)
                    comparisons.append(ma_elem == gt_elem)
            return all(comparisons)
        else:
            logger.info(f"Evaluating {model_answer} as a string.")
            ma_elem = self.normalize_str(model_answer)
            gt_elem = self.normalize_str(ground_truth)
            return ma_elem == gt_elem

    def normalize_number_str(self, number_str: str) -> float:
        for char in ["$", "%", ","]:
            number_str = number_str.replace(char, "")
        try:
            return float(number_str)
        except ValueError:
            logger.error(f"String {number_str} cannot be normalized to number str.")
            return float("inf")

    def split_string(self, s: str, char_list: Optional[List[str]] = None) -> list[str]:
        r"""Split a string based on a list of characters.

        Args:
            s (str): The string to split.
            char_list (Optional[List[str]], optional): T
                he list of characters to split on.
                (default: :obj:`None`)
        """
        if char_list is None:
            char_list = [",", ";"]
        pattern = f"[{''.join(char_list)}]"
        return re.split(pattern, s)

    def normalize_str(self, input_str, remove_punct=True) -> str:
        r"""Normalize a string.

        Args:
            input_str: The input string to normalize.
            remove_punct: Whether to remove punctuation.

        Returns:
            str: The normalized string.
        """
        no_spaces = re.sub(r"\s", "", input_str)
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        else:
            return no_spaces.lower()
