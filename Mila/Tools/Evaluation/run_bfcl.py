"""Run one arm of a paired BFCL evaluation: function calling, against one served engine.

BFCL (the Berkeley Function Calling Leaderboard, Gorilla's bfcl-eval package) renders each entry
with the model's own function-calling prompt and sends it to a server's Completions route. Every
arm here is a server -- reference_server.py for `hf`, MIS for `mila`, llama-server for `llamacpp`
-- and every arm is sent the same token ids: the request BFCL builds is replaced with one carrying
the ids BFCL's own HuggingFace tokenizer produces. Decoding is greedy, one entry at a time.
compare_arms.py pairs two arms by entry.
"""

import argparse
import json
import os
import pathlib
import sys
import time
import types

import servers

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct-FC"

# Single-turn Python: the AST-checked categories, live and not, about 3,500 entries. Multi-turn,
# memory and web search are BFCL's agentic categories; web search needs the network during a run.
DEFAULT_CATEGORIES = "python"


def query_with_token_ids(self, inference_data):
    """
    OSSHandler._query_prompting, sending token ids where it sends text. The rendering is BFCL's,
    unchanged; only the tokenizing moves to this side, so every server sees the same ids.
    """
    formatted_prompt = self._format_prompt(inference_data["message"], inference_data["function"])
    inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}
    prompt_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
    remaining = self.max_context_length - len(prompt_ids) - 2
    max_tokens = min(4096, remaining) if remaining > 0 else 1000
    extra_body = {}

    if hasattr(self, "stop_token_ids"):
        extra_body["stop_token_ids"] = self.stop_token_ids

    started = time.time()
    response = self.client.completions.create(
        model=self.model_path_or_id,
        temperature=self.temperature,
        prompt=[prompt_ids],
        max_tokens=max_tokens,
        extra_body=extra_body,
        timeout=72000,
    )

    return response, time.time() - started


def install_token_id_requests(model):
    """Replace the request of the handler `model` uses, and refuse a handler that builds its own."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler

    handler = MODEL_CONFIG_MAPPING[model].model_handler

    if not issubclass(handler, OSSHandler) or handler._query_prompting is not OSSHandler._query_prompting:
        sys.exit(f"{model}'s BFCL handler ({handler.__name__}) builds its own request, so it cannot be sent token ids.")

    OSSHandler._query_prompting = query_with_token_ids


def write_id_limit(project_root, categories, limit):
    """The first `limit` entries of each category, in BFCL's own run-ids file."""
    from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument, sort_key

    selected = {}

    for category in parse_test_category_argument(categories):
        entries = sorted(load_dataset_entry(category), key=sort_key)
        selected[category] = [entry["id"] for entry in entries[:limit]]

    (project_root / "test_case_ids_to_generate.json").write_text(json.dumps(selected, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arm", choices=sorted(servers.SERVER_NAMES))
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="the run's directory; this arm writes to <output>/<arm>/bfcl")
    parser.add_argument("--url", required=True, help="the arm's server: reference_server.py, MIS or llama-server")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="BFCL's registry name, which picks its prompt format")
    parser.add_argument("--categories", default=DEFAULT_CATEGORIES, help="BFCL categories or collections, comma-separated")
    parser.add_argument("--local-model-path", type=pathlib.Path,
                        help="the model's HuggingFace files on disk, for the tokenizer, instead of the Hub")
    parser.add_argument("--limit", type=int, default=0,
                        help="first N entries of each category; for a smoke run, never for a result")
    arguments = parser.parse_args()

    output = arguments.output / arguments.arm / "bfcl"

    if any(output.glob("result/**/*_result.json")):
        sys.exit(f"{output} already holds a run. Choose another --output; compare_arms.py reads whatever is there.")

    output.mkdir(parents=True, exist_ok=True)
    categories = arguments.categories.split(",")

    # bfcl-eval reads these when it is imported, so they are set before anything imports it.
    os.environ["BFCL_PROJECT_ROOT"] = str(output.resolve())
    os.environ["REMOTE_OPENAI_BASE_URL"] = f"{arguments.url}/v1"
    os.environ.setdefault("REMOTE_OPENAI_API_KEY", "EMPTY")

    from bfcl_eval._llm_response_generation import main as generate
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.eval_checker.eval_runner import main as evaluate

    if arguments.model not in MODEL_CONFIG_MAPPING:
        sys.exit(f"BFCL has no model named {arguments.model}; `bfcl models` lists them.")

    tokenizer_name = str(arguments.local_model_path or MODEL_CONFIG_MAPPING[arguments.model].model_name)
    served = servers.preflight(arguments.arm, arguments.url, tokenizer_name)
    print(f"{servers.SERVER_NAMES[arguments.arm]} serves {served['model']['id']}")

    install_token_id_requests(arguments.model)

    if arguments.limit:
        write_id_limit(output, categories, arguments.limit)

    settings = {
        "model": arguments.model,
        "categories": arguments.categories,
        "tokenizer": tokenizer_name,
        "limit": arguments.limit,
        "temperature": 0.0,
    }
    record = servers.environment(arguments.arm, "bfcl-eval", served, settings)
    servers.write_environment(output, record)

    generation = types.SimpleNamespace(
        model=[arguments.model],
        test_category=categories,
        temperature=0.0,
        include_input_log=True,
        exclude_state_log=False,
        # One entry at a time; BFCL's default is a hundred concurrent requests.
        num_threads=1,
        num_gpus=1,
        gpu_memory_utilization=0.9,
        backend="vllm",
        skip_server_setup=True,
        local_model_path=str(arguments.local_model_path) if arguments.local_model_path else None,
        result_dir=None,
        allow_overwrite=False,
        run_ids=bool(arguments.limit),
        enable_lora=False,
        max_lora_rank=None,
        lora_modules=None,
    )
    generate(generation)
    evaluate([arguments.model], categories, None, None, bool(arguments.limit))


if __name__ == "__main__":
    main()
