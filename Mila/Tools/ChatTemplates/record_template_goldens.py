"""
Render the conversations the protocol parity tests use through each checkpoint's own chat template.

The goldens in Mila/Tests/Dnn/Models/{Qwen,Gemma,LlaMa}/*.ChatProtocol.cpp are this script's output. The template is
the oracle (ModelHandle.md 3.4): it ships with the weights, and transformers' render_jinja_template is the
renderer apply_chat_template calls, so a golden is exactly the prompt the checkpoint was published with.

Usage:
    python record_template_goldens.py qwen|gemma|llama31|llama32 [--case NAME]

Needs `transformers` and `jinja2` only (no PyTorch), and the checkpoint's template in the HuggingFace cache:
Qwen/Qwen3.8-27B, google/gemma-4-12b-it, meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.2-3B-Instruct.
A case the template refuses prints as a comment saying so; its test is a refusal, not a golden.

Each case prints as a C++ raw string literal, ready to paste. Change a case here and its test in the same change.
"""
import argparse
import json
import pathlib
import sys

from transformers.utils.chat_template_utils import render_jinja_template

HUB = pathlib.Path.home() / ".cache" / "huggingface" / "hub"

TEMPLATES = {
    "qwen": ("models--Qwen--Qwen3.8-27B", {}),
    "gemma": ("models--google--gemma-4-12b-it", {"bos_token": "<bos>"}),
    "llama31": ("models--meta-llama--Llama-3.1-8B-Instruct", {"bos_token": "<|begin_of_text|>"}),
    # 3.2 defaults its date to today; the goldens fix it so they do not change at midnight.
    "llama32": ("models--meta-llama--Llama-3.2-3B-Instruct",
                {"bos_token": "<|begin_of_text|>", "date_string": "26 Jul 2024"}),
}

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string", "description": "The city"},
                                  "units": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}

TIME = {"type": "function", "function": {
    "name": "get_time", "description": "Get the local time in a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}

SYSTEM = {"role": "system", "content": "You are a concise assistant."}
ASK = {"role": "user", "content": "What is the weather in Paris?"}
ASK_BOTH = {"role": "user", "content": "Weather and time in Paris?"}


def call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def result(call_id, name, content):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


WEATHER_CALL = call("c1", "get_weather", {"city": "Paris"})
TIME_CALL = call("c2", "get_time", {"city": "Paris"})
WEATHER_RESULT = result("c1", "get_weather", "Sunny, 21 C")
TIME_RESULT = result("c2", "get_time", "14:05")

# name -> (messages, tools, template variables). Shared by both families, so a difference between the two
# suites is a difference between the templates.
CASES = {
    "plain": ([{"role": "user", "content": "What is the capital of Australia?"}], None, {}),
    "system": ([SYSTEM, {"role": "user", "content": "What is the capital of Australia?"}], None, {}),
    "trimmed": ([{"role": "system", "content": "  Be brief.\n"}, {"role": "user", "content": "  hi  \n"}], None, {}),
    "multi_turn": ([{"role": "user", "content": "Capital of Australia?"},
                    {"role": "assistant", "content": "Canberra."},
                    {"role": "user", "content": "And of Canada?"}], None, {}),
    "tools": ([SYSTEM, ASK], [WEATHER, TIME], {}),
    "tools_no_system": ([ASK], [WEATHER], {}),
    "one_call_resume": ([SYSTEM, ASK, {"role": "assistant", "content": "", "tool_calls": [WEATHER_CALL]},
                         WEATHER_RESULT], [WEATHER], {}),
    "two_calls_resume": ([ASK_BOTH,
                          {"role": "assistant", "content": "Let me check both.", "tool_calls": [WEATHER_CALL, TIME_CALL]},
                          WEATHER_RESULT, TIME_RESULT], [WEATHER, TIME], {}),
    "two_calls_out_of_order": ([ASK_BOTH, {"role": "assistant", "content": "", "tool_calls": [WEATHER_CALL, TIME_CALL]},
                                TIME_RESULT, WEATHER_RESULT], [WEATHER, TIME], {}),
    "closed_cycle": ([ASK,
                      {"role": "assistant", "content": "", "reasoning_content": "I need the weather tool.",
                       "tool_calls": [WEATHER_CALL]},
                      WEATHER_RESULT,
                      {"role": "assistant", "content": "It is sunny and 21 C.", "reasoning_content": "The tool said sunny."},
                      {"role": "user", "content": "And tomorrow?"}], [WEATHER], {}),
    "typed_arguments": ([{"role": "user", "content": "Configure it."},
                         {"role": "assistant", "content": "",
                          "tool_calls": [call("c1", "configure",
                                              {"tokens": 4096, "enabled": True, "mode": "high", "opts": {"z": 1, "a": [1, 2]}})]},
                         result("c1", "configure", "ok")], None, {}),
    "json_result": ([ASK, {"role": "assistant", "content": "", "tool_calls": [WEATHER_CALL]},
                     result("c1", "get_weather", '{"output": "Sunny", "chunk_id": 7}')], [WEATHER], {}),
}

# Template variables at each of their values, per family, over the cases where they change the rendering.
VARIABLES = {
    "qwen": {
        "thinking_off": ("system", {"enable_thinking": False}),
        "effort_low": ("system", {"reasoning_effort": "low"}),
        "effort_medium": ("system", {"reasoning_effort": "medium"}),
        "effort_xhigh": ("system", {"reasoning_effort": "xhigh"}),
        "effort_low_thinking_off": ("system", {"enable_thinking": False, "reasoning_effort": "low"}),
        "closed_cycle_preserve_off": ("closed_cycle", {"preserve_thinking": False}),
        "one_call_resume_thinking_off": ("one_call_resume", {"enable_thinking": False}),
    },
    "llama31": {
        "tools_in_system": ("tools", {"tools_in_user_message": False}),
        "dated": ("system", {"date_string": "14 Oct 2026"}),
    },
    "llama32": {
        "tools_in_system": ("tools", {"tools_in_user_message": False}),
        "dated": ("system", {"date_string": "14 Oct 2026"}),
    },
    "gemma": {
        "thinking_on": ("system", {"enable_thinking": True}),
        "thinking_on_no_system": ("plain", {"enable_thinking": True}),
        "tools_thinking_on": ("tools", {"enable_thinking": True}),
        "one_call_resume_thinking_on": ("one_call_resume", {"enable_thinking": True}),
        "closed_cycle_preserve_on": ("closed_cycle", {"preserve_thinking": True}),
    },
}


def load_template(family):
    directory, special = TEMPLATES[family]
    snapshots = sorted((HUB / directory / "snapshots").iterdir(), key=lambda path: path.stat().st_mtime)
    template = snapshots[-1] / "chat_template.jinja"

    if template.exists():
        return template.read_text(encoding="utf-8"), special

    # Llama's ships inside tokenizer_config.json.
    config = json.loads((snapshots[-1] / "tokenizer_config.json").read_text(encoding="utf-8"))
    return config["chat_template"], special


def render(template, special, messages, tools, variables):
    rendered, _ = render_jinja_template(
        conversations=[messages], tools=tools, chat_template=template, add_generation_prompt=True,
        **{**special, **variables})
    return rendered[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=sorted(TEMPLATES))
    parser.add_argument("--case")
    arguments = parser.parse_args()

    template, special = load_template(arguments.family)
    cases = {name: case for name, case in CASES.items()}

    for name, (base, variables) in VARIABLES[arguments.family].items():
        messages, tools, _ = CASES[base]
        cases[name] = (messages, tools, variables)

    for name, (messages, tools, variables) in cases.items():
        if arguments.case and name != arguments.case:
            continue

        try:
            text = render(template, special, messages, tools, variables)
        except Exception as error:
            print(f"// {name}: the template refuses it: {error}\n")
            continue

        print(f"// {name}\nR\"golden({text})golden\"\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
