"""BFCL handler for Qwen 3.8: the prompt is the model's own chat template, and calls are read in the form it writes.

bfcl-eval's Qwen handlers stop at Qwen3, whose calls are JSON. Qwen 3.8 writes

    <tool_call>
    <function=name>
    <parameter=key>
    value
    </parameter>
    </function>
    </tool_call>

with every value as text, so a value's type comes from the function's schema. BFCL decodes a reply from the stored
result alone, where no schema is in reach, so the calls are typed here when the reply is parsed, from the entry's
own schema. The result BFCL stores is `{"calls": [...], "reply": "..."}`: the typed calls it scores, and the reply as
written, so a malformed call -- which yields no call -- can still be read.

The prompt is the template's default -- for Qwen 3.8, thinking on at its default effort -- since a model is measured
as it is published to run. Calls are read from the answer after the reasoning only.
"""

import ast
import json
import re
from typing import Any

from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.utils import convert_to_function_call, convert_to_tool
from overrides import override

REGISTRY_NAME = "Qwen/Qwen3.8-27B-FC"

TOOL_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAMETER = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


def typed_value(text, schema):
    """A parameter's text as the type its schema declares; text that does not parse as that type stays text.

    A parameter with no schema -- of a function the prompt did not declare -- is read as JSON or a Python literal
    where it parses as one.
    """
    # The template writes a newline either side of every value.
    if text.startswith("\n"):
        text = text[1:]

    if text.endswith("\n"):
        text = text[:-1]

    kind = schema.get("type") if schema else None

    if kind == "string":
        return text

    if text.strip().lower() == "null":
        return None

    try:
        if kind == "integer":
            return int(text.strip())

        if kind == "number":
            number = text.strip()

            return float(number) if any(mark in number.lower() for mark in (".", "e", "inf", "nan")) else int(number)

        if kind == "boolean":
            lowered = text.strip().lower()

            if lowered in ("true", "false"):
                return lowered == "true"

            return text
    except ValueError:
        return text

    try:
        return json.loads(text)
    except ValueError:
        pass

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def split_reasoning(reply):
    """The reply's reasoning and its answer.

    The template's default opens the reasoning block in the prompt, so a reply is the reasoning, `</think>`, then the
    answer. A reply cut off before `</think>` has no answer, so a call sketched while reasoning is never read as one
    made.
    """
    if "</think>" not in reply:
        return reply.strip(), ""

    reasoning, _, answer = reply.partition("</think>")

    return reasoning.replace("<think>", "").strip(), answer.strip()


def parse_calls(reply, tools):
    """Every well-formed call in the reply, its arguments typed by the named tool's schema."""
    schemas = {tool["function"]["name"]: tool["function"]["parameters"].get("properties", {}) for tool in tools}
    calls = []

    for name, body in TOOL_CALL.findall(reply):
        name = name.strip()
        properties = schemas.get(name, {})
        arguments = {key.strip(): typed_value(value, properties.get(key.strip()))
                     for key, value in PARAMETER.findall(body)}
        calls.append({"name": name, "arguments": arguments})

    return calls


class Qwen38FCHandler(QwenFCHandler):
    """QwenFCHandler with Qwen 3.8's template for the prompt and its XML form for the calls."""

    def __init__(self, model_name, temperature, registry_name, is_fc_model, dtype="bfloat16", **kwargs):
        super().__init__(model_name, temperature, registry_name, is_fc_model, dtype, **kwargs)
        self._tools = []

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        self._tools = convert_to_tool(test_entry["function"], GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)

        return {"message": [], "function": self._tools}

    @override
    def _format_prompt(self, messages, function):
        return self.tokenizer.apply_chat_template(
            messages, tools=function or None, add_generation_prompt=True, tokenize=False)

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        reply = api_response.choices[0].text
        reasoning, answer = split_reasoning(reply)
        calls = parse_calls(answer, self._tools)

        if calls:
            history = {"role": "assistant", "content": "", "reasoning_content": reasoning,
                       "tool_calls": [{"type": "function", "function": call} for call in calls]}
        else:
            history = {"role": "assistant", "content": answer, "reasoning_content": reasoning}

        return {
            "model_responses": json.dumps({"calls": calls, "reply": reply}, ensure_ascii=False),
            "reasoning_content": reasoning,
            "model_responses_message_for_chat_history": history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @staticmethod
    def _stored_calls(result):
        stored = json.loads(result)

        if not isinstance(stored, dict) or not isinstance(stored.get("calls"), list):
            raise ValueError(f"Not a stored Qwen 3.8 reply: {result}")

        return stored["calls"]

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        return [{call["name"]: call["arguments"]} for call in self._stored_calls(result)]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        return convert_to_function_call([{call["name"]: call["arguments"]} for call in self._stored_calls(result)])


def register():
    """Add the handler to BFCL's registry under REGISTRY_NAME."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

    MODEL_CONFIG_MAPPING[REGISTRY_NAME] = ModelConfig(
        model_name="Qwen/Qwen3.8-27B",
        display_name="Qwen3.8-27B (FC)",
        url="https://huggingface.co/Qwen/Qwen3.8-27B",
        org="Qwen",
        license="apache-2.0",
        model_handler=Qwen38FCHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        # convert_to_tool writes a dotted name with underscores, so the checker maps them back.
        underscore_to_dot=True,
    )
