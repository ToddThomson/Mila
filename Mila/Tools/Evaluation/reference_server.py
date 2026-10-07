"""Serve a HuggingFace model on OpenAI's Completions route, with the semantics MIS has there.

The reference arm for a harness that talks to a server rather than loading a model itself --
BFCL, which expects one. It answers exactly what MIS answers on /v1/completions: a prompt as
text or token ids, one per request; greedy decoding only; stop sequences cut the reply; the
end-of-turn token is not part of the reply and every other control token is. So two runs of the
same harness, one against this server and one against MIS, differ in the engine and nothing else.
"""

import argparse
import asyncio
import concurrent.futures
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer


def invalid_request(message):
    content = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}

    return JSONResponse(status_code=400, content=content)


def read_prompt(prompt):
    """MIS's rule: text, token ids, or a batch of exactly one of either. Returns (text, ids)."""
    if isinstance(prompt, str):
        return prompt, []

    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a string, a list of token ids, or a list holding one of either")

    if all(isinstance(token, int) for token in prompt):
        return "", list(prompt)

    if len(prompt) != 1:
        raise ValueError(f"prompt holds a batch of {len(prompt)}; this server completes one prompt per request")

    return read_prompt(prompt[0])


def read_stop(stop):
    if stop is None:
        return []

    if isinstance(stop, str):
        stop = [stop]

    return [sequence for sequence in stop if sequence]


def truncate_at_stop(text, stop):
    positions = [text.find(sequence) for sequence in stop]
    found = [position for position in positions if position >= 0]

    if not found:
        return text

    return text[:min(found)]


class Reference:
    """One model, one request at a time: a batch pads, and padding changes a greedy reply."""

    def __init__(self, model_id, device, context_length):
        self.name = model_id
        self.device = device
        self.context_length = context_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        eos = self.model.generation_config.eos_token_id

        if eos is None:
            eos = []
        elif isinstance(eos, int):
            eos = [eos]

        self.end_of_turn = set(eos) | {self.tokenizer.eos_token_id}

    def complete(self, prompt_ids, max_tokens, extra_stop_ids):
        """Greedy continuation, without the token that ended it. Returns (ids, finished)."""
        end_ids = sorted(self.end_of_turn | set(extra_stop_ids))
        inputs = torch.tensor([prompt_ids], device=self.device)
        mask = torch.ones_like(inputs)

        with torch.no_grad():
            output = self.model.generate(
                inputs,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=max_tokens,
                eos_token_id=end_ids,
                pad_token_id=end_ids[0],
            )

        new_ids = output[0][len(prompt_ids):].tolist()
        finished = bool(new_ids) and new_ids[-1] in end_ids

        if finished:
            new_ids = new_ids[:-1]

        return new_ids, finished


def build_app(reference):
    app = FastAPI(title="Mila evaluation reference server")

    @app.get("/v1/models")
    async def models():
        card = {"id": reference.name, "object": "model", "owned_by": "reference", "context_window": reference.context_length}

        return {"object": "list", "data": [card]}

    @app.post("/v1/completions")
    async def completions(http_request: Request):
        body = await http_request.json()

        try:
            text, prompt_ids = read_prompt(body.get("prompt", ""))
        except ValueError as error:
            return invalid_request(str(error))

        if body.get("temperature", 0) > 0:
            return invalid_request("the reference decodes greedily only; send temperature 0")

        if not prompt_ids:
            prompt_ids = reference.tokenizer.encode(text, add_special_tokens=False)

        remaining = reference.context_length - len(prompt_ids)

        if remaining <= 0:
            return invalid_request(f"Prompt length {len(prompt_ids)} tokens exceeds context_length {reference.context_length}.")

        max_tokens = min(body.get("max_tokens", 256), remaining)
        extra_stop_ids = body.get("stop_token_ids") or []
        loop = asyncio.get_running_loop()
        new_ids, finished = await loop.run_in_executor(
            reference.executor, reference.complete, prompt_ids, max_tokens, extra_stop_ids)
        # Control tokens stay in the text, as MIS's decode keeps them.
        reply = reference.tokenizer.decode(new_ids, skip_special_tokens=False)
        reply = truncate_at_stop(reply, read_stop(body.get("stop")))
        choice = {"text": reply, "index": 0, "finish_reason": "stop" if finished else "length"}
        usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(new_ids),
            "total_tokens": len(prompt_ids) + len(new_ids),
        }

        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": reference.name,
            "choices": [choice],
            "usage": usage,
        }

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="HuggingFace model id or local path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-length", type=int, default=8192,
                        help="prompt and reply together; set it as MILA_CONTEXT_LENGTH is set for MIS")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    arguments = parser.parse_args()

    reference = Reference(arguments.model, arguments.device, arguments.context_length)
    uvicorn.run(build_app(reference), host=arguments.host, port=arguments.port, log_level="warning")


if __name__ == "__main__":
    main()
