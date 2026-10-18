"""lm-eval's served model at the model's own defaults, and lm-eval's command line to run it.

A task's `generation_kwargs` are written for greedy decoding and short replies, which measures a configuration nobody
runs when the model reasons by default. `served-defaults` keeps the task's `max_gen_toks` for what lm-eval uses it
for -- how much of a long prompt to keep -- and changes only what is sent: no sampling knob, so the server applies
the model's recommended setting, and a reply budget the server lowers to what its context has left. The prompt is
the template's default, reasoning included. With `reasoning_end` given, the reply is scored from after it, and a
reply that never reaches it has no answer; every raw reply is appended to `raw_replies` when that is given.
Run as a script, this registers the model and hands the command line to lm-eval.
"""

import json

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI

# What a request may reply with; the server lowers it to what its context has left.
REPLY_BUDGET = 32768


@register_model("served-defaults")
class ServedDefaultsAPI(LocalCompletionsAPI):
    def __init__(self, reasoning_end=None, raw_replies=None, **kwargs):
        super().__init__(**kwargs)
        self.reasoning_end = reasoning_end or None
        self.raw_replies = raw_replies or None

    def _create_payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kwargs):
        payload = super()._create_payload(messages, generate=generate, gen_kwargs=gen_kwargs, seed=seed, eos=eos,
                                          **kwargs)

        if generate:
            for knob in ("temperature", "top_p", "top_k", "seed"):
                payload.pop(knob, None)

            payload["max_tokens"] = REPLY_BUDGET

        return payload

    def reasoning_open(self, context):
        """Whether the prompt leaves the template's reasoning block open for the reply to continue.

        A task that writes the start of the answer into the assistant's turn -- RULER does -- has the template close
        the block in the prompt, and the whole reply is the answer.
        """
        if not isinstance(context, str):
            context = self.tokenizer.decode(context)

        start = self.reasoning_end.replace("</", "<", 1)

        return context.rfind(start) > context.rfind(self.reasoning_end)

    def answer(self, reply, context):
        """The reply from after the reasoning; empty when the reasoning it opened never ended."""
        if self.reasoning_end is None or (context is not None and not self.reasoning_open(context)):
            return reply

        _, found, answer = reply.partition(self.reasoning_end)

        return answer.strip() if found else ""

    def parse_generations(self, outputs, contexts=None, **kwargs):
        replies = LocalCompletionsAPI.parse_generations(outputs, **kwargs)
        contexts = list(contexts) if contexts is not None else [None] * len(replies)

        if self.raw_replies:
            with open(self.raw_replies, "a", encoding="utf-8") as log:
                for reply in replies:
                    log.write(json.dumps({"reply": reply}, ensure_ascii=False) + "\n")

        return [self.answer(reply, context) for reply, context in zip(replies, contexts)]


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
