# Gemma 4 Chat Protocol

Authoritative source:
[ai.google.dev/gemma/docs/core/prompt-formatting-gemma4](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4).
This document records the protocol and how the Mila chat harness
(`Mila/Samples/Chat/Src/`) implements it.

## Control tokens

| Token            | Role                                            |
|------------------|-------------------------------------------------|
| `<|turn>` / `<turn|>` | Begin / end a dialogue turn                 |
| `system` / `user` / `model` | Role markers (follow `<|turn>`)      |
| `<|think|>`      | Activates thinking mode (in the system turn)    |
| `<|channel>` / `<channel|>` | Delimit the internal reasoning channel |
| `thought`        | Channel name emitted after `<|channel>`         |
| `<|tool_call>` / `<tool_call|>` | Model requests a tool            |
| `<|tool_response>` / `<tool_response|>` | Tool result returned     |
| `<|image|>` / `<|audio|>` | Multimodal embedding placeholders      |

## Turn structure

```
<|turn>user
[content]<turn|>
<|turn>model
[response]<turn|>
```

## Response structure (thinking enabled)

```
<|channel>thought
[internal reasoning]
<channel|>[final answer]<turn|>
```

The reasoning sits between `<|channel>` (after the `thought` label) and
`<channel|>`; the final answer follows `<channel|>`. With thinking **disabled**
the model emits an empty `thought` channel and the answer follows `<channel|>`.

## Enabling thinking

`<|think|>` is placed in the system instruction:

```
<|turn>system
<|think|>You are a helpful assistant.<turn|>
```

## Mila implementation status

- **Response parsing** — `Chat.ChannelParser` splits on `<|channel>` / `<channel|>`,
  routing reasoning to the dim "Thinking" block (shown only with `/model ... thinking`)
  and the answer to history. Confirmed against captured output.
- **Thinking activation** — `formatGemmaPrompt` emits a dedicated `<|turn>system`
  turn led by the `<|think|>` trigger when thinking is enabled (folding the trigger
  into the user turn does not activate thinking).
- **Token stripping** — `stripSpecialTokens` removes the turn/think/tool tokens
  (channel markers are consumed by the parser first).
- **Token registration** — `BpeVocabulary::loadGemma` registers the Gemma 4 control
  tokens (`<|think|>`, `<|turn>`/`<turn|>`, `<|channel>`/`<channel|>`, the tool
  tokens) from the loaded vocabulary, so the prompt encodes them atomically. Tokens
  absent from a given checkpoint are skipped.

## Resolved

- **Turn delimiters.** Confirmed empirically: the checkpoint vocabulary contains
  `<|turn>` / `<turn|>` (and `<|think|>`, `<|channel>` / `<channel|>`) but NOT the
  Gemma 3-style `<start_of_turn>` / `<end_of_turn>` — `loadGemma` warns on the absent
  pair. `formatGemmaPrompt` and the tokenizer registration use `<|turn>` / `<turn|>`.

## Open questions / to verify (use `/raw` to capture)

- **Thinking activation end-to-end.** With the correct `<|turn>` delimiters and the
  `<|think|>`-led system turn, confirm via `/raw` that the model emits a *populated*
  `<|channel>thought ... <channel|>` and the dim Thinking block renders.
