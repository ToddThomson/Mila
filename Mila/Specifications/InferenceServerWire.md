# Inference Server Wire

How the inference server turns each API's request into one conversation, and one parsed reply back into
that API's response, with no knowledge of which model family it is serving. The family's grammar is the
library's, behind the protocol interface of `ModelHandle.md` 3.4; this document is the server's half.

Written 2026-10-13 at `0.21.0-dev+67`. **Draft.** It depends on `ModelHandle.md` Phase 2, which is itself a
draft, and changes nothing in `Mila/Src` beyond what that phase proposes and the amendment in section 3.2.

---

## 1. Problem Statement

The server speaks three APIs -- OpenAI Chat Completions, OpenAI Responses and Anthropic Messages -- for three
families, each buffered or streamed. Each endpoint reduces its request to "system text, a text-only history
and the last user message", then reaches the family's grammar through a per-family branch. Tool handling was
added to two endpoints, separately, on top of that reduction, and never to the third.

### 1.1 Chat Completions, audited 2026-10-13

A plain single-turn text request works. Everything past it is mishandled
(`protocols/openai/chat.py`):

| # | Defect | Consequence |
|---|---|---|
| 1 | The reply is never parsed for a tool call | the call reaches the client as raw grammar in `content`, `finish_reason: stop` |
| 2 | The last message is assumed to be the user's | a `role: "tool"` result, the normal last message after a call, is rendered as a user turn |
| 3 | An assistant message's `tool_calls` is not read, only its `content` (usually `null`) | the model's own earlier call disappears from the history it is shown |
| 4 | `content` as a list of parts is not extracted | the list reaches the renderer unread; Responses and Messages extract it |
| 5 | `developer` is not a role it knows | OpenAI's current name for `system` is replayed as an ordinary turn |
| 6 | Unrecognised fields are ignored silently | `max_completion_tokens`, `tool_choice`, `parallel_tool_calls`, `response_format`, `n`, `seed`, `logprobs` and `stream_options` change nothing, and the reply does not say so |
| 7 | Every stream chunk carries a new id, and the first carries no `role` | a stream that does not identify itself as one completion |

Measured against Qwen3.8-27B-fp4 on the RTX 5060 Ti: Canberra for a factual prompt, and for a weather tool
the raw `<tool_call>` span in `content`.

### 1.2 The same split broke the endpoints that do have tools

- **Buffered Responses never received the raw collect** that buffered Messages had, so a Gemma call was
  stripped of its markers and repeated to the token budget, and a reply after a tool result began with a
  leaked `thought` label. Found on the Gemma 4 26B-A4B and fixed at `+67` (`routes/factory.py`
  `_generate_buffered`); it had affected the 12B since the Responses path was written. Codex never saw it
  because it streams.
- **Each endpoint carries its own family branches**: `model_worker.py` 10, `responses.py` 8, `messages.py` 6,
  `models.py` 5, `routes/factory.py` 4, `prompt.py` 2, beside two bridge modules and Llama's template in
  `prompt.py`.
- **Only the first tool call of a turn survives**, on every endpoint: the library's `parseToolCall` returns one
  `optional<ToolCall>`, and so do both bridges. OpenAI's `tool_calls` and Anthropic's content blocks are
  lists.

The defects are not individually hard. They are what a design produces in which each (API, family, buffered or
streamed) combination is its own code path: three APIs, three families and two modes are eighteen, and a fix
lands in the one a client happened to exercise.

### 1.3 Why Chat has not met this

Chat has the same family dependence -- `ModelHandle.md` section 1 counts it -- and has been bitten by the same
drift (`BACKLOG.md`: "Chat renders Gemma's prompt itself, and its template and the library's have drifted
apart"). What it lacks is the multiplication. Chat has one front door, one conversation type of its own and
one tool loop that it runs: it decides what is offered, executes the call and replays the result itself, so it
only ever renders histories it wrote. The server runs no tool loop. The client does, and sends back whatever
history its own loop produced, in its own API's shape, buffered or streamed by its own choice. Every path of
Chat is exercised by a person each session; a path of the server is exercised only by a client that uses it,
and the clients run here so far -- Codex over streamed Responses, Claude Code over streamed Messages -- touched
two of the eighteen.

---

## 2. What Is Model-Dependent

An endpoint does two kinds of work.

- **Wire work**: read the API's JSON, validate it, and write the response in the API's shape -- stream events,
  ids, end reasons, usage. It is identical for every model.
- **Grammar work**: render a conversation and its tool declarations as the family's prompt; parse the family's
  raw output into reasoning, text and tool calls; know the text that ends a turn or a call. It varies by
  family, never by model within one.

The grammar work is exactly what `ModelHandle.md` 3.4's `Protocol` answers. So the endpoints need no family
knowledge at all, provided the protocol is complete on the reply side (3.2).

---

## 3. Design

### 3.1 One conversation, from every API

Each endpoint's parser produces a `Conversation` (`Dnn/Models/Conversation.ixx`): turns of role `System`,
`User`, `Assistant` or `Tool`, an assistant turn carrying its `tool_calls`, plus the request's tool
declarations as JSON. That type exists and both library protocols already render it; the server reduces to
text before reaching it today.

| API | System | Earlier call | Tool result |
|---|---|---|---|
| Chat Completions | `system` and `developer` messages | assistant message `tool_calls` | `role: "tool"`, by `tool_call_id` |
| Responses | `instructions` and system items | `function_call` item | `function_call_output` item, by `call_id` |
| Messages | `system` | assistant `tool_use` block | user `tool_result` block, by `tool_use_id` |

Correlation ids are wire bookkeeping: `Conversation::ToolCall` has none, because no template renders one. Each
parser resolves a result's id to its call when it builds the turns and sets the `Tool` turn's `tool_name` from
it (`ModelHandle.md` 3.4), keeping the order the client sent; an id that names no earlier call is refused with a
400, never rendered under a guessed name. The formatter mints an id for a new call -- for Gemma the existing
digest of name and arguments (`ModelHandle.md` section 5).

Content given as parts is reduced to its text parts; a part the server cannot pass to the model (an image, a
file) is refused with a 400 naming it, never dropped.

### 3.2 One reply, parsed once -- an amendment to `ModelHandle.md` 3.4

`Protocol::parseToolCall` returns one call. The server needs the whole turn:

```cpp
struct Reply
{
    std::string reasoning;                    // the reasoning channel's text, empty when there was none
    std::string content;                      // what the user is shown
    std::vector<ToolCall> tool_calls;         // every call, in the order the model made them
    std::string malformed_tool_call;          // a call span the grammar could not read; empty when none
};

virtual Reply parseReply( std::string_view text ) const = 0;
```

It replaces `parseToolCall` in the interface and subsumes `Gemma::extractAnswer`, both bridges' `answer_text`,
and the server's `clean_response_text`. A malformed call is reported, never dropped:
`Reply::malformed_tool_call` carries the span as the model wrote it (`ModelHandle.md` 3.4). What it does to a
live request -- prose, logged -- stays the server's. Carried to
`ModelHandle.md` 3.4 in this change.

### 3.3 Family-blind endpoints

An endpoint is then a parser to `Conversation`, a formatter from `Reply`, and nothing else. The prompt comes
from `protocol.formatPrompt( conversation, options )`, the reply from `protocol.parseReply( text )`, the stops
from `protocol.markers()`. `ModelFamily`, `SESSION_FOR`, `_NATIVE_TOOL_FAMILIES`, `_TOOL_CAPABLE_FAMILIES`,
both bridges, `prompt.py` and every `loaded.family` branch are deleted. A new family needs no server change,
and a new API needs only its parser and formatter.

How the server obtains the protocol depends on the release (10.1): a binding object over the library's
`Protocol` in v0.21, or the handle's `protocol()` once the server consumes `Mila::AI`. The endpoint code is the
same either way, which is what makes doing it before the handle worthwhile.

### 3.4 One generation path

A reply is generated once, by the streaming primitive, as raw text. Buffered means collected, then parsed;
streamed means forwarded as it arrives. The blocking `generate()` plus `decode()` path, which strips the
grammar and has no stop at the end of a call, is no longer used by the server: its differences from the
streaming path are the class of defect in 1.2.

### 3.5 Streaming a turn that may contain a call

Today a Gemma or Qwen turn streamed over Messages is held until the turn ends, because the block type --
text or tool use -- must be sent before its content and the parse runs after the model finishes. That is a
limitation of where the server parses, not of the APIs. The open marker of a call is one control token
(`markers().tool_call_open`), so text is forwarded live until it appears, and only the call itself is
buffered, then emitted as the API's tool block. Reasoning is routed by the same markers.

### 3.6 Honour or refuse

Every field an API defines is either acted on or refused with a 400 naming it. None is ignored: a client that
asked for something and got a normal-looking reply cannot tell it was not done. Fields outside the API's
definition are left alone, as OpenAI's and Anthropic's servers treat them.

Decided 2026-10-14 (Todd, "go with your recommendations for now", reviewed by running Codex and Claude Code). A
field that only describes the request and changes no reply -- `user`, `metadata`, `store: false` -- is accepted;
a refusal is for a field whose value would change the reply and cannot be acted on, so a field sent at its
API's default (`n: 1`, `frequency_penalty: 0`) is accepted too. No endpoint writes a system turn the request did
not send: the "You are a helpful assistant." each one supplied is gone, and an absent system message is the
template's to render.

**System text wherever an API places it** (decided 2026-10-15). Each API has more than one place for system
text: Responses' `instructions` and `developer` / `system` items, Chat Completions' `system` and `developer`
messages, Messages' `system` field and the `role: "system"` messages Anthropic defines inside `messages`
([mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)).
Templates take less: Qwen 3.8 accepts one system turn, first; Gemma 4 renders one anywhere. The harnesses send
more -- Codex 0.162.1 opens with two `developer` messages, and Claude Code 2.1.300 sends a `system` message after
the first user message on every request. So:

- Every system turn before the model's first reply joins one leading system turn, in the order sent, separated by
  a blank line. A system message applies from where it stands onward, and every reply the model writes stands
  after it, so the joined turn is what each of them was told.
- A system turn after a model reply stays where it was sent, for the template: Gemma renders it there, and Qwen's
  refusal is a 400 naming it. Moving it ahead of the replies that preceded it would tell the model they followed
  it.
- A system turn with no text renders nothing and is dropped.
- Messages' `role: "system"` is placed as Anthropic places it: after a `user` message, and last or before an
  `assistant` message; any other position is a 400, as it is there. `clear_at` other than `never` and a
  message's own `output_config` are refused.

**Reasoning reaches the client in each API's own shape** (decided 2026-10-14): Chat Completions in
`message.reasoning_content` (and `delta.reasoning_content` streamed), which OpenAI does not define and a client that
does not know it ignores; Responses as a `reasoning` output item with its text as `summary_text`; Messages as a
`thinking` content block. Reasoning a client sends back -- `reasoning_content`, a `reasoning` item, a `thinking`
block -- becomes the assistant turn's `Turn::reasoning`, which the template renders or drops by its own rule.

Chat Completions:

| Field | Decided | Note |
|---|---|---|
| `messages`, `tools` | honour | 3.1; `developer` is `system`; an assistant's `reasoning_content` is its reasoning |
| `max_completion_tokens`, `max_tokens` | honour | the first wins when both are sent |
| `temperature`, `top_p`, `stop`, `stream` | honour | |
| `stream_options.include_usage` | honour | a final usage chunk |
| `tool_choice` | honour `auto` and `none`; refuse `required` and a named function | 10.3: asking is not doing |
| `parallel_tool_calls` | honour | `false` keeps the first call of a turn |
| `seed` | refuse | the binding takes no seed (10.4) |
| `n` > 1 | refuse | one completion per request |
| `logprobs`, `top_logprobs` | refuse | not exposed by the binding |
| `frequency_penalty`, `presence_penalty`, `logit_bias` other than their defaults | refuse | the sampler has none |
| `response_format` other than `text` | refuse | Mila has no constrained decoding |
| `reasoning_effort` | `none` sets the template's `enable_thinking` false, and is met as sent by a template without it. A level: a template with its own `reasoning_effort` variable takes it by exact name and refuses any other, naming what is accepted; a template with only `enable_thinking` turns thinking on; a template with neither refuses it | Decided 2026-10-15. OpenAI's effort is its reasoning switch -- `none` is off, a level is on and how much. A level is never mapped onto another model's level (`ModelHandle.md` 3.4); a thinking-only template has one depth, so its levels are not told apart. Absent: the template's default (10.6) |
| image or audio content parts | refuse | 3.1 |

Responses:

| Field | Decided | Note |
|---|---|---|
| `input` (text, `message`, `function_call`, `function_call_output`, `reasoning` items), `instructions`, `tools` | honour | 3.1; any other item type is refused, naming it |
| `tools` of a type other than `function` | not offered, logged | a built-in tool the server cannot run is never declared to the model |
| `max_output_tokens`, `temperature`, `top_p`, `stream` | honour | |
| `tool_choice`, `parallel_tool_calls` | as Chat Completions | |
| `reasoning.effort` | as Chat Completions' `reasoning_effort` | |
| `reasoning.summary` | honour | the reasoning is returned whole as its summary |
| `text.format` other than `text` | refuse | no constrained decoding |
| `previous_response_id`, `store: true` | refuse | the server keeps no state between requests |
| `include` | honour `reasoning.encrypted_content` | the reasoning travels in the item and comes back in it; nothing is encrypted, and the client never reads it |

Messages:

| Field | Decided | Note |
|---|---|---|
| `messages` (text, `tool_use`, `tool_result`, `thinking` blocks; `system` messages placed as above), `system`, `tools` | honour | 3.1; a `tool_result` is split out of its user message as a `Tool` turn (5) |
| `tools` with a server-side `type` | not offered, logged | as Responses |
| `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences`, `stream`, `metadata` | honour | |
| `tool_choice` | honour `auto` and `none`; refuse `any` and `tool` | 10.3 |
| `tool_choice.disable_parallel_tool_use` | honour | keeps the first call |
| `thinking.type` | `adaptive` sets the template's `enable_thinking` true, `disabled` sets it false; for a template without the variable, both are met as sent -- the model does not think, which `adaptive` allows. `enabled` needs `budget_tokens`, which is refused | Decided 2026-10-15: `adaptive` is Anthropic's "on", with how much left to the model, and Claude Code sends it on every request; leaving `thinking` out is the template's default. No template has a thinking budget |
| `thinking.display` | `omitted` returns the thinking block with empty text and the reasoning in its `signature`, from which a block sent back is read; `summarized` as absent | as Responses carries reasoning in `encrypted_content`: the client never reads it, and the template gets back what the model wrote |
| `output_config.effort` | honour a level the template's own `reasoning_effort` variable names; refuse any other, and refuse the field for a template without the variable, naming what is accepted | Decided 2026-10-15. Anthropic's effort shapes all output and Qwen's variable only its reasoning: the nearest the template has, by exact name, both being guidance. Claude Code sends it only when an effort is set -- at `auto` it sends none |
| `output_config.format`, any other `output_config` key | refuse | no constrained decoding, as Responses' `text.format` |
| `context_management.edits` | `clear_thinking_20251015` with `keep: "all"` sets the template's `preserve_thinking` true, and is met as sent by a template without it; `keep` as a turn count, `clear_tool_uses_20250919` and any other edit are refused | Decided 2026-10-15. Claude Code sends keep-all on every request; `preserve_thinking` is the model's own switch for it. Gemma's keeps the reasoning of earlier turns that called tools, not of those that only answered -- the nearest its format has |
| image and document blocks, `redacted_thinking` | refuse | 3.1 |

---

## 4. Alternatives Rejected

**Add tools to `chat.py` as Responses and Messages added them.** The fix is a day; it is the third copy of
the same family branching, and 1.2 is what the first two copies produced.

**A server-side conversation type in Python.** `Conversation` already exists in the library, and every protocol
renders it. A Python mirror is a second definition of the same shape that the binding would translate
between.

**Parse in the server, from the markers.** The markers say where a call is, not how its arguments are written
-- Gemma's `<|"|>` strings and Qwen's parameter blocks are grammar. Parsing is the library's
(`Direction.md` 3.1).

---

## 5. Traps

- **A tool result has a role in the conversation, not in the wire.** Anthropic sends it inside a user message;
  the parser must split it out as a `Tool` turn, or Qwen renders it as user text -- defect 2 in another form.
- **Gemma resumes an open model turn after a result** (`continue_open`). That is the protocol's to decide from a
  conversation ending in a `Tool` turn, never a flag an endpoint passes.
- **Gemma's 12B and 26B take the empty thought channel on the prompt when thinking is off**
  (`GemmaChatProtocol.md`). The server removed it once, in July 2026, and every tool call failed. A family-blind
  server must not lose it by rendering through an option it does not set.
- **A refusal is a behaviour change for a client that sends a field today and is silently served.** The 400s in
  3.6 are announced in the release notes.
- **The foreign-harness flows are the regression gate** (`ROADMAP.md`, both releases): Codex CLI and Claude
  Code CLI pass unchanged after every step.

---

## 6. Validation

Bounds written before any run; each gate forced to fail once. Every gate runs for Gemma 4 12B, Gemma 4
26B-A4B, Qwen 3.8 27B and Llama 3.1 8B, on both cards.

- **W1. One round trip per API.** For each API, buffered and streamed: a tool is offered, the model calls it,
  the client returns a result, the model answers from it. The call reaches the client in the API's own shape
  and the answer contains the result's value.
- **W2. Two calls in one turn** reach the client as two calls, on every API.
- **W3. Same request, same reply.** Buffered and streamed replies to one seeded request carry the same text
  and the same calls.
- **W4. Every refused field refuses**, and every honoured field changes the reply (a test per row of 3.6).
- **W5. No family branch.** A search of the server's source for a family name finds none outside its tests.
- **W6. The harnesses.** Codex CLI and Claude Code CLI complete a file-reading task against each family.
- **Negatives.** N1: the tool result rendered as a user turn -- W1 fails for Qwen. N2: `parseReply` keeping only
  the first call -- W2 fails. N3: the buffered path generating through `decode()` -- W1 fails for Gemma.

---

## 7. Phasing

**Phase 1 -- the protocol, complete.** `ModelHandle.md` Phase 2 with the amendment in 3.2: `Reply`,
`parseReply` returning every call, the three families' implementations, Llama's fold, Gemma's renderers made
one. Parity first (`ModelHandle.md` P1). Library only.

**Phase 2 -- the binding.** One protocol object for the loaded model, replacing the eleven per-family grammar
functions (`gemma_format_prompt`, `qwen_parse_tool_call` and the rest).

**Phase 3 -- the server.** The three endpoints as parsers and formatters over it (3.1, 3.3), one generation path
(3.4), the honour-or-refuse tables (3.6). *Exit:* W1-W6 and the negatives.

**Phase 4 -- live streaming of tool turns** (3.5). Independent of the rest once Phase 3 has landed.

---

## 8. Non-Goals

- A tool loop in the server. The client runs it; the server is stateless per request.
- Constrained decoding for `response_format` or for a forced `tool_choice`. Refused until the library has it.
- APIs beyond the three.

---

## 9. Relationship To Other Specs

- `ModelHandle.md` -- 3.4's protocol is the interface this document consumes; 3.2 here amends it.
- `GemmaChatProtocol.md` -- Gemma's grammar, including the primer and the resume.
- `MilaISCodexAgent.md` -- the Llama tool bridge in `tool_bridge.py`, deleted when Llama's grammar is the
  library's.
- `Direction.md` 3.1 -- the grammar is the library's.

---

## 10. Open Decisions

1. **The release.** v0.21.0's Applications Ship asks that the server run every published model *correctly*;
   v0.22.0's Applications asks that it contain no model-specific code, once it consumes `Mila::AI`. The second
   is where the server is going -- Mila as a family-blind harness over models (Todd, 2026-10-13) -- so v0.21's
   correctness is reached on that road, not beside it. Phases 1-3 meet both, and the endpoint code is the same
   either side of the handle (3.3). Recommend Phases 1-3 in v0.21.0, Phase 4 in v0.22.0. The alternative --
   patch Chat Completions now and do this in v0.22.0 -- is the rejected alternative in section 4.
2. **Where the server gets the protocol in v0.21.** A binding object per loaded model (Phase 2), or the free
   functions behind one family-neutral binding call. Recommend the object: it is what `Mila::AI` will hand
   out.
3. **`tool_choice` beyond `auto` and `none`.** `required` and a named function can be asked of the model in the
   prompt, or enforced, which needs constrained decoding. Asking is not doing (3.6). Recommend refusing them
   until they can be enforced.
4. **`seed`.** Honour it, which adds a seed to the binding's generate, or refuse it.
5. **Llama in the server.** Llama's tool grammar is the server's own copy until Phase 1 folds it. Recommend no
   Llama tool support on Chat Completions before then -- a Llama tool request is refused -- rather than built
   on the copy Phase 1 deletes.
6. **Reasoning when a request names no mode.** *Decided 2026-10-14 (Todd): the template's default.* The
   server imposes no made-up setting, as Chat does not (`ModelHandle.md` 3.4); today every endpoint fixes
   thinking off for every model ("a harness wants the answer"), which is the server choosing for the model.
   With the template's default Qwen 3.8 thinks and Gemma 4 does not, which changes what Codex and Claude Code
   receive from Qwen, so it lands with Phase 3, once the reasoning channel reaches the client as the API's
   reasoning block rather than as text.
