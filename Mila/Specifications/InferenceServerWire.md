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
parser resolves an id to its call when it builds the turns, and the formatter mints one for a new call -- for
Gemma the existing digest of name and arguments (`ModelHandle.md` section 5).

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
};

virtual Reply parseReply( std::string_view text ) const = 0;
```

It replaces `parseToolCall` in the interface and subsumes `Gemma::extractAnswer`, both bridges' `answer_text`,
and the server's `clean_response_text`. A malformed call is the family's to report, as today; the server's
choice of what a malformed call does to a live request -- prose, logged -- stays the server's. Carried to
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

Proposed for Chat Completions; Responses and Messages get the same table in this document before their
endpoints move.

| Field | Proposed | Note |
|---|---|---|
| `messages`, `tools` | honour | 3.1 |
| `max_completion_tokens`, `max_tokens` | honour | the first wins when both are sent |
| `temperature`, `top_p`, `stop`, `stream` | honour | already |
| `stream_options.include_usage` | honour | a final usage chunk |
| `tool_choice` | honour `auto` and `none`; the rest open | 10.3 |
| `parallel_tool_calls` | honour | `false` keeps the first call of a turn |
| `seed` | open | the binding takes no seed today (10.4) |
| `n` > 1 | refuse | one completion per request |
| `logprobs`, `top_logprobs` | refuse | not exposed by the binding |
| `response_format` other than `text` | refuse | Mila has no constrained decoding |
| image or audio content parts | refuse | 3.1 |

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
