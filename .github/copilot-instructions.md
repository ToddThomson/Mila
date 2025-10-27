# Mila — Copilot Instructions

## Purpose
- Primary focus is the Module and Operation component redesign. All guidance and code should prioritize a correct, type-safe, device-aware module foundation.

## Code generation policy
- Generate code only when explicitly requested (e.g., "implement", "update", "write code", "generate", "create code"). Otherwise provide analysis, design guidance, and minimal examples.
- Mila is at the Alpha stage of development, please do not consider backward compatibility with previous versions when generating code.

## Doxygen / file header policy
- File-level Doxygen comments must be concise summaries (one to three short sentences).
  - Purpose: provide a quick summary of the file intent for readers and tools.
  - Must NOT repeat detailed API, implementation notes, or usage examples.
- Detailed documentation belongs in the module/class/function-level Doxygen comments (module API).
  - Put behavior, parameters, return semantics, ownership/lifetime, threading assumptions, and examples on the relevant symbol's Doxygen block.
  - Module-level comments (module partitions) should describe the public API surface and usage patterns.
- Example file-level header (preferred):
  - Brief one-line summary: "Configuration for the Residual module."
  - Optional short second sentence for scope: "Provides fluent setters used by Residual and backend factories."
- Rationale: keeps files scannable and avoids duplicated, stale documentation across many files.

## Coding Style

### Blank Lines Around Blocks
- Add blank line before control flow blocks (if, for, while, switch)
- Add blank line after closing brace of blocks
- Exception: No blank line between `} else {` or `} catch {`

### Blank Lines Around Return Statements
- Add blank line before `return` statement (final return in function)
- Exception: Early returns (guard clauses) don't need blank line
- Exception: Single-statement functions don't need blank line

## High-level constraints
- Project is alpha: breaking changes and simplifications are acceptable.
- Backward compatibility is NOT required. Do not use Deprecated APIs.
- Do not use Mila deprecated API
- Host code: C++23 using modules and module partitions. Tests: GTest. Build: CMake + Ninja.

## Comment policy
- NEVER generate trivial comments that simply restate what the code does. For example, do not generate comments like:
  - `// increment i` for the line `i++;`
  Such trivial, repetitive comments must not be produced by Copilot.
- Use only ASCII characters (no Unicode checkmarks, emojis, or special symbols)
- Don't add simple validation comments (e.g., "Good", "Correct", "OK", "Bad")
- Comments should explain WHAT the code's intent or contract is, or WHY a non-obvious approach is required — not restate HOW the code performs obvious operations.
  - Good: `// accumulate running mean across batch to avoid a second pass`
  - Good: `// Use integer index to preserve pointer stability required by the SIMD kernel`
- Prefer documenting:
  - Function/module contract: inputs, outputs, side-effects, threading assumptions, and performance/precision trade-offs.
  - Non-obvious algorithms, invariants, and corner cases that callers or maintainers must preserve.
  - API expectations: ownership, lifetime, and accumulation semantics (overwrite vs accumulate).
- Keep comments technical and informative, not evaluative or apologetic.
- Do not include reasoning or justification for design decisions in code comments (keep rationale in design documents or commit messages).
- Avoid commenting trivial lines of code that are self-explanatory; prefer a brief block comment describing the overall purpose of the surrounding code instead.
- Documentation comments (Doxygen) should describe behavior, usage, public contracts and examples — not explain why changes were made.

## Doxygen guidance for generated code
- When emitting Doxygen for symbols:
  - Use the full signature and describe preconditions, postconditions, and side-effects.
  - Prefer param/return tags for public methods.
  - Use short examples only in the symbol comment (not in file headers).
- Avoid emitting long prose in file headers; put detail in the API-level documentation.

## Notes for AI assistant
- When recommending code, prefer modern C++ idioms (RAII, smart pointers, STL algorithms).
- Always include testing suggestions and consider CPU/CUDA parity.
- In explanatory text (not code), you may use formatting symbols for clarity, but generated code comments must follow the comment policy above
- Keep commit messages and explanatory responses separate from code documentation
- Unit tests are structured by project, namespace and class — place tests under the Tests tree following the repository project layout and mirror the production namespace/class organization.
