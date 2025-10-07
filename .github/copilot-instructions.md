# Mila — Copilot Instructions

## Purpose
- Primary focus: `Tensor.ixx` system redesign. All guidance and code should prioritize a correct, type-safe, device-aware tensor foundation.

## Code generation policy
- Generate code only when explicitly requested (e.g., "implement", "update", "write code", "generate", "create code"). Otherwise provide analysis, design guidance, and minimal examples.

## High-level constraints
- Project is alpha: breaking changes and simplifications are acceptable. Backward compatibility is NOT required.
- Host code: C++23 using modules and module partitions. Tests: GTest. Build: CMake + Ninja.

## Comment policy
- Use only ASCII characters (no Unicode checkmarks, emojis, or special symbols)
- Avoid simple validation comments (e.g., "Good", "Correct", "OK", "Bad")
- Comments should explain WHY or WHAT, not judge correctness
- Keep comments technical and informative, not evaluative
- Do not include reasoning or justification for design decisions in code comments
- Documentation comments (Doxygen) should describe behavior and usage, not explain why changes were made

## Notes for AI assistant
- When recommending code, prefer modern C++ idioms (RAII, smart pointers, STL algorithms).
- Always include testing suggestions and consider CPU/CUDA parity.
- Keep suggestions concise and focused on the `Tensor.ixx` design.
- In explanatory text (not code), you may use formatting symbols for clarity, but generated code comments must follow the comment policy above
- Keep commit messages and explanatory responses separate from code documentation