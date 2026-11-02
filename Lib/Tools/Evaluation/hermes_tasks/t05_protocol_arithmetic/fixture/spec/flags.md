# Header flags (WTP/2)

| Bit | Name | Meaning |
|---|---|---|
| 0 | E | The body is encrypted (`security.md`) |
| 1 | X | An extended header follows the base header (`framing.md`) |
| 2 | F | Final frame of a message |
| 3-7 | | Reserved, must be zero |

The extended header carries the message id (4 bytes) and the total message length (4 bytes).
