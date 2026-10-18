# Framing (WTP/1) -- SUPERSEDED by WTP/2

A frame is at most 8192 bytes. The header is 10 bytes. There is no extended header.

Encrypted bodies carry a 12-byte nonce and a 16-byte tag, and the payload is not padded.
