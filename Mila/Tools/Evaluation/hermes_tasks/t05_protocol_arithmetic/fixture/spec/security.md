# Encryption (WTP/2)

When the E flag is set, the frame body is encrypted with an AEAD cipher that works in 16-byte blocks.

The body is, in order:

1. an 8-byte nonce;
2. the ciphertext;
3. a 16-byte authentication tag.

Before encryption the application payload is padded PKCS#7-style to a whole number of 16-byte blocks: 1 to 16
bytes of padding are always added, so a payload that is already a multiple of 16 bytes gains a full block. The
ciphertext is exactly as long as the padded payload. Padding is not application payload.
