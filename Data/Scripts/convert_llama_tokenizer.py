#!/usr/bin/env python3
# ============================================================================
# File: convert_llama_tokenizer.py
# Convert Llama 3.2 tokenizer to Mila binary format
# ============================================================================

"""
Convert Llama 3.2 TikToken/BPE tokenizer from HuggingFace to Mila binary format.

Supported models (Llama 3.2 only):
    meta-llama/Llama-3.2-1B
    meta-llama/Llama-3.2-3B
    meta-llama/Llama-3.2-1B-Instruct
    meta-llama/Llama-3.2-3B-Instruct

Note on prior Llama versions:
    Llama 1/2 use a SentencePiece tokenizer (32K vocab) which requires the
    sp_model path and per-token score extraction. This is intentionally out of
    scope for Mila alpha.2 — add a SentencePiece branch here if Llama 1/2
    support is needed in a future release.

Usage:
    python convert_llama_tokenizer.py --model meta-llama/Llama-3.2-1B --output ../Tokenizers/llama32_tokenizer.bin
"""

import argparse
import struct
from transformers import AutoTokenizer

# Supported Llama 3.2 text model variants (TikToken BPE, 128K vocab)
SUPPORTED_MODELS = [
    'meta-llama/Llama-3.2-1B',
    'meta-llama/Llama-3.2-3B',
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
]


def write_string(f, s: str):
    """Write a length-prefixed string."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('I', len(encoded)))
    f.write(encoded)


def convert_llama_tokenizer(model_name: str, output_path: str):
    """
    Convert Llama 3.2 tokenizer to Mila binary format.

    Llama 3.2 uses a TikToken-based BPE tokenizer (128,256 vocab), not
    SentencePiece. Scores are a SentencePiece concept and are not part of BPE;
    they are written as 0.0 and are unused by Mila's BPE tokenizer.

    Binary format:
    Header:
      - vocab_size (uint32)
      - use_byte_fallback (uint8)

    Vocabulary section:
      - For each token:
        - token_length (uint32)
        - token_bytes (utf-8 encoded)
        - score (float32)  -- always 0.0 for BPE, reserved for SP compatibility
        - token_id (uint32)

    Special tokens:
      - has_bos (uint32, 0 or 1)
      - bos_token_id (uint32, if has_bos)   -- Llama 3.2: <|begin_of_text|> = 128000
      - has_eos (uint32, 0 or 1)
      - eos_token_id (uint32, if has_eos)   -- Llama 3.2: <|end_of_text|>   = 128001
      - has_pad (uint32, 0 or 1)
      - pad_token_id (uint32, if has_pad)
      - has_unk (uint32, 0 or 1)
      - unk_token_id (uint32, if has_unk)   -- Llama 3.2: no UNK token
    """

    print(f"Loading {model_name} tokenizer from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # Llama 3.2 uses TikToken BPE — scores are not applicable.
    # score field written as 0.0, reserved for future SentencePiece compatibility.
    print("Using TikToken/BPE tokenizer — scores not applicable, writing 0.0")
    vocab = tokenizer.get_vocab()
    pieces = [(token, 0.0, token_id) for token, token_id in vocab.items()]

    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('I', vocab_size))
        f.write(struct.pack('B', 1))  # use_byte_fallback = true

        # Write vocabulary
        print("Writing vocabulary...")
        for token, score, token_id in pieces:
            write_string(f, token)
            f.write(struct.pack('f', score))
            f.write(struct.pack('I', token_id))

        # Write special tokens
        print("Writing special tokens...")

        # BOS token: <|begin_of_text|> = 128000
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is not None:
            f.write(struct.pack('I', 1))
            f.write(struct.pack('I', bos_token_id))
            print(f"  BOS token: '{tokenizer.bos_token}' (ID: {bos_token_id})")
        else:
            f.write(struct.pack('I', 0))

        # EOS token: <|end_of_text|> = 128001
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None:
            f.write(struct.pack('I', 1))
            f.write(struct.pack('I', eos_token_id))
            print(f"  EOS token: '{tokenizer.eos_token}' (ID: {eos_token_id})")
        else:
            f.write(struct.pack('I', 0))

        # PAD token: Llama 3.2 has no dedicated pad token
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            f.write(struct.pack('I', 1))
            f.write(struct.pack('I', pad_token_id))
            print(f"  PAD token: '{tokenizer.pad_token}' (ID: {pad_token_id})")
        else:
            f.write(struct.pack('I', 0))
            print(f"  PAD token: None")

        # UNK token: Llama 3.2 has no UNK token (TikToken BPE uses byte fallback)
        unk_token_id = tokenizer.unk_token_id
        if unk_token_id is not None:
            f.write(struct.pack('I', 1))
            f.write(struct.pack('I', unk_token_id))
            print(f"  UNK token: '{tokenizer.unk_token}' (ID: {unk_token_id})")
        else:
            f.write(struct.pack('I', 0))
            print(f"  UNK token: None (expected — TikToken uses byte fallback)")

    print(f"\nConversion complete!")
    print(f"  Output: {output_path}")

    # Test the conversion
    print("\nTesting tokenizer...")
    test_text = "Hello, world! This is a test."
    token_ids = tokenizer.encode(test_text, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids)
    print(f"  Input:   '{test_text}'")
    print(f"  Tokens:  {token_ids}")
    print(f"  Decoded: '{decoded}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Llama 3.2 tokenizer to Mila binary format'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for Mila tokenizer file'
    )

    args = parser.parse_args()
    convert_llama_tokenizer(args.model, args.output)