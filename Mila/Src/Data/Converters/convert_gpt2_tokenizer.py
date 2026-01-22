#!/usr/bin/env python3
# ============================================================================
# File: convert_gpt2_tokenizer.py
# Convert GPT-2 tokenizer to Mila binary format
# ============================================================================

"""
Convert GPT-2 tokenizer from HuggingFace to Mila binary format.

Usage:
    python convert_gpt2_tokenizer.py --model gpt2 --output ../Tokenizers/gpt2_tokenizer.bin
"""

import argparse
import struct
from transformers import GPT2Tokenizer


def write_string(f, s: str):
    """Write a length-prefixed string."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('I', len(encoded)))
    f.write(encoded)


def convert_gpt2_tokenizer(model_name: str, output_path: str):
    """
    Convert GPT-2 tokenizer to Mila binary format.
    
    Binary format:
    Header:
      - vocab_size (uint32)
      - num_merges (uint32)
    
    Vocabulary section:
      - For each token:
        - token_length (uint32)
        - token_bytes (utf-8 encoded)
        - token_id (uint32)
    
    BPE Merges section:
      - For each merge pair:
        - token1_length (uint32)
        - token1_bytes (utf-8)
        - token2_length (uint32)
        - token2_bytes (utf-8)
    
    Special tokens:
      - has_eos (uint32, 0 or 1)
      - eos_token_id (uint32, if has_eos)
      - has_bos (uint32, 0 or 1)
      - bos_token_id (uint32, if has_bos)
      - has_pad (uint32, 0 or 1)
      - pad_token_id (uint32, if has_pad)
    """
    
    print(f"Loading {model_name} tokenizer from HuggingFace...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # Get the encoder (token -> id mapping)
    encoder = tokenizer.get_vocab()
    
    # Get BPE merges
    bpe_merges = tokenizer.bpe_ranks
    num_merges = len(bpe_merges)
    print(f"Number of BPE merges: {num_merges}")
    
    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('I', vocab_size))
        f.write(struct.pack('I', num_merges))
        
        # Write vocabulary
        print("Writing vocabulary...")
        for token, token_id in encoder.items():
            write_string(f, token)
            f.write(struct.pack('I', token_id))
        
        # Write BPE merges (sorted by rank)
        print("Writing BPE merges...")
        sorted_merges = sorted(bpe_merges.items(), key=lambda x: x[1])
        for (token1, token2), rank in sorted_merges:
            write_string(f, token1)
            write_string(f, token2)
        
        # Write special tokens
        print("Writing special tokens...")
        
        # EOS token (GPT-2 uses <|endoftext|> for both BOS and EOS)
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None:
            f.write(struct.pack('I', 1))  # has_eos = true
            f.write(struct.pack('I', eos_token_id))
            print(f"  EOS token: '{tokenizer.eos_token}' (ID: {eos_token_id})")
        else:
            f.write(struct.pack('I', 0))  # has_eos = false
        
        # BOS token (same as EOS for GPT-2)
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is not None:
            f.write(struct.pack('I', 1))  # has_bos = true
            f.write(struct.pack('I', bos_token_id))
            print(f"  BOS token: '{tokenizer.bos_token}' (ID: {bos_token_id})")
        else:
            f.write(struct.pack('I', 0))  # has_bos = false
        
        # PAD token
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            f.write(struct.pack('I', 1))  # has_pad = true
            f.write(struct.pack('I', pad_token_id))
            print(f"  PAD token: '{tokenizer.pad_token}' (ID: {pad_token_id})")
        else:
            f.write(struct.pack('I', 0))  # has_pad = false
            print(f"  PAD token: None (GPT-2 doesn't use padding)")
    
    print(f"\nConversion complete!")
    print(f"  Output: {output_path}")
    
    # Test the conversion
    print("\nTesting tokenizer...")
    test_text = "Hello, world! This is a test."
    token_ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(token_ids)
    print(f"  Input: '{test_text}'")
    print(f"  Tokens: {token_ids}")
    print(f"  Decoded: '{decoded}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert GPT-2 tokenizer to Mila binary format'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
        help='GPT-2 model variant'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for Mila tokenizer file'
    )
    
    args = parser.parse_args()
    convert_gpt2_tokenizer(args.model, args.output)