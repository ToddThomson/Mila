#!/usr/bin/env python3
# ============================================================================
# File: convert_llama_tokenizer.py
# Convert Llama tokenizer to Mila binary format
# ============================================================================

"""
Convert Llama SentencePiece tokenizer from HuggingFace to Mila binary format.

Usage:
    python convert_llama_tokenizer.py --model meta-llama/Llama-2-7b-hf --output ../Tokenizers/llama2_tokenizer.bin
    python convert_llama_tokenizer.py --model meta-llama/Llama-3.2-1B --output ../Tokenizers/llama3_tokenizer.bin
"""

import argparse
import struct
from transformers import AutoTokenizer


def write_string(f, s: str):
    """Write a length-prefixed string."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('I', len(encoded)))
    f.write(encoded)


def convert_llama_tokenizer(model_name: str, output_path: str):
    """
    Convert Llama tokenizer to Mila binary format.
    
    Binary format:
    Header:
      - vocab_size (uint32)
      - use_byte_fallback (uint8)
    
    Vocabulary section:
      - For each piece:
        - piece_length (uint32)
        - piece_bytes (utf-8 encoded)
        - score (float32)
        - token_id (uint32)
    
    Special tokens:
      - has_bos (uint32, 0 or 1)
      - bos_token_id (uint32, if has_bos)
      - has_eos (uint32, 0 or 1)
      - eos_token_id (uint32, if has_eos)
      - has_pad (uint32, 0 or 1)
      - pad_token_id (uint32, if has_pad)
      - has_unk (uint32, 0 or 1)
      - unk_token_id (uint32, if has_unk)
    """
    
    print(f"Loading {model_name} tokenizer from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # Get the vocabulary
    vocab = tokenizer.get_vocab()
    
    # For SentencePiece models, we need to get scores
    # These are stored in the tokenizer's vocab
    if hasattr(tokenizer, 'sp_model'):
        # LlamaTokenizer uses sentencepiece
        print("Using SentencePiece model")
        sp_model = tokenizer.sp_model
        
        # Extract pieces with scores
        pieces = []
        for token_id in range(vocab_size):
            piece = sp_model.id_to_piece(token_id)
            score = sp_model.get_score(token_id)
            pieces.append((piece, score, token_id))
    else:
        # Fallback: use uniform scores
        print("Warning: Not a SentencePiece model, using uniform scores")
        pieces = [(token, 0.0, token_id) for token, token_id in vocab.items()]
    
    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('I', vocab_size))
        f.write(struct.pack('B', 1))  # use_byte_fallback = true
        
        # Write vocabulary pieces
        print("Writing vocabulary...")
        for piece, score, token_id in pieces:
            write_string(f, piece)
            f.write(struct.pack('f', score))
            f.write(struct.pack('I', token_id))
        
        # Write special tokens
        print("Writing special tokens...")
        
        # BOS token
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is not None:
            f.write(struct.pack('I', 1))  # has_bos = true
            f.write(struct.pack('I', bos_token_id))
            print(f"  BOS token: '{tokenizer.bos_token}' (ID: {bos_token_id})")
        else:
            f.write(struct.pack('I', 0))  # has_bos = false
        
        # EOS token
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None:
            f.write(struct.pack('I', 1))  # has_eos = true
            f.write(struct.pack('I', eos_token_id))
            print(f"  EOS token: '{tokenizer.eos_token}' (ID: {eos_token_id})")
        else:
            f.write(struct.pack('I', 0))  # has_eos = false
        
        # PAD token
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            f.write(struct.pack('I', 1))  # has_pad = true
            f.write(struct.pack('I', pad_token_id))
            print(f"  PAD token: '{tokenizer.pad_token}' (ID: {pad_token_id})")
        else:
            f.write(struct.pack('I', 0))  # has_pad = false
            print(f"  PAD token: None")
        
        # UNK token
        unk_token_id = tokenizer.unk_token_id
        if unk_token_id is not None:
            f.write(struct.pack('I', 1))  # has_unk = true
            f.write(struct.pack('I', unk_token_id))
            print(f"  UNK token: '{tokenizer.unk_token}' (ID: {unk_token_id})")
        else:
            f.write(struct.pack('I', 0))  # has_unk = false
    
    print(f"\nConversion complete!")
    print(f"  Output: {output_path}")
    
    # Test the conversion
    print("\nTesting tokenizer...")
    test_text = "Hello, world! This is a test."
    token_ids = tokenizer.encode(test_text, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids)
    print(f"  Input: '{test_text}'")
    print(f"  Tokens: {token_ids}")
    print(f"  Decoded: '{decoded}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Llama tokenizer to Mila binary format'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='HuggingFace model name (e.g., meta-llama/Llama-2-7b-hf)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for Mila tokenizer file'
    )
    
    args = parser.parse_args()
    convert_llama_tokenizer(args.model, args.output)