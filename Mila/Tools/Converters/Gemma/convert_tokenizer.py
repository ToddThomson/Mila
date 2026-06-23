#!/usr/bin/env python3
# ============================================================================
# File: convert_tokenizer.py
# Convert Gemma 4 tokenizer to Mila binary format
# ============================================================================

"""
Convert the Gemma 4 SentencePiece tokenizer from HuggingFace to Mila binary format.

No SentencePiece runtime dependency is needed: everything algorithm-specific is
read from the HF fast tokenizer HERE (vocab, per-piece scores, and BPE merges),
and the Mila runtime then operates on the extracted data. The converter inspects
the fast tokenizer's serialized model to decide between the two SentencePiece
algorithms and writes a `model_type` byte so the Mila loader (BpeVocabulary::
loadGemma) picks the matching decode path:

  - BPE     -> reuse Mila's existing merge-by-rank path (merges written below).
  - Unigram -> a Viterbi decode over piece scores (no merges; Mila runtime TBD).

Gemma's vocabulary pieces are raw UTF-8 with U+2581 ('lower one eighth block')
standing in for spaces, so the loader uses byte_level=false + a SentencePiece
pre-tokenization (where loadLlama32 uses byte_level=true + the Llama3 tiktoken
regex + the GPT-2 byte-encoder).

Binary format (Gemma extension of the shared Mila tokenizer binary):
    Header:
      - vocab_size        (uint32)
      - use_byte_fallback (uint8)   -- 1 (SentencePiece byte fallback: <0xNN> pieces)
      - model_type        (uint8)   -- 1 = BPE, 2 = Unigram, 0 = unknown
      - num_merges        (uint32)  -- 0 for Unigram
    Vocabulary (per token, ascending id):
      - token_length (uint32), token_bytes (utf-8), score (float32), token_id (uint32)
    Merges (present iff num_merges > 0, in rank order):
      - left_length (uint32), left, right_length (uint32), right
    Special tokens (each: has flag uint32, then id uint32 if set): BOS, EOS, PAD, UNK

Usage:
    python Gemma/convert_tokenizer.py --model google/gemma-4-12b-it --output <weights-dir>/gemma/gemma_tokenizer.bin
"""

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).parent.parent ) )

import argparse
import json
import struct
from transformers import AutoTokenizer


def _check_hf_error( model_name: str, e: Exception ):
    name = type( e ).__name__
    msg  = str( e )
    if 'GatedRepo' in name or ('403' in msg and 'gated' in msg.lower()):
        print( f"\nError: '{model_name}' is a gated model." )
        print( f"  1. Accept Google's license at https://huggingface.co/{model_name}" )
        print(  "  2. Authenticate: hf auth login" )
        sys.exit( 1 )
    if 'RepositoryNotFound' in name or '404' in msg:
        print( f"\nError: '{model_name}' not found on HuggingFace." )
        print(  "  Check the model name and your network connection." )
        sys.exit( 1 )
    raise e


SUPPORTED_MODELS = [
    'google/gemma-4-12b',
    'google/gemma-4-12b-it',
]

MODEL_TYPE_CODE = { 'BPE': 1, 'Unigram': 2 }


def write_string( f, s: str ):
    """Write a length-prefixed UTF-8 string."""
    encoded = s.encode( 'utf-8' )
    f.write( struct.pack( 'I', len( encoded ) ) )
    f.write( encoded )


def _model_section( tokenizer ) -> dict:
    """The fast tokenizer's serialized `model` block (vocab/merges/type)."""
    backend = getattr( tokenizer, 'backend_tokenizer', None )
    if backend is None:
        raise RuntimeError(
            "Gemma requires a fast (tokenizers-backed) AutoTokenizer to read the "
            "model type and merges; got a slow tokenizer." )
    return json.loads( backend.to_str() ).get( 'model', {} )


def _parse_merges( raw_merges ) -> list[tuple[str, str]]:
    """Normalize tokenizers' merge list to (left, right) pairs.

    Accepts both the legacy 'left right' string form and the newer [left, right]
    pair form. SentencePiece pieces use U+2581 for spaces, never a literal ASCII
    space, so splitting a string merge on its single space separator is safe.
    """
    merges = []
    for m in raw_merges:
        if isinstance( m, (list, tuple) ):
            left, right = m[ 0 ], m[ 1 ]
        else:
            left, right = m.split( ' ', 1 )
        merges.append( ( left, right ) )
    return merges


def convert_gemma_tokenizer( model_name: str, output_path: str ):
    print( f"Loading {model_name} tokenizer from HuggingFace..." )

    try:
        tokenizer = AutoTokenizer.from_pretrained( model_name )
    except Exception as e:
        _check_hf_error( model_name, e )

    model = _model_section( tokenizer )
    model_type = model.get( 'type', 'BPE' )
    type_code = MODEL_TYPE_CODE.get( model_type, 0 )
    print( f"Tokenizer model type: {model_type} (code {type_code})" )

    # id -> piece (authoritative, includes added/special tokens; contiguous for Gemma)
    vocab = tokenizer.get_vocab()                          # {piece: id}
    vocab_size = len( vocab )
    id_to_piece = [ '' ] * vocab_size
    for piece, idx in vocab.items():
        if 0 <= idx < vocab_size:
            id_to_piece[ idx ] = piece

    # Gemma's control / added tokens (<bos>, <eos>, <start_of_turn>, <end_of_turn>, ...)
    # are NOT reliably surfaced by get_vocab() (it returns the base pieces, and
    # convert_tokens_to_ids falls back to <unk> for them). get_added_vocab() is the
    # canonical source; patch them into id_to_piece at their ids so the Mila runtime
    # can match them atomically and decode them correctly.
    added_vocab = tokenizer.get_added_vocab()
    for tok, idx in added_vocab.items():
        if 0 <= idx < vocab_size:
            id_to_piece[ idx ] = tok

    # Scores: Unigram carries per-piece scores in model.vocab ([piece, score]);
    # BPE does not use scores at decode time -> 0.0.
    id_to_score = [ 0.0 ] * vocab_size
    if model_type == 'Unigram':
        for entry in model.get( 'vocab', [] ):
            piece, score = entry[ 0 ], float( entry[ 1 ] )
            idx = vocab.get( piece )
            if idx is not None and 0 <= idx < vocab_size:
                id_to_score[ idx ] = score

    merges = [] if model_type == 'Unigram' else _parse_merges( model.get( 'merges', [] ) )
    print( f"Vocabulary size: {vocab_size}" )
    print( f"Merges: {len( merges )}" )

    out = Path( output_path )
    out.parent.mkdir( parents=True, exist_ok=True )

    with open( out, 'wb' ) as f:
        f.write( struct.pack( 'I', vocab_size ) )
        f.write( struct.pack( 'B', 1 ) )                  # use_byte_fallback
        f.write( struct.pack( 'B', type_code ) )
        f.write( struct.pack( 'I', len( merges ) ) )

        print( "Writing vocabulary..." )
        for i in range( vocab_size ):
            write_string( f, id_to_piece[ i ] )
            f.write( struct.pack( 'f', id_to_score[ i ] ) )
            f.write( struct.pack( 'I', i ) )

        if merges:
            print( "Writing merges..." )
            for left, right in merges:
                write_string( f, left )
                write_string( f, right )

        print( "Writing special tokens..." )

        def write_special( name: str, token_id ):
            if token_id is not None:
                f.write( struct.pack( 'I', 1 ) )
                f.write( struct.pack( 'I', int( token_id ) ) )
                print( f"  {name}: ID {token_id}" )
            else:
                f.write( struct.pack( 'I', 0 ) )
                print( f"  {name}: None" )

        write_special( 'BOS', tokenizer.bos_token_id )
        write_special( 'EOS', tokenizer.eos_token_id )
        write_special( 'PAD', tokenizer.pad_token_id )
        write_special( 'UNK', tokenizer.unk_token_id )

    # Report the instruct turn-boundary ids the GemmaModel stop set needs
    # (resolves the GemmaModel eosToken/stopTokens REVIEW against the real tokenizer).
    for name in ( '<start_of_turn>', '<end_of_turn>' ):
        print( f"  (chat) {name}: ID {added_vocab.get( name, vocab.get( name ) )}" )

    print( f"\nConversion complete!\n  Output: {output_path}" )

    # ---- Sanity round-trip through the HF tokenizer -------------------------
    print( "\nTesting tokenizer (HF reference round-trip)..." )
    test_text = "The capital of France is Paris."
    ids = tokenizer.encode( test_text, add_special_tokens=False )
    print( f"  Input:   {test_text!r}" )
    print( f"  Tokens:  {ids}" )
    print( f"  Pieces:  {[ tokenizer.convert_ids_to_tokens( i ) for i in ids ]}" )
    print( f"  Decoded: {tokenizer.decode( ids )!r}" )
    print( "\nNote: Mila-side parity is validated once BpeVocabulary::loadGemma exists "
           "(encode round-trip vs these HF tokens)." )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Gemma 4 tokenizer to Mila format' )
    parser.add_argument( '--model', type=str, required=True, choices=SUPPORTED_MODELS,
        help='HuggingFace model name' )
    parser.add_argument( '--output', type=str, required=True,
        help='Output path for the Mila tokenizer file' )

    args = parser.parse_args()
    convert_gemma_tokenizer( args.model, args.output )
