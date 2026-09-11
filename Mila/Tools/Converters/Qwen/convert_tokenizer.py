#!/usr/bin/env python3
# ============================================================================
# File: convert_tokenizer.py
# Convert the Qwen 3.8 tokenizer to Mila binary format
# ============================================================================

"""
Convert the Qwen 3.8 tokenizer from HuggingFace to Mila binary format.

Qwen 3.8 uses a GPT-2 style byte-level BPE: a ByteLevel pre-tokenizer over a
regex split, explicit merge rules by rank, and no byte fallback -- every byte
already has a vocabulary entry, so nothing can fall through. That makes this a
close relative of `Gpt2/convert_tokenizer.py` and a distant one of Gemma's
SentencePiece path, which shares only the file layout.

THIS CONVERTER READS `tokenizer.json` DIRECTLY AND DOES NOT USE `transformers`.
That is the one decision here worth stating, because the obvious alternative is
wrong. `transformers` 5.12.1's `Qwen2Tokenizer` rebuilds the backend from
`vocab.json` + `merges.txt` and OVERWRITES the checkpoint's pre-tokenizer with a
hardcoded Qwen2-era pattern (`tokenization_qwen2.py:33`) that lacks `\\p{M}`.
The two disagree on every script whose combining marks survive NFC -- Devanagari,
Thai, Arabic -- where the stale pattern splits a mark away from its base letter
and produces 2x the tokens. The checkpoint's pattern is the correct one and the
vocabulary proves it: it contains pieces such as `วัสดี` that span base letters
and marks, which the stale pattern can never emit, so the trainer used the
`\\p{M}` form. Both decode back to the same text, so the divergence is invisible
except as worse output. See the Phase 4 note in `Specifications/Qwen3.8.md`.

Three things differ from Llama 3.x, which is also byte-level BPE:

  1. THE SPLIT REGEX IS QWEN'S OWN. It differs from Llama 3's in two places:
     Qwen splits every digit (`\\p{N}`) where Llama chunks three at a time, and
     Qwen admits combining marks into the letter run (`[\\p{L}\\p{M}]+`). Only the
     second is observable. The digit rule is inert for this vocabulary -- it holds
     no multi-digit piece and no digit-digit merge rule, so the merge loop cannot
     join digits whichever pattern grouped them -- while the mark rule changes the
     ids on every script whose marks survive NFC. The checkpoint's own pattern is
     checked against the constant below on every run so a checkpoint revision
     cannot change it behind the Mila runtime's back.

  2. MERGES ARE WRITTEN. Mila's Llama loader uses the max-munch path and carries
     no merge table; Qwen's `ignore_merges` is false and its vocabulary is not
     max-munch equivalent, so the merge-by-rank path is the correct one and the
     ranks have to travel.

  3. THERE IS NO BOS. `add_bos_token` is false and `bos_token` is null; EOS is
     `<|im_end|>` (the conversational turn end), and `<|endoftext|>` serves as
     PAD and as the document terminator. Both are QwenModel's stop set.

A NORMALIZER IS DROPPED. The checkpoint applies NFC before pre-tokenization and
Mila has no normalization stage, so text that is not already NFC-normalized may
encode differently here than in HuggingFace. ASCII is unaffected (NFC is the
identity on it), and so is any text that arrives already composed, which covers
ordinary input from an editor or a terminal. Recorded rather than fixed: adding a
normalization stage is a runtime change, not a converter one.

Binary format (the shared Mila tokenizer binary -- same layout Gemma writes):
    Header:
      - vocab_size        (uint32)
      - use_byte_fallback (uint8)   -- 0 for Qwen: byte-level BPE, nothing falls through
      - model_type        (uint8)   -- 1 = BPE
      - num_merges        (uint32)
    Vocabulary (per token, ascending id):
      - token_length (uint32), token_bytes (utf-8), score (float32), token_id (uint32)
    Merges (in rank order):
      - left_length (uint32), left, right_length (uint32), right
    Special tokens (each: has flag uint32, then id uint32 if set): BOS, EOS, PAD, UNK

`score` is a SentencePiece Unigram concept with no meaning for BPE and is written
as 0.0, exactly as the Llama and Gemma BPE paths do.

Usage:
    python Qwen/convert_tokenizer.py --model Qwen/Qwen3.8-27B \\
        --output <weights-dir>/qwen/qwen38_tokenizer.bin
"""

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).parent.parent ) )

import argparse
import json
import struct


SUPPORTED_MODELS = [
    'Qwen/Qwen3.8-27B',
]

# The checkpoint's pre-tokenization regex, pinned. Mila's runtime carries this same
# pattern as QWEN3_PRETOKENIZATION_PATTERN (Src/Data/Tokenizers/Bpe/BpePreTokenizationMode.ixx);
# a checkpoint that changed it would tokenize differently in HuggingFace than in Mila
# with no symptom other than worse output, so the run fails instead.
EXPECTED_PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" )

# Control tokens the Mila runtime must match atomically rather than as subwords.
# Reported here against the real vocabulary so BpeVocabulary::loadQwen -- which
# registers them by name from the loaded vocabulary -- can be checked against a
# printed id rather than a hardcoded constant.
CONTROL_TOKENS = [
    '<|endoftext|>', '<|im_start|>', '<|im_end|>',
    '<think>', '</think>',
    '<tool_call>', '</tool_call>',
    '<tool_response>', '</tool_response>',
]

TOKENIZER_FILES = [ 'tokenizer.json', 'tokenizer_config.json' ]

MODEL_TYPE_BPE = 1


def _check_hf_error( model_name: str, e: Exception ):
    name = type( e ).__name__
    msg = str( e )

    if 'RepositoryNotFound' in name or '404' in msg:
        print( f"\nError: '{model_name}' not found on HuggingFace." )
        print( '  Check the model name and your network connection.' )
        print( '  Qwen 3.8 is Apache 2.0 and ungated -- no authentication is needed.' )
        sys.exit( 1 )

    raise e


def resolve_checkpoint( model_name: str ) -> Path:
    """Local directory holding the tokenizer files, downloading only those two."""
    local = Path( model_name )

    if local.is_dir():
        return local

    from huggingface_hub import snapshot_download

    print( f'Resolving {model_name} (downloads only what is missing from the hub cache)...' )

    try:
        return Path( snapshot_download( model_name, allow_patterns=TOKENIZER_FILES ) )
    except Exception as e:
        _check_hf_error( model_name, e )


def write_string( f, s: str ):
    """Write a length-prefixed UTF-8 string."""
    encoded = s.encode( 'utf-8' )
    f.write( struct.pack( 'I', len( encoded ) ) )
    f.write( encoded )


def _parse_merges( raw_merges ) -> list[ tuple[ str, str ] ]:
    """Normalize the tokenizers merge list to (left, right) pairs.

    Accepts both the legacy 'left right' string form and the newer [left, right]
    pair form. Byte-level pieces encode a space as U+0120, never as a literal
    ASCII space, so splitting a string merge on its single separator is safe.
    """
    merges = []

    for m in raw_merges:
        if isinstance( m, (list, tuple) ):
            left, right = m[ 0 ], m[ 1 ]
        else:
            left, right = m.split( ' ', 1 )

        merges.append( ( left, right ) )

    return merges


def _check_pretokenize_regex( serialized: dict ):
    """Fail if the checkpoint's split pattern is not the one the Mila runtime carries."""
    pre = serialized.get( 'pre_tokenizer' ) or {}
    patterns = []

    for entry in pre.get( 'pretokenizers', [ pre ] ):
        pattern = ( entry.get( 'pattern' ) or {} ).get( 'Regex' )

        if pattern is not None:
            patterns.append( pattern )

    if patterns != [ EXPECTED_PRETOKENIZE_REGEX ]:
        raise ValueError(
            'The checkpoint pre-tokenization regex is not the one Mila carries.\n'
            f'  checkpoint: {patterns}\n'
            f'  expected:   [{EXPECTED_PRETOKENIZE_REGEX!r}]\n'
            '  Update QWEN3_PRETOKENIZATION_PATTERN in '
            'Src/Data/Tokenizers/Bpe/BpePreTokenizationMode.ixx to match, then update '
            'EXPECTED_PRETOKENIZE_REGEX here.' )


def _special_token_text( entry ) -> str | None:
    """A tokenizer_config special-token field, which is either a string or an AddedToken dict."""
    if entry is None:
        return None

    if isinstance( entry, dict ):
        return entry.get( 'content' )

    return entry


def convert_qwen_tokenizer( model_name: str, output_path: str ):
    checkpoint = resolve_checkpoint( model_name )
    print( f'Reading tokenizer from {checkpoint}...' )

    serialized = json.loads( ( checkpoint / 'tokenizer.json' ).read_text( encoding='utf-8' ) )
    config = json.loads( ( checkpoint / 'tokenizer_config.json' ).read_text( encoding='utf-8' ) )

    _check_pretokenize_regex( serialized )

    model = serialized.get( 'model', {} )
    model_type = model.get( 'type' )

    if model_type != 'BPE':
        raise ValueError( f'Expected a BPE tokenizer model; the checkpoint declares {model_type!r}.' )

    # The learned pieces, plus the added control tokens -- which occupy the contiguous
    # tail above them rather than a separate range, so one id space covers both.
    vocab = dict( model.get( 'vocab', {} ) )

    for added in serialized.get( 'added_tokens', [] ):
        vocab[ added[ 'content' ] ] = added[ 'id' ]

    vocab_size = len( vocab )
    id_to_piece = [ None ] * vocab_size

    for piece, index in vocab.items():
        if 0 <= index < vocab_size:
            id_to_piece[ index ] = piece

    # A hole would be written as an empty piece and silently decode to nothing, so
    # the id space is required to be dense rather than assumed to be.
    holes = [ i for i, piece in enumerate( id_to_piece ) if piece is None ]

    if holes:
        raise ValueError(
            f'{len( holes )} vocabulary ids have no piece (first: {holes[ :5 ]}); '
            'the Mila format indexes by id and cannot represent a hole.' )

    merges = _parse_merges( model.get( 'merges', [] ) )

    print( f'  Tokenizer model type: {model_type}' )
    print( f'  Vocabulary size: {vocab_size} '
           f'({len( serialized.get( "added_tokens", [] ) )} added control tokens)' )
    print( f'  Merges: {len( merges )}' )

    out = Path( output_path )
    out.parent.mkdir( parents=True, exist_ok=True )

    with open( out, 'wb' ) as f:
        f.write( struct.pack( 'I', vocab_size ) )
        f.write( struct.pack( 'B', 0 ) )                   # use_byte_fallback
        f.write( struct.pack( 'B', MODEL_TYPE_BPE ) )
        f.write( struct.pack( 'I', len( merges ) ) )

        print( 'Writing vocabulary...' )

        for i in range( vocab_size ):
            write_string( f, id_to_piece[ i ] )
            f.write( struct.pack( 'f', 0.0 ) )             # score: unused by BPE
            f.write( struct.pack( 'I', i ) )

        print( 'Writing merges...' )

        for left, right in merges:
            write_string( f, left )
            write_string( f, right )

        print( 'Writing special tokens...' )

        def write_special( name: str, key: str ):
            text = _special_token_text( config.get( key ) )
            token_id = vocab.get( text ) if text is not None else None

            if text is not None and token_id is None:
                raise ValueError(
                    f'tokenizer_config names {key}={text!r}, which is not in the vocabulary.' )

            if token_id is not None:
                f.write( struct.pack( 'I', 1 ) )
                f.write( struct.pack( 'I', int( token_id ) ) )
                print( f'  {name}: {text!r} (ID {token_id})' )
            else:
                f.write( struct.pack( 'I', 0 ) )
                print( f'  {name}: None' )

        write_special( 'BOS', 'bos_token' )                # Qwen 3.8: none
        write_special( 'EOS', 'eos_token' )                # <|im_end|>
        write_special( 'PAD', 'pad_token' )                # <|endoftext|>
        write_special( 'UNK', 'unk_token' )                # none: byte-level BPE

    print( f'\nConversion complete!\n  Output: {output_path}' )

    _report_control_tokens( vocab )
    _report_reference_tokenization( checkpoint )


def _report_control_tokens( vocab: dict ):
    """Print the ids BpeVocabulary::loadQwen registers, and flag any that are absent.

    An absent control token encodes as subword fragments, so anything keyed on it --
    the turn boundary, the thinking channel, tool calls -- silently stops working.
    """
    print( '\nControl tokens:' )

    for name in CONTROL_TOKENS:
        index = vocab.get( name )

        if index is None:
            print( f'  {name}: ABSENT -- will encode as subwords' )
        else:
            print( f'  {name}: ID {index}' )


def _report_reference_tokenization( checkpoint: Path ):
    """Ground truth for the Mila-side parity test, from the checkpoint's own tokenizer.

    The digit cases are not decoration: they are the one place Llama's regex and
    Qwen's disagree, so a Mila loader that picked the wrong pattern fails HERE and
    nowhere else in ordinary English text.
    """
    from tokenizers import Tokenizer

    reference = Tokenizer.from_file( str( checkpoint / 'tokenizer.json' ) )

    print( '\nCheckpoint reference (no special tokens added):' )

    samples = [
        'The capital of France is Paris.',
        'Hello, world! This is a test.',
        '42 items cost $3.50',
    ]

    for text in samples:
        ids = reference.encode( text, add_special_tokens=False ).ids
        pieces = [ reference.id_to_token( i ) for i in ids ]
        print( f'  {text!r}' )
        print( f'    ids:    {ids}' )
        print( f'    pieces: {pieces}' )
        print( f'    decode: {reference.decode( ids )!r}' )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert the Qwen 3.8 tokenizer to Mila format' )
    parser.add_argument( '--model', type=str, required=True,
        help=f'HuggingFace model name (supported: {", ".join( SUPPORTED_MODELS )}) '
             'or a local checkpoint directory' )
    parser.add_argument( '--output', type=str, required=True,
        help='Output path for the Mila tokenizer file' )

    args = parser.parse_args()
    convert_qwen_tokenizer( args.model, args.output )
