# Llama 3.1 8B Instruct's instruction retention in HuggingFace, BF16 (Quantization.md Part III, decision 6).
#
# The prompt is built as LlamaInstructionRetentionCudaTests builds it: <|begin_of_text|>, a system turn holding the
# instruction, a user turn holding the first book tokens (Gutenberg's line wraps joined) and the question, then the
# assistant header; the book is cut so the prompt and a 256-token reply fill the length. The reply is greedy and is
# judged as the C++ harness judges it. The prompt goes in 1024-token chunks through the KV cache.
#
#   python hf_llama_instruction_retention.py --lengths 65536 16384
#
# A 64K cache with the BF16 weights needs both cards and some layers' weights in host memory (--gpu0-layers,
# --cpu-layers), which run on the first card and keep their cache there. Decode runs about 0.4 s a token.

import argparse
import os
import sys
import time
from pathlib import Path

CHECKPOINT = 'meta-llama/Llama-3.1-8B-Instruct'
BOOKS = Path( __file__ ).resolve().parents[ 4 ] / 'Data' / 'Datasets' / 'PG19' / 'raw' / 'test'
REPLY_TOKENS = 256
CHUNK = 1024
STOP = { 128009, 128001 }
QUESTION = 'Who is the main character of this book, and what do they want?'

# The 5060 Ti, then the 4070.
DEVICES = 'GPU-9a81c7d1-9db2-16b3-c256-2f991ec2a22c,GPU-11770557-a821-1b71-35db-34e342bae260'

GEMMA_SCRIPTS = Path( __file__ ).resolve().parents[ 1 ] / 'Gemma' / 'gemma_4_BF16'


def join_wraps( crlf: str ) -> str:
    """The book text as every G2 and L3 harness joins it (hf_long_context_loss.py)."""
    sys.path.insert( 0, str( GEMMA_SCRIPTS ) )
    from hf_long_context_loss import join_wraps as shared

    return shared( crlf )


def words_of( reply: str ) -> list[ str ]:
    """The reply's words, letters only, as the C++ harness splits them."""
    words, letters = [], []

    for c in reply + ' ':
        if c.isspace():
            if letters:
                words.append( ''.join( letters ) )
                letters = []
        elif c.isascii() and c.isalpha():
            letters.append( c )

    return words


INSTRUCTIONS = [
    ( 'three capital words', 'Whatever the user asks, reply in exactly three words, all in capital letters.',
      lambda r: len( words_of( r ) ) == 3 and all( w.isupper() for w in words_of( r ) ), False ),
    ( 'begins BANANA', 'Begin every reply with the word BANANA.',
      lambda r: bool( words_of( r ) ) and words_of( r )[ 0 ] == 'BANANA', False ),
    ( 'ends OVER', 'End every reply with the word OVER.',
      lambda r: bool( words_of( r ) ) and words_of( r )[ -1 ] == 'OVER', True ),
]


def grouped_sdpa( module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs ):
    """transformers' sdpa_attention_forward with one SDPA call per KV head over its query heads.

    The stock path expands K and V to every query head whenever a mask is passed, as every chunk after the first
    passes one: 1 GB a layer at 65K keys, which the first card cannot spare.
    """
    import torch

    groups = module.num_key_value_groups
    is_causal = is_causal if is_causal is not None else getattr( module, 'is_causal', True )
    is_causal = query.shape[ 2 ] > 1 and attention_mask is None and is_causal
    outputs = []

    for head in range( key.shape[ 1 ] ):
        q = query[ :, head * groups:( head + 1 ) * groups ]
        k = key[ :, head:head + 1 ].expand( -1, groups, -1, -1 ).contiguous()
        v = value[ :, head:head + 1 ].expand( -1, groups, -1, -1 ).contiguous()
        outputs.append( torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout, scale=scaling, is_causal=is_causal ) )

    return torch.cat( outputs, dim=1 ).transpose( 1, 2 ).contiguous(), None


def prompt( tokenizer, instruction: str, book: str, length: int ) -> list[ int ] | None:
    head = [ 128000 ] + tokenizer.encode(
        '<|start_header_id|>system<|end_header_id|>\n\n' + instruction
        + '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n', add_special_tokens=False )
    tail = tokenizer.encode( '\n\n' + QUESTION + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
                             add_special_tokens=False )
    book_tokens = length - len( head ) - len( tail ) - REPLY_TOKENS

    with open( BOOKS / f'{book}.txt', 'rb' ) as f:
        raw = f.read( book_tokens * 6 )

    text = tokenizer.encode( join_wraps( raw.decode( 'utf-8', errors='ignore' ) ), add_special_tokens=False )

    if len( text ) < book_tokens:
        return None

    return head + text[ :book_tokens ] + tail


def generate( model, tokens: list[ int ] ) -> list[ int ]:
    import torch
    from transformers import DynamicCache

    with torch.inference_mode():
        cache = DynamicCache()
        device = model.model.embed_tokens.weight.device
        ids = torch.tensor( [ tokens ], device=device )
        logits = None

        for start in range( 0, ids.shape[ 1 ], CHUNK ):
            logits = model( input_ids=ids[ :, start:start + CHUNK ], past_key_values=cache, use_cache=True,
                            logits_to_keep=1 ).logits

        generated = []

        for _ in range( REPLY_TOKENS ):
            token = int( logits[ 0, -1 ].argmax() )

            if token in STOP:
                break

            generated.append( token )
            logits = model( input_ids=torch.tensor( [ [ token ] ], device=device ), past_key_values=cache,
                            use_cache=True, logits_to_keep=1 ).logits

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--lengths', type=int, nargs='+', default=[ 65536, 16384 ] )
    parser.add_argument( '--books', nargs='+', default=[ '30312', '3608' ] )
    parser.add_argument( '--gpu0-layers', type=int, default=14 )
    parser.add_argument( '--cpu-layers', type=int, default=7 )
    args = parser.parse_args()

    os.environ.setdefault( 'CUDA_VISIBLE_DEVICES', DEVICES )

    import torch
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    AttentionInterface.register( 'sdpa', grouped_sdpa )

    device_map = { 'model.embed_tokens': 0, 'model.rotary_emb': 0, 'model.norm': 1, 'lm_head': 1 }

    for layer in range( 32 ):
        device_map[ f'model.layers.{layer}' ] = ( 0 if layer < args.gpu0_layers
                                                  else 'cpu' if layer < args.gpu0_layers + args.cpu_layers else 1 )

    tokenizer = AutoTokenizer.from_pretrained( CHECKPOINT )
    model = AutoModelForCausalLM.from_pretrained( CHECKPOINT, dtype=torch.bfloat16, device_map=device_map,
                                                  attn_implementation='sdpa' ).eval()

    for length in args.lengths:
        for book in args.books:
            for name, text, holds, judged_at_end in INSTRUCTIONS:
                tokens = prompt( tokenizer, text, book, length )

                if tokens is None:
                    print( f'{length:>7} {book:>6} {name:<20} book too short', flush=True )
                    continue

                start = time.time()
                generated = generate( model, tokens )
                reply = tokenizer.decode( generated )
                cut = judged_at_end and len( generated ) >= REPLY_TOKENS
                verdict = 'cut' if cut else 'holds' if holds( reply ) else 'lost'
                line = reply.replace( '\n', ' ' )
                line = line if len( line ) <= 60 else line[ :28 ] + '...' + line[ -29: ]

                print( f'{length:>7} {book:>6} {name:<20} {verdict:<5} prompt {len( tokens )} reply {len( generated )} '
                       f'{time.time() - start:.0f} s  {line}', flush=True )


if __name__ == '__main__':
    main()
