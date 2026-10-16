#!/usr/bin/env python3
# ============================================================================
# File: hf_qwen_conversation_loss.py
# BF16 reference loss for Qwen 3.8 27B on held-out conversations, layer-streamed
# ============================================================================

"""
Score held-out conversations under the BF16 checkpoint, the reference for Mila's
`QwenPackedArtifactTests.DISABLED_QualityOnConversations`.

The input is what `Tools/Quantization/calibration_text.py --held-out-tool` writes: conversations
rendered through the chat template, separated by 0x1E. Each is scored alone from its own start, every
position after the first, exactly as the Mila test scores it. The checkpoint does not fit this
machine, so `hf_qwen_layer_stream.StreamedReference` holds one decoder layer at a time; the walk is
layer-major, every conversation through layer i before layer i+1 is read, so each layer is read once.

`--top-k-output` also writes, for every scored position, the reference's TOP_K most likely next tokens
and their log-probabilities, which `QwenPackedArtifactTests.DISABLED_DivergenceOnConversations` measures a
build's KL divergence and top-1 agreement against. The conversation's own token ids travel with them, so
the build is fed exactly what the reference read. Layout, little-endian: the magic `MQTK`, int32 k, int32
conversation count; then per conversation int32 T, int32 tokens[T], int32 ids[(T - 1) * k], float32
log_probabilities[(T - 1) * k], position j being the distribution after tokens[0..j].

`--llama-kl-base` writes instead the base file `llama-perplexity --kl-divergence` reads, so a GGUF build is
measured against this same reference, and `DISABLED_DivergenceAgainstLlamaBase` reads it for Mila's builds. The
conversations are laid end to end and cut into `--chunks` chunks of `--n-ctx` tokens, each opening with
`<|endoftext|>`, the vocabulary's BOS -- so a reader that puts BOS first changes nothing -- and each chunk's second
half is stored as llama.cpp's own `log_softmax` stores it: two floats, scale and minimum log-probability, then
one uint16 per vocabulary entry over the 16 nats below the largest logit.

Usage:
    python Qwen/qwen38_BF16/hf_qwen_layer_stream.py --self-test      # prove the driver first
    python Qwen/qwen38_BF16/hf_qwen_conversation_loss.py --model Qwen/Qwen3.8-27B \\
        --conversations ../Quantization/corpus/held_out_tool_qwen38.txt \\
        --top-k-output ../Quantization/corpus/held_out_tool_qwen38.topk
"""

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).parent ) )

import argparse
import json
import math
import struct

import numpy as np
import torch

from hf_qwen_layer_stream import StreamedReference, ShardedWeights, TEXT_PREFIX, resolve_checkpoint, _release

SEPARATOR = '\x1e'

# Rows of the head evaluated at once: 512 x 248K logits is 0.5 GiB in FP32.
HEAD_ROWS = 512

# Reference predictions kept per position. The probability left outside them is carried as one remainder,
# so the KL a build is measured by is bounded below by the true one and loses only how that remainder splits.
TOP_K = 32

END_OF_TEXT = 248044


def llama_rows( logits ):
    """
    Rows of logits as llama.cpp's uint16 log_softmax writes them: per row two float32 (scale, minimum
    log-probability) as four uint16, then one uint16 per vocabulary entry, padded to an even count.
    """
    maximum = logits.max( dim=-1, keepdim=True ).values
    minimum = torch.maximum( logits.min( dim=-1, keepdim=True ).values, maximum - 16 )
    log_sum_exp = torch.log( torch.exp( logits - maximum ).double().sum( dim=-1, keepdim=True ) ).float()
    minimum_log_probability = minimum - maximum - log_sum_exp
    scale = ( maximum - minimum ) / 65535.0
    inverse_scale = torch.where( scale > 0, 1.0 / scale, torch.zeros_like( scale ) )
    quantized = torch.where( logits > minimum, torch.round( inverse_scale * ( logits - minimum ) ),
                             torch.zeros_like( logits ) ).clamp( 0, 65535 ).to( torch.int32 )

    header = torch.cat( [ scale, minimum_log_probability ], dim=-1 ).cpu().numpy().astype( '<f4' ).view( '<u2' )
    values = quantized.cpu().numpy().astype( '<u2' )

    if values.shape[ 1 ] % 2:
        values = np.concatenate( [ values, np.zeros( ( values.shape[ 0 ], 1 ), dtype='<u2' ) ], axis=1 )

    return np.ascontiguousarray( np.concatenate( [ header, values ], axis=1 ) )


@torch.no_grad()
def conversation_loss( model_name, conversations_path, device, max_length, limit, top_k_output, dtype,
                       llama_kl_base=None, n_ctx=2048, chunks=8 ):
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    checkpoint = resolve_checkpoint( model_name )
    config_json = json.loads( ( checkpoint / 'config.json' ).read_text() )
    text_config = Qwen3_5TextConfig( **config_json[ 'text_config' ] )
    text_config._attn_implementation = 'eager'

    tokenizer = AutoTokenizer.from_pretrained( model_name )
    texts = Path( conversations_path ).read_bytes().decode( 'utf-8' ).split( SEPARATOR )
    sequences = [ tokenizer( text, add_special_tokens=False ).input_ids for text in texts ]
    too_long = sum( len( ids ) > max_length for ids in sequences )
    sequences = [ ids for ids in sequences if 2 <= len( ids ) <= max_length ]

    if limit > 0:
        sequences = sequences[ :limit ]

    print( f'{Path( conversations_path ).name}: {len( sequences )} conversations '
           f'({too_long} longer than {max_length} left out), {sum( map( len, sequences ) )} tokens' )

    if llama_kl_base:
        stream = [ token for ids in sequences for token in ids ]
        width = n_ctx - 1

        if len( stream ) < chunks * width:
            raise SystemExit( f'{len( stream )} tokens hold fewer than {chunks} chunks of {n_ctx}' )

        sequences = [ [ END_OF_TEXT ] + stream[ c * width:( c + 1 ) * width ] for c in range( chunks ) ]
        print( f'llama.cpp base: {chunks} chunks of {n_ctx}, positions {n_ctx // 2} to {n_ctx - 2} of each' )

    weights = ShardedWeights( checkpoint )
    driver = StreamedReference( text_config, weights, device, dtype )

    hidden = [ weights.get_rows( f'{TEXT_PREFIX}embed_tokens.weight', ids )
               .to( device=device, dtype=dtype ).unsqueeze( 0 ) for ids in sequences ]

    # Per conversation, as StreamedReference.run builds them for one prompt.
    contexts = []

    for state in hidden:
        position_ids = torch.arange( state.shape[ 1 ], device=device ).view( 1, 1, -1 ).expand( 4, 1, -1 )
        causal_mask, linear_mask = driver._masks( state, position_ids[ 0 ] )
        contexts.append( ( position_ids[ 0 ], driver.rotary( state, position_ids[ 1: ] ), causal_mask, linear_mask ) )

    for i in range( text_config.num_hidden_layers ):
        layer_type = text_config.layer_types[ i ]
        layer = driver._build_layer( i )

        for c, ( text_positions, position_embeddings, causal_mask, linear_mask ) in enumerate( contexts ):
            hidden[ c ] = layer(
                hidden[ c ],
                position_embeddings=position_embeddings,
                attention_mask=linear_mask if layer_type == 'linear_attention' else causal_mask,
                position_ids=text_positions,
                past_key_values=None )

        del layer
        _release( device )
        print( f'  layer {i + 1}/{text_config.num_hidden_layers} {layer_type}', flush=True )

    with torch.device( 'meta' ):
        norm = Qwen3_5RMSNorm( text_config.hidden_size, eps=text_config.rms_norm_eps )

    norm.load_state_dict( { 'weight': weights.get( f'{TEXT_PREFIX}norm.weight' )
                                      .to( device=device, dtype=dtype ) }, assign=True )
    lm_head = weights.get( 'lm_head.weight' ).to( device=device, dtype=dtype )

    total_log_probability, total_positions = 0.0, 0
    top_k = []
    base = None

    if llama_kl_base:
        base = open( llama_kl_base, 'wb' )
        base.write( b'_logits_' + struct.pack( '<iii', n_ctx, lm_head.shape[ 0 ], len( sequences ) ) )
        base.write( np.asarray( [ token for ids in sequences for token in ids ], dtype='<i4' ).tobytes() )

    for c, ( ids, state ) in enumerate( zip( sequences, hidden ) ):
        normed = norm( state )[ 0 ]

        if base is not None:
            for start in range( n_ctx // 2, n_ctx - 1, HEAD_ROWS // 2 ):
                end = min( start + HEAD_ROWS // 2, n_ctx - 1 )
                base.write( llama_rows( ( normed[ start:end ] @ lm_head.T ).float() ).tobytes() )
        targets = torch.tensor( ids[ 1: ], device=device )
        log_probability = 0.0
        top_ids, top_log_probabilities = [], []

        for start in range( 0, len( ids ) - 1, HEAD_ROWS ):
            end = min( start + HEAD_ROWS, len( ids ) - 1 )
            log_probabilities = torch.log_softmax( ( normed[ start:end ] @ lm_head.T ).float(), dim=-1 )
            log_probability += log_probabilities.gather( 1, targets[ start:end ].unsqueeze( 1 ) ).sum().item()

            if top_k_output:
                best = torch.topk( log_probabilities, TOP_K, dim=-1 )
                top_ids.append( best.indices.to( torch.int32 ).cpu().numpy() )
                top_log_probabilities.append( best.values.cpu().numpy() )

        if top_k_output:
            top_k.append( ( ids, np.concatenate( top_ids ), np.concatenate( top_log_probabilities ) ) )

        total_log_probability += log_probability
        total_positions += len( ids ) - 1
        print( f'  conversation {c:3d}: {len( ids ):5d} tokens, perplexity '
               f'{math.exp( -log_probability / ( len( ids ) - 1 ) ):9.3f}', flush=True )

    print( f'\nBF16 reference: perplexity {math.exp( -total_log_probability / total_positions ):.3f} '
           f'over {total_positions} positions' )

    if base is not None:
        base.close()
        print( f'llama.cpp base -> {llama_kl_base}' )

    if top_k_output:
        with open( top_k_output, 'wb' ) as output:
            output.write( b'MQTK' + struct.pack( '<ii', TOP_K, len( top_k ) ) )

            for tokens, ids, log_probabilities in top_k:
                output.write( struct.pack( '<i', len( tokens ) ) )
                output.write( np.asarray( tokens, dtype='<i4' ).tobytes() )
                output.write( np.ascontiguousarray( ids, dtype='<i4' ).tobytes() )
                output.write( np.ascontiguousarray( log_probabilities, dtype='<f4' ).tobytes() )

        print( f'top {TOP_K} per position -> {top_k_output}' )


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--model', type=str, default='Qwen/Qwen3.8-27B',
        help='HuggingFace model name or a local checkpoint directory' )
    parser.add_argument( '--conversations', type=str, required=True,
        help='held-out conversations, separated by 0x1E' )
    parser.add_argument( '--device', type=str, default='cuda' )
    parser.add_argument( '--max-length', type=int, default=8192,
        help='conversations longer than this are left out, as the Mila test leaves them out (default: 8192)' )
    parser.add_argument( '--limit', type=int, default=0, help='score only the first N conversations' )
    parser.add_argument( '--top-k-output', type=str,
        help='also write the reference top-k at every position, for the Mila divergence test' )

    parser.add_argument( '--dtype', type=str, default='bfloat16', choices=[ 'bfloat16', 'float32' ],
        help='compute precision; float32 checks that bfloat16 arithmetic is not what the reference measures' )

    parser.add_argument( '--llama-kl-base', type=str,
        help='write the base file llama-perplexity --kl-divergence reads, instead of scoring conversations' )
    parser.add_argument( '--n-ctx', type=int, default=2048, help='chunk length for --llama-kl-base (default: 2048)' )
    parser.add_argument( '--chunks', type=int, default=8, help='chunks for --llama-kl-base (default: 8)' )

    args = parser.parse_args()
    conversation_loss( args.model, args.conversations, args.device, args.max_length, args.limit, args.top_k_output,
                       torch.float32 if args.dtype == 'float32' else torch.bfloat16,
                       args.llama_kl_base, args.n_ctx, args.chunks )
