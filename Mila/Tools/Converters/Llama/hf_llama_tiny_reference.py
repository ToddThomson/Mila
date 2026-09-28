# HuggingFace reference for LlamaTransformer (ModelFamilyParity.md 8.4, L1).
#
# Builds a tiny LlamaForCausalLM and draws every weight at random -- every norm away from 1, so a
# misplaced norm cannot hide behind an identity. The model is saved as a HuggingFace checkpoint and
# converted with Llama/convert_weights.py at FP32 and BF16, so one capture holds the converter's names
# and the network's wiring together. No network, no GPU.
#
# It records HuggingFace's logits at the prefill's last position and at each decode step, and the
# log-probability it assigns to each next token of the whole sequence -- the reference for
# LlamaTransformer's sequenceLogLikelihood.
#
# Head dimension 128, a group of 2: the geometry Llama's flash prefill serves, at test scale.
#
# --rope-scaling llama3 gives the model Llama 3.1's frequency scaling at an original context of 32. Scaling
# changes the frequencies at every position, so it shows inside the 32-token capture; at head dimension 128 the
# original 32 puts frequencies in all three of the rule's bands -- kept, interpolated and divided
# (ModelFamilyParity.md 8.4, L2).
#
#   python hf_llama_tiny_reference.py --output-dir ../../../../Data/models/llama/llama_tiny
#   python hf_llama_tiny_reference.py --rope-scaling llama3 --output-dir ../../../../Data/models/llama/llama_tiny_scaled

import argparse
import sys
from pathlib import Path

sys.path.insert( 0, str( Path( __file__ ).resolve().parent ) )

import torch
import transformers
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from convert_weights import convert_llama


PROMPT_TOKENS = 24
DECODE_TOKENS = 8


ROPE_PARAMETERS = {
    'none': { 'rope_type': 'default', 'rope_theta': 10000.0 },
    'llama3': {
        'rope_type': 'llama3',
        'rope_theta': 10000.0,
        'factor': 4.0,
        'low_freq_factor': 1.0,
        'high_freq_factor': 4.0,
        'original_max_position_embeddings': 32,
    },
}


def tiny_config( rope_scaling: str ) -> LlamaConfig:
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_parameters=ROPE_PARAMETERS[ rope_scaling ],
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False )

    config._attn_implementation = 'eager'

    return config


def randomize( model: torch.nn.Module, seed: int ):
    """Every tensor drawn; the norms, which a model initializes to one, are drawn away from it."""
    generator = torch.Generator().manual_seed( seed )

    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if 'norm' in name and name.endswith( 'weight' ):
                tensor.copy_( 0.5 + torch.rand( tensor.shape, generator=generator ) )
            else:
                tensor.copy_( torch.randn( tensor.shape, generator=generator ) * 0.05 )


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--seed', type=int, default=20260927 )
    parser.add_argument( '--rope-scaling', choices=sorted( ROPE_PARAMETERS ), default='none' )
    parser.add_argument( '--output-dir', type=Path, required=True )
    arguments = parser.parse_args()

    output = arguments.output_dir.resolve()
    checkpoint = output / 'hf_checkpoint'

    config = tiny_config( arguments.rope_scaling )
    torch.manual_seed( arguments.seed )
    model = LlamaForCausalLM( config ).eval()
    randomize( model, arguments.seed + 1 )

    output.mkdir( parents=True, exist_ok=True )
    model.save_pretrained( checkpoint, safe_serialization=True )

    total = PROMPT_TOKENS + DECODE_TOKENS
    ids = [ ( i * 37 + 5 ) % config.vocab_size for i in range( total ) ]
    tokens = torch.tensor( [ ids ], dtype=torch.long )

    with torch.no_grad():
        all_logits = model( input_ids=tokens, use_cache=False ).logits[ 0 ]

    # The prefill's last position, then the position each decode step feeds.
    logits = all_logits[ PROMPT_TOKENS - 1 : total ].contiguous()

    # Position i predicts token i + 1, so the last position predicts nothing.
    next_token_log_probabilities = torch.log_softmax( all_logits[ : total - 1 ].to( torch.float64 ), dim=-1 ) \
        .gather( 1, tokens[ 0, 1 : ].unsqueeze( 1 ) ).squeeze( 1 )

    loss = torch.nn.functional.cross_entropy( all_logits[ : total - 1 ], tokens[ 0, 1 : ] )

    save_file( {
        'tokens': torch.tensor( ids, dtype=torch.int32 ),
        'logits': logits.to( torch.float32 ),
        'next_token_log_probabilities': next_token_log_probabilities.to( torch.float32 ),
    }, str( output / 'llama_tiny_reference.safetensors' ), metadata={
        'prompt_tokens': str( PROMPT_TOKENS ),
        'decode_tokens': str( DECODE_TOKENS ),
        'seed': str( arguments.seed ),
        'rope_scaling': arguments.rope_scaling,
        'transformers': transformers.__version__,
        'torch': torch.__version__,
    } )

    # A forward-slash path: the converter names the model after the path's last component.
    convert_llama( checkpoint.as_posix(), str( output / 'llama_tiny_fp32.bin' ), 'float32' )
    convert_llama( checkpoint.as_posix(), str( output / 'llama_tiny_bf16.bin' ), 'bfloat16' )

    print( f'\nwrote {output}' )
    print( f'logits max |value| {logits.abs().max().item():.4f}, '
           f'argmax per step {logits.argmax( dim=-1 ).tolist()}' )
    print( f'log-likelihood {next_token_log_probabilities.sum().item():.9f} over {total - 1} positions; '
           f'HuggingFace loss agrees: {abs( -next_token_log_probabilities.mean().item() - loss.item() ) < 1e-5}' )


if __name__ == '__main__':
    main()
