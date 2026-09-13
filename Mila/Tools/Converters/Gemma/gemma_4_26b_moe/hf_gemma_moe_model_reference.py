# HuggingFace reference for GemmaTransformer with kMixtureOfExperts (Specifications/Gemma4MoE.md, Phase 8).
#
# Builds a tiny Gemma4ForCausalLM with the MoE block enabled and draws every weight at random -- every
# norm, router scale and layer_scalar away from 1, so a misplaced norm or scale cannot hide behind an
# identity. The model is saved as a HuggingFace checkpoint and converted with Gemma/convert_weights.py at
# FP32 and BF16, so one capture holds the converter's names and the block's wiring together. The logits
# recorded are HuggingFace's before the final softcap, which Mila applies at the sampler. No network, no GPU.
#
# Two layers, sliding then global, window 8 against a 12-token prompt: the sliding mask is exercised.
#
#   python hf_gemma_moe_model_reference.py --output-dir ../../../../Data/models/gemma/gemma4_moe_tiny

import argparse
import sys
from pathlib import Path

sys.path.insert( 0, str( Path( __file__ ).resolve().parent.parent ) )

import torch
import transformers
from safetensors.torch import save_file
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from convert_weights import convert_gemma


PROMPT_TOKENS = 12
DECODE_TOKENS = 3


def tiny_config() -> Gemma4TextConfig:
    config = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        global_head_dim=128,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        max_position_embeddings=32,
        sliding_window=8,
        layer_types=[ 'sliding_attention', 'full_attention' ],
        rope_parameters={
            'sliding_attention': { 'rope_type': 'default', 'rope_theta': 10000.0 },
            'full_attention': { 'rope_type': 'proportional', 'partial_rotary_factor': 0.25, 'rope_theta': 250000.0 },
        },
        rms_norm_eps=1e-6,
        hidden_activation='gelu_pytorch_tanh',
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        attention_bias=False,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        enable_moe_block=True,
        num_experts=8,
        top_k_experts=2,
        moe_intermediate_size=64 )

    config._attn_implementation = 'eager'
    config._experts_implementation = 'eager'

    return config


def randomize( model: torch.nn.Module, seed: int ):
    """Every tensor drawn; anything a model would initialize to one is drawn away from it."""
    generator = torch.Generator().manual_seed( seed )

    def uniform( tensor, low, high ):
        return low + ( high - low ) * torch.rand( tensor.shape, generator=generator )

    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if name.endswith( 'layer_scalar' ):
                tensor.copy_( uniform( tensor, 0.5, 1.0 ) )
            elif name.endswith( 'per_expert_scale' ):
                tensor.copy_( uniform( tensor, 0.8, 1.2 ) )
            elif name.endswith( 'router.scale' ) or ( 'norm' in name and name.endswith( 'weight' ) ):
                tensor.copy_( uniform( tensor, 0.5, 1.5 ) )
            else:
                tensor.copy_( torch.randn( tensor.shape, generator=generator ) * 0.05 )


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--seed', type=int, default=20260912 )
    parser.add_argument( '--output-dir', type=Path, required=True )
    arguments = parser.parse_args()

    output = arguments.output_dir
    checkpoint = output / 'hf_checkpoint'

    config = tiny_config()
    torch.manual_seed( arguments.seed )
    model = Gemma4ForCausalLM( config ).eval()
    randomize( model, arguments.seed + 1 )

    output.mkdir( parents=True, exist_ok=True )
    model.save_pretrained( checkpoint, safe_serialization=True )

    total = PROMPT_TOKENS + DECODE_TOKENS
    ids = [ ( i * 37 + 5 ) % config.vocab_size for i in range( total ) ]
    tokens = torch.tensor( [ ids ], dtype=torch.long )

    with torch.no_grad():
        outputs = model( input_ids=tokens, output_hidden_states=True, use_cache=False )
        raw_logits = model.lm_head( outputs.hidden_states[ -1 ] )[ 0 ]

    cap = config.final_logit_softcapping
    softcapped = torch.tanh( raw_logits / cap ) * cap

    # The prefill's last position, then the position each decode step feeds.
    logits = raw_logits[ PROMPT_TOKENS - 1 : total ].contiguous()

    save_file( {
        'tokens': torch.tensor( ids, dtype=torch.int32 ),
        'logits': logits.to( torch.float32 ),
    }, str( output / 'gemma4_moe_tiny_reference.safetensors' ), metadata={
        'prompt_tokens': str( PROMPT_TOKENS ),
        'decode_tokens': str( DECODE_TOKENS ),
        'seed': str( arguments.seed ),
        'transformers': transformers.__version__,
        'torch': torch.__version__,
    } )

    convert_gemma( str( checkpoint ), str( output / 'gemma4_moe_tiny_fp32.bin' ), 'float32' )
    convert_gemma( str( checkpoint ), str( output / 'gemma4_moe_tiny_bf16.bin' ), 'bfloat16' )

    print( f'\nwrote {output}' )
    print( f'raw logits reproduce the model output after the softcap: '
           f'{torch.allclose( softcapped, outputs.logits[ 0 ], atol=1e-6 )}' )
    print( f'logits max |value| {logits.abs().max().item():.4f}, '
           f'argmax per step {logits.argmax( dim=-1 ).tolist()}' )


if __name__ == '__main__':
    main()
