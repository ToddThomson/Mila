# A Q4_0 GGUF of Llama 3.1 8B Instruct holding exactly the weights Mila's Q4_0 build holds (ModelFamilyParity.md 8.4,
# L3, test 3).
#
# Published Q4_0 GGUFs of this model are fitted with an importance matrix, mix in Q4_1 and Q6_K, and so hold other
# weights than the reference rule gives. This writes one that llama.cpp runs and that Mila's build equals code for
# code: every projection rounded by gguf-py's port of llama.cpp's reference Q4_0 rule, the embedding table and head
# at BF16 as in Mila's Llama packages, norms at F32. Metadata, tokenizer and the RoPE scaling table are copied from
# a published GGUF of the same model; query and key rows are permuted into llama.cpp's rotary layout as its
# converter does.
#
#   python llama_q4_0_gguf.py --template <any Llama-3.1-8B-Instruct .gguf> --output ../../../../Data/models/llama/llama31_8b_instruct_q4_0.gguf

import argparse
import json
import re
from pathlib import Path

import numpy as np
import gguf
from huggingface_hub import snapshot_download
from safetensors import safe_open

CHECKPOINT = 'meta-llama/Llama-3.1-8B-Instruct'

PROJECTIONS = {
    'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj', 'attn_v': 'self_attn.v_proj',
    'attn_output': 'self_attn.o_proj', 'ffn_gate': 'mlp.gate_proj', 'ffn_up': 'mlp.up_proj', 'ffn_down': 'mlp.down_proj',
}
NORMS = { 'attn_norm': 'input_layernorm', 'ffn_norm': 'post_attention_layernorm' }


def permute( weights: np.ndarray, heads: int ) -> np.ndarray:
    """HuggingFace's rotate-half rows to llama.cpp's interleaved pairs, as convert_hf_to_gguf.py's LlamaModel does."""
    return weights.reshape( heads, 2, weights.shape[ 0 ] // heads // 2, *weights.shape[ 1: ] ).swapaxes( 1, 2 ).reshape( weights.shape )


def bf16_bits( values: np.ndarray ) -> np.ndarray:
    """The stored BF16 bit patterns, unchanged: safetensors hands BF16 over as float32 widened exactly."""
    return ( values.astype( np.float32 ).view( np.uint32 ) >> 16 ).astype( np.uint16 )


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--template', type=Path, required=True )
    parser.add_argument( '--output', type=Path, required=True )
    arguments = parser.parse_args()

    template = gguf.GGUFReader( arguments.template )
    source = Path( snapshot_download( CHECKPOINT, allow_patterns=[ '*.safetensors', '*.json' ] ) )

    with open( source / 'config.json' ) as config_file:
        config = json.load( config_file )

    heads = config[ 'num_attention_heads' ]
    kv_heads = config[ 'num_key_value_heads' ]

    with open( source / 'model.safetensors.index.json' ) as index_file:
        shard_of = json.load( index_file )[ 'weight_map' ]

    shards = { name: safe_open( source / name, framework='pt' ) for name in set( shard_of.values() ) }

    def read( name: str ) -> np.ndarray:
        return shards[ shard_of[ name ] ].get_tensor( name ).float().numpy()

    writer = gguf.GGUFWriter( arguments.output, arch='llama' )

    for field in template.fields.values():
        # The template's quantizer and importance-matrix records describe its weights, not these.
        if ( field.name in ( gguf.Keys.General.ARCHITECTURE, 'general.quantized_by' ) or field.name.startswith( 'GGUF.' )
                or field.name.startswith( 'quantize.' ) ):
            continue

        value_type = field.types[ 0 ]
        sub_type = field.types[ -1 ] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value( field.name, field.contents(), value_type, sub_type=sub_type )

    for tensor in template.tensors:
        block = re.fullmatch( r'blk\.(\d+)\.(\w+)\.weight', tensor.name )

        if tensor.name == 'rope_freqs.weight':
            writer.add_tensor( tensor.name, np.array( tensor.data ), raw_dtype=gguf.GGMLQuantizationType.F32 )
        elif tensor.name in ( 'token_embd.weight', 'output.weight' ):
            values = read( 'model.embed_tokens.weight' if tensor.name == 'token_embd.weight' else 'lm_head.weight' )
            writer.add_tensor( tensor.name, bf16_bits( values ), raw_shape=values.shape, raw_dtype=gguf.GGMLQuantizationType.BF16 )
        elif tensor.name == 'output_norm.weight':
            writer.add_tensor( tensor.name, read( 'model.norm.weight' ) )
        elif block and block.group( 2 ) in NORMS:
            writer.add_tensor( tensor.name, read( f'model.layers.{block.group( 1 )}.{NORMS[ block.group( 2 ) ]}.weight' ) )
        elif block and block.group( 2 ) in PROJECTIONS:
            values = read( f'model.layers.{block.group( 1 )}.{PROJECTIONS[ block.group( 2 ) ]}.weight' )

            if block.group( 2 ) == 'attn_q':
                values = permute( values, heads )
            elif block.group( 2 ) == 'attn_k':
                values = permute( values, kv_heads )

            writer.add_tensor( tensor.name, gguf.quants.quantize( values, gguf.GGMLQuantizationType.Q4_0 ),
                raw_dtype=gguf.GGMLQuantizationType.Q4_0 )
            print( f'  {tensor.name}', flush=True )
        else:
            raise ValueError( f'no source for {tensor.name}' )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file( progress=True )
    writer.close()

    print( f'wrote {arguments.output}' )


if __name__ == '__main__':
    main()
