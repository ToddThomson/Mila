# ============================================================================
# File: convert_gpt2.py
# Convert GPT-2 weights to Mila format
# ============================================================================

"""
Convert GPT-2 weights from HuggingFace to Mila binary format.

Usage:
    python convert_gpt2.py --model gpt2 --output ../Weights/gpt2/gpt2_small.bin
    python convert_gpt2.py --model gpt2-medium --output ../Weights/gpt2/gpt2_medium.bin
    python convert_gpt2.py --model gpt2-large --output ../Weights/gpt2/gpt2_large.bin
    python convert_gpt2.py --model gpt2-xl --output ../Weights/gpt2/gpt2_xl.bin
"""

import argparse
from transformers import GPT2LMHeadModel
from common import MilaWeightWriter, convert_dtype


def convert_gpt2(model_name: str, output_path: str, dtype: str = 'float32'):
    """
    Convert GPT-2 model to Mila format.
    
    GPT-2 Architecture (Mila component mapping):
    - transformer.wte.weight -> token_embedding.weight
    - transformer.wpe.weight -> position_embedding.weight
    - transformer.h.{i}.ln_1.weight -> layers.{i}.norm1.weight
    - transformer.h.{i}.ln_1.bias -> layers.{i}.norm1.bias
    - transformer.h.{i}.attn.c_attn.weight -> layers.{i}.attention.qkv_proj.weight
    - transformer.h.{i}.attn.c_attn.bias -> layers.{i}.attention.qkv_proj.bias
    - transformer.h.{i}.attn.c_proj.weight -> layers.{i}.attention.out_proj.weight
    - transformer.h.{i}.attn.c_proj.bias -> layers.{i}.attention.out_proj.bias
    - transformer.h.{i}.ln_2.weight -> layers.{i}.norm2.weight
    - transformer.h.{i}.ln_2.bias -> layers.{i}.norm2.bias
    - transformer.h.{i}.mlp.c_fc.weight -> layers.{i}.mlp.fc1.weight
    - transformer.h.{i}.mlp.c_fc.bias -> layers.{i}.mlp.fc1.bias
    - transformer.h.{i}.mlp.c_proj.weight -> layers.{i}.mlp.fc2.weight
    - transformer.h.{i}.mlp.c_proj.bias -> layers.{i}.mlp.fc2.bias
    - transformer.ln_f.weight -> final_norm.weight
    - transformer.ln_f.bias -> final_norm.bias
    - lm_head.weight -> lm_head.weight
    """
    
    print(f"Loading {model_name} from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    config = model.config
    
    print(f"Model config:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  n_positions: {config.n_positions}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_head: {config.n_head}")
    
    writer = MilaWeightWriter(output_path)
    
    # Set metadata
    writer.set_metadata({
        'architecture': 'gpt2',
        'model_name': model_name,
        'vocab_size': config.vocab_size,
        'max_position_embeddings': config.n_positions,
        'embedding_dim': config.n_embd,
        'num_layers': config.n_layer,
        'num_heads': config.n_head,
        'hidden_dim': config.n_inner if config.n_inner else config.n_embd * 4,
        'use_bias': True,
        'activation': 'gelu',
        'norm_type': 'layernorm',
        'attention_type': 'standard',
        'positional_encoding': 'learned',
    })
    
    # Convert weights
    state_dict = model.state_dict()
    
    # Token embeddings
    writer.add_tensor(
        'token_embedding.weight',
        convert_dtype(state_dict['transformer.wte.weight'].numpy(), dtype)
    )
    
    # Position embeddings
    writer.add_tensor(
        'position_embedding.weight',
        convert_dtype(state_dict['transformer.wpe.weight'].numpy(), dtype)
    )
    
    # Transformer layers
    for i in range(config.n_layer):
        prefix_hf = f'transformer.h.{i}'
        prefix_mila = f'layers.{i}'
        
        # Layer Norm 1 (pre-attention)
        writer.add_tensor(
            f'{prefix_mila}.norm1.weight',
            convert_dtype(state_dict[f'{prefix_hf}.ln_1.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.norm1.bias',
            convert_dtype(state_dict[f'{prefix_hf}.ln_1.bias'].numpy(), dtype)
        )
        
        # Attention (c_attn is fused QKV projection)
        writer.add_tensor(
            f'{prefix_mila}.attention.qkv_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.attn.c_attn.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.attention.qkv_proj.bias',
            convert_dtype(state_dict[f'{prefix_hf}.attn.c_attn.bias'].numpy(), dtype)
        )
        
        # Attention output projection
        writer.add_tensor(
            f'{prefix_mila}.attention.out_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.attn.c_proj.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.attention.out_proj.bias',
            convert_dtype(state_dict[f'{prefix_hf}.attn.c_proj.bias'].numpy(), dtype)
        )
        
        # Layer Norm 2 (pre-MLP)
        writer.add_tensor(
            f'{prefix_mila}.norm2.weight',
            convert_dtype(state_dict[f'{prefix_hf}.ln_2.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.norm2.bias',
            convert_dtype(state_dict[f'{prefix_hf}.ln_2.bias'].numpy(), dtype)
        )
        
        # MLP
        writer.add_tensor(
            f'{prefix_mila}.mlp.fc1.weight',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.c_fc.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.mlp.fc1.bias',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.c_fc.bias'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.mlp.fc2.weight',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.c_proj.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.mlp.fc2.bias',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.c_proj.bias'].numpy(), dtype)
        )
    
    # Final layer norm
    writer.add_tensor(
        'final_norm.weight',
        convert_dtype(state_dict['transformer.ln_f.weight'].numpy(), dtype)
    )
    writer.add_tensor(
        'final_norm.bias',
        convert_dtype(state_dict['transformer.ln_f.bias'].numpy(), dtype)
    )
    
    # LM head (output projection to vocabulary)
    writer.add_tensor(
        'lm_head.weight',
        convert_dtype(state_dict['lm_head.weight'].numpy(), dtype)
    )
    
    # Write to file
    writer.write()
    
    print(f"\n Conversion complete!")
    print(f"  Output: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert GPT-2 weights to Mila format')
    parser.add_argument('--model', type=str, default='gpt2',
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                       help='GPT-2 model variant')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for Mila weight file')
    parser.add_argument('--dtype', type=str, default='float32',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Target dtype for weights')
    
    args = parser.parse_args()
    convert_gpt2(args.model, args.output, args.dtype)

