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
    Convert GPT-2 model to Mila component format.
    
    GPT-2 Architecture (Mila component mapping):
    - transformer.wte.weight -> lenc.wte
    - transformer.wpe.weight -> lenc.wpe

    - transformer.h.{i}.ln_1.weight -> tf_layer_{i}.ln_1.weight
    - transformer.h.{i}.ln_1.bias -> tf_layer_{i}.ln_1.bias

    - transformer.h.{i}.attn.c_attn.weight -> tf_layer_{i}.fc_qkv_proj.weight
    - transformer.h.{i}.attn.c_attn.bias -> tf_layer_{i}.fc_qkv_proj.bias

    - transformer.h.{i}.attn.c_proj.weight -> tf_layer_{i}.fc_out_proj.weight
    - transformer.h.{i}.attn.c_proj.bias -> tf_layer_{i}.fc_out_proj.bias

    - transformer.h.{i}.ln_2.weight -> tf_layer_{i}.ln_2.weight
    - transformer.h.{i}.ln_2.bias -> tf_layer_{i}.ln_2.bias

    - transformer.h.{i}.mlp.c_fc.weight -> tf_layer_{i}.mlp.fc_1.weight
    - transformer.h.{i}.mlp.c_fc.bias -> tf_layer_{i}.mlp.fc_1.bias
    - transformer.h.{i}.mlp.c_proj.weight -> tf_layer_{i}.mlp.fc_2.weight
    - transformer.h.{i}.mlp.c_proj.bias -> tf_layer_{i}.mlp.fc_2.bias

    - transformer.ln_f.weight -> ln_final.weight
    - transformer.ln_f.bias -> ln_final.bias

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
        'dtype': dtype,
        'vocab_size': config.vocab_size,
        'max_seq_length': config.n_positions,
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

    # Token embeddings (HuggingFace transformer.wte -> Mila wte)
    writer.add_tensor(
        'lenc.wte',
        convert_dtype(state_dict['transformer.wte.weight'].numpy(), dtype)
    )
    
    # Position embeddings (transformer.wpe -> wpe)
    writer.add_tensor(
        'lenc.wpe',
        convert_dtype(state_dict['transformer.wpe.weight'].numpy(), dtype)
    )
    
    # Transformer layers using the new mapping (tf_layer_{i}.*)
    for i in range(config.n_layer):
        prefix_hf = f'transformer.h.{i}'
        prefix_mila = f'tf_layer_{i}'
        
        # Layer Norm 1 (pre-attention)
        writer.add_tensor(
            f'{prefix_mila}.ln_1.weight',
            convert_dtype(state_dict[f'{prefix_hf}.ln_1.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.ln_1.bias',
            convert_dtype(state_dict[f'{prefix_hf}.ln_1.bias'].numpy(), dtype)
        )
        
        # Attention (c_attn is fused QKV projection -> fc_qkv_proj)
        # Note: Transpose weights for fc_qkv_proj to match Mila's expected layout (out_features, in_features)
        writer.add_tensor(
            f'{prefix_mila}.fc_qkv_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.attn.c_attn.weight'].numpy().T, dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.fc_qkv_proj.bias',
            convert_dtype(state_dict[f'{prefix_hf}.attn.c_attn.bias'].numpy(), dtype)
        )
        
        # Attention output projection -> fc_out_proj
        # Note: Transpose weights for fc_out_proj to match Mila's expected layout (out_features, in_features)
        writer.add_tensor(
            f'{prefix_mila}.fc_out_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.attn.c_proj.weight'].numpy().T, dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.fc_out_proj.bias',
            convert_dtype(state_dict[f'{prefix_hf}.attn.c_proj.bias'].numpy(), dtype)
        )
        
        # Layer Norm 2 (pre-MLP)
        writer.add_tensor(
            f'{prefix_mila}.ln_2.weight',
            convert_dtype(state_dict[f'{prefix_hf}.ln_2.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.ln_2.bias',
            convert_dtype(state_dict[f'{prefix_hf}.ln_2.bias'].numpy(), dtype)
        )
        
        # MLP (c_fc -> mlp.fc_1, c_proj -> mlp.fc_2)
        # Note: Transpose weights to match Mila's expected layout (out_features, in_features)
        writer.add_tensor(
            f'{prefix_mila}.mlp.fc_1.weight',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.c_fc.weight'].numpy().T, dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.mlp.fc_1.bias',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.c_fc.bias'].numpy(), dtype)
        )

        writer.add_tensor(
            f'{prefix_mila}.mlp.fc_2.weight',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.c_proj.weight'].numpy().T, dtype)
        )
        
        writer.add_tensor(
            f'{prefix_mila}.mlp.fc_2.bias',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.c_proj.bias'].numpy(), dtype)
        )
    
    # Final layer norm -> ln_final
    writer.add_tensor(
        'ln_final.weight',
        convert_dtype(state_dict['transformer.ln_f.weight'].numpy(), dtype)
    )
    writer.add_tensor(
        'ln_final.bias',
        convert_dtype(state_dict['transformer.ln_f.bias'].numpy(), dtype)
    )
    
    # LM head (output projection to vocabulary)
    # Note: GPT-2's lm_head has no bias term by design (Weight tying with wte)
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
