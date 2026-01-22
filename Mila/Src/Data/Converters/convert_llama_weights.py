# ============================================================================
# File: convert_llama.py
# Convert Llama weights to Mila format
# ============================================================================

"""
Convert Llama 3/3.1/3.2 weights from HuggingFace to Mila binary format.

Usage:
    python convert_llama.py --model meta-llama/Llama-3.2-1B --output ../Weights/llama/llama-3.2-1b.bin
    python convert_llama.py --model meta-llama/Llama-3.2-3B --output ../Weights/llama/llama-3.2-3b.bin
    python convert_llama.py --model meta-llama/Meta-Llama-3-8B --output ../Weights/llama/llama-3-8b.bin

Note: You'll need HuggingFace access token for official Meta models.
Set it with: huggingface-cli login
"""

import argparse
from transformers import AutoModelForCausalLM, AutoConfig
from common import MilaWeightWriter, convert_dtype


def convert_llama(model_name: str, output_path: str, dtype: str = 'bfloat16'):
    """
    Convert Llama model to Mila format.
    
    Llama Architecture (Mila component mapping):
    - model.embed_tokens.weight -> token_embedding.weight
    - model.layers.{i}.input_layernorm.weight -> layers.{i}.norm1.weight
    - model.layers.{i}.self_attn.q_proj.weight -> layers.{i}.attention.q_proj.weight
    - model.layers.{i}.self_attn.k_proj.weight -> layers.{i}.attention.k_proj.weight
    - model.layers.{i}.self_attn.v_proj.weight -> layers.{i}.attention.v_proj.weight
    - model.layers.{i}.self_attn.o_proj.weight -> layers.{i}.attention.out_proj.weight
    - model.layers.{i}.post_attention_layernorm.weight -> layers.{i}.norm2.weight
    - model.layers.{i}.mlp.gate_proj.weight -> layers.{i}.mlp.gate_proj.weight
    - model.layers.{i}.mlp.up_proj.weight -> layers.{i}.mlp.up_proj.weight
    - model.layers.{i}.mlp.down_proj.weight -> layers.{i}.mlp.down_proj.weight
    - model.norm.weight -> final_norm.weight
    - lm_head.weight -> lm_head.weight
    
    Note: Llama uses RMSNorm (no bias), GQA, RoPE, and SwiGLU
    """
    
    print(f"Loading {model_name} from HuggingFace...")
    print("(This may take a while for large models...)")
    
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype='auto',  # Use model's native dtype
        device_map='cpu',    # Load to CPU to avoid GPU memory issues
    )
    
    print(f"\nModel config:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"  rope_theta: {config.rope_theta}")
    
    writer = MilaWeightWriter(output_path)
    
    # Set metadata
    writer.set_metadata({
        'architecture': 'llama3',
        'model_name': model_name,
        'vocab_size': config.vocab_size,
        'max_position_embeddings': config.max_position_embeddings,
        'embedding_dim': config.hidden_size,
        'num_layers': config.num_hidden_layers,
        'num_heads': config.num_attention_heads,
        'num_kv_heads': config.num_key_value_heads,
        'hidden_dim': config.intermediate_size,
        'use_bias': False,
        'activation': 'swiglu',
        'norm_type': 'rmsnorm',
        'norm_epsilon': config.rms_norm_eps,
        'attention_type': 'grouped_query',
        'positional_encoding': 'rope',
        'rope_theta': config.rope_theta,
        'rope_scaling': getattr(config, 'rope_scaling', None),
    })
    
    # Convert weights
    state_dict = model.state_dict()
    
    # Token embeddings
    writer.add_tensor(
        'token_embedding.weight',
        convert_dtype(state_dict['model.embed_tokens.weight'].numpy(), dtype)
    )
    
    # Transformer layers
    for i in range(config.num_hidden_layers):
        prefix_hf = f'model.layers.{i}'
        prefix_mila = f'layers.{i}'
        
        # Input RMSNorm (pre-attention)
        writer.add_tensor(
            f'{prefix_mila}.norm1.weight',
            convert_dtype(state_dict[f'{prefix_hf}.input_layernorm.weight'].numpy(), dtype)
        )
        
        # Attention (separate Q, K, V projections for GQA)
        writer.add_tensor(
            f'{prefix_mila}.attention.q_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.self_attn.q_proj.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.attention.k_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.self_attn.k_proj.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.attention.v_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.self_attn.v_proj.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.attention.out_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.self_attn.o_proj.weight'].numpy(), dtype)
        )
        
        # Post-attention RMSNorm (pre-MLP)
        writer.add_tensor(
            f'{prefix_mila}.norm2.weight',
            convert_dtype(state_dict[f'{prefix_hf}.post_attention_layernorm.weight'].numpy(), dtype)
        )
        
        # SwiGLU MLP (3 projections: gate, up, down)
        writer.add_tensor(
            f'{prefix_mila}.mlp.gate_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.gate_proj.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.mlp.up_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.up_proj.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.mlp.down_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.down_proj.weight'].numpy(), dtype)
        )
    
    # Final RMSNorm
    writer.add_tensor(
        'final_norm.weight',
        convert_dtype(state_dict['model.norm.weight'].numpy(), dtype)
    )
    
    # LM head (output projection to vocabulary)
    writer.add_tensor(
        'lm_head.weight',
        convert_dtype(state_dict['lm_head.weight'].numpy(), dtype)
    )
    
    # Write to file
    writer.write()
    
    print(f"\n? Conversion complete!")
    print(f"  Output: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Llama weights to Mila format')
    parser.add_argument('--model', type=str, required=True,
                       help='Llama model name (e.g., meta-llama/Llama-3.2-1B)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for Mila weight file')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Target dtype for weights')
    
    args = parser.parse_args()
    convert_llama(args.model, args.output, args.dtype)