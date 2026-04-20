import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

model_id = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.eval()

print("Layer 0 attributes (named_children):")
layer0 = model.model.layers[0]
for name, module in layer0.named_children():
    print(f"  {name}: {type(module).__name__}")

print("\nAttention named_children:")
for name, module in layer0.self_attn.named_children():
    print(f"  {name}: {type(module).__name__}")

# Helpers
def stats_from_last_token(t: torch.Tensor):
    # expects [B, T, D] or [B, D]
    if t.dim() == 3:
        last = t[0, -1, :].detach().cpu()
    elif t.dim() == 2:
        last = t[0, :].detach().cpu()
    else:
        raise RuntimeError("unexpected tensor dim for stats")
    return {
        "shape": tuple(last.shape),
        "min": float(last.min().item()),
        "max": float(last.max().item()),
        "mean": float(last.mean().item()),
        "std": float(last.std().item()),
    }

# Prepare a single example input
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Get embeddings
embed_mod = model.get_input_embeddings()
emb = embed_mod(input_ids)  # [B, T, D]
print(f"\nInput embeddings shape: {tuple(emb.shape)}")

# If the layer expects RMSNormed inputs, use input_layernorm if present
hidden = emb
if hasattr(layer0, "input_layernorm"):
    try:
        hidden = layer0.input_layernorm(hidden)
        print("Applied layer0.input_layernorm before projections.")
    except Exception as e:
        print(f"Could not apply input_layernorm: {e}")

# Try q_proj / k_proj first
q_proj = getattr(layer0.self_attn, "q_proj", None)
k_proj = getattr(layer0.self_attn, "k_proj", None)
qkv_proj = getattr(layer0, "qkv_proj", None) or getattr(layer0.self_attn, "qkv_proj", None)

if q_proj is not None and k_proj is not None:
    try:
        q = q_proj(hidden)
        k = k_proj(hidden)
        print("\nFound separate q_proj and k_proj on attention.")
        print(f"  q.shape: {tuple(q.shape)}  k.shape: {tuple(k.shape)}")
        print("  q last-token stats:", stats_from_last_token(q))
        print("  k last-token stats:", stats_from_last_token(k))
    except Exception as e:
        print(f"Error executing q_proj/k_proj: {e}")
else:
    print("\nSeparate q_proj/k_proj not found on attention.")
    if qkv_proj is not None:
        try:
            qkv = qkv_proj(hidden)
            print(f"Found fused qkv_proj (shape {tuple(qkv.shape)}).")
            # Print qkv last-token stats — you can slice by head dims if needed.
            print("  qkv last-token stats:", stats_from_last_token(qkv))
            print("  To extract Q and K from qkv you need head counts/head_dim available in the attention config.")
        except Exception as e:
            print(f"Error executing qkv_proj: {e}")
    else:
        print("No qkv_proj found either. Inspect layer0.self_attn for projection names above.")