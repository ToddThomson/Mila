import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained( model_id, torch_dtype=torch.float32 )

# Print layer 0 structure so we can see the exact attribute names
print( "Layer 0 attributes:" )
for name, module in model.model.layers[0].named_children():
    print( f"  {name}: {type(module).__name__}" )

print( "\nMLP attributes:" )
for name, module in model.model.layers[0].mlp.named_children():
    print( f"  {name}: {type(module).__name__}" )

print( "\nAttention attributes:" )
for name, module in model.model.layers[0].self_attn.named_children():
    print( f"  {name}: {type(module).__name__}" )