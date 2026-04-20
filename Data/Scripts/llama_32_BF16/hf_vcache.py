import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
print(f"Token ids: {input_ids}")
print(f"Seq len: {input_ids.shape[1]}")

captured = {}

def make_hook(name):
    def fn(module, input, output):
        captured[name] = output.detach()
    return fn

hooks = []
hooks.append(model.transformer.h[0].attn.c_attn.register_forward_hook(make_hook('layer_0.qkv')))
hooks.append(model.transformer.h[1].attn.c_attn.register_forward_hook(make_hook('layer_1.qkv')))

with torch.no_grad():
    model(input_ids)

for h in hooks:
    h.remove()

def extract_v( qkv, layer_name, num_heads=12, head_size=64 ):
    # qkv shape: [1, T, 2304]
    T = qkv.shape[1]
    q, k, v = qkv.split( 768, dim=-1 )  # each [1, T, 768]
    # Reshape to [B, NH, T, HS]
    v = v.view( 1, T, num_heads, head_size ).permute( 0, 2, 1, 3 )  # [1, 12, T, 64]
    k = k.view( 1, T, num_heads, head_size ).permute( 0, 2, 1, 3 )  # [1, 12, T, 64]
    print( f"\n=== {layer_name} ===" )
    for head in range( 2 ):  # show head 0 and head 1
        print( f"\n  V head {head}, positions 0..{T-1} (first 8 elements each):" )
        for pos in range( T ):
            vals = v[0, head, pos, :8].tolist()
            formatted = "  ".join( f"{x:10.6f}" for x in vals )
            print( f"    pos {pos}: [ {formatted} ... ]" )
        print( f"\n  K head {head}, positions 0..{T-1} (first 8 elements each):" )
        for pos in range( T ):
            vals = k[0, head, pos, :8].tolist()
            formatted = "  ".join( f"{x:10.6f}" for x in vals )
            print( f"    pos {pos}: [ {formatted} ... ]" )

extract_v( captured['layer_0.qkv'], 'Layer 0' )
extract_v( captured['layer_1.qkv'], 'Layer 1' )