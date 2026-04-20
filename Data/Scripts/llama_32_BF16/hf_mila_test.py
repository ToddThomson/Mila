import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained( 'gpt2' )
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained( 'gpt2' )

# This matches Mila's decode step 0:
# - Prefill: "Once upon a time" (4 tokens)
# - First decoded token: ',' (token 11)
input_text = "Once upon a time,"
input_ids = tokenizer.encode( input_text, return_tensors='pt' )
print( f"Token ids: {input_ids}" )
print( f"Seq len: {input_ids.shape[1]}" )

captured = {}

def make_hook_last( name ):
    """Captures min/max of the LAST token position only."""
    def fn( module, input, output ):
        t = output if isinstance( output, torch.Tensor ) else output[0]
        last = t[0, -1, :]
        captured[name] = ( last.min().item(), last.max().item() )
    return fn

hooks = []
for i, block in enumerate( model.transformer.h ):
    hooks.append( block.ln_1.register_forward_hook( make_hook_last( f'layer_{i}.ln_1' ) ) )
    hooks.append( block.attn.c_attn.register_forward_hook( make_hook_last( f'layer_{i}.fc_qkv_proj' ) ) )
    hooks.append( block.attn.c_proj.register_forward_hook( make_hook_last( f'layer_{i}.fc_out_proj' ) ) )
    hooks.append( block.ln_2.register_forward_hook( make_hook_last( f'layer_{i}.ln_2' ) ) )
    hooks.append( block.mlp.c_fc.register_forward_hook( make_hook_last( f'layer_{i}.mlp.fc_1' ) ) )
    hooks.append( block.mlp.act.register_forward_hook( make_hook_last( f'layer_{i}.mlp.gelu' ) ) )
    hooks.append( block.mlp.c_proj.register_forward_hook( make_hook_last( f'layer_{i}.mlp.fc_2' ) ) )
    hooks.append( block.register_forward_hook( make_hook_last( f'layer_{i}.residual_out' ) ) )

with torch.no_grad():
    out = model( input_ids )
    logits = out.logits[0, -1]
    top5 = torch.topk( logits, 5 )
    print( f"\nTop 5 predictions after '{input_text}':" )
    for v, idx in zip( top5.values, top5.indices ):
        print( f"  {tokenizer.decode([idx.item()])!r:15} {v.item():.4f}" )

for h in hooks:
    h.remove()

print( "\n=== Per-layer values at LAST token position ===" )
for i in range( 12 ):
    print( f"\nlayer_{i}:" )
    for key in [
        f'layer_{i}.ln_1',
        f'layer_{i}.fc_qkv_proj',
        f'layer_{i}.fc_out_proj',
        f'layer_{i}.ln_2',
        f'layer_{i}.mlp.fc_1',
        f'layer_{i}.mlp.gelu',
        f'layer_{i}.mlp.fc_2',
        f'layer_{i}.residual_out',
    ]:
        if key in captured:
            mn, mx = captured[key]
            print( f"  {key}: [{mn:.3f}, {mx:.3f}]" )