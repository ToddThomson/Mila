import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Same input as your Mila test
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
print(f"Token ids: {input_ids}")

# Storage for hook outputs
captured = {}

def make_hook(name):
    def fn(module, input, output):
        t = output if isinstance(output, torch.Tensor) else output[0]
        captured[name] = (t.min().item(), t.max().item(), t.shape)
    return fn

hooks = []

# Encoder output (wte + wpe)
hooks.append(model.transformer.drop.register_forward_hook(make_hook('lenc_out')))

for i, block in enumerate(model.transformer.h):
    # ln_1
    hooks.append(block.ln_1.register_forward_hook(make_hook(f'layer_{i}.ln_1')))
    # fc_qkv_proj (c_attn)
    hooks.append(block.attn.c_attn.register_forward_hook(make_hook(f'layer_{i}.fc_qkv_proj')))
    # fc_out_proj (c_proj)
    hooks.append(block.attn.c_proj.register_forward_hook(make_hook(f'layer_{i}.fc_out_proj')))
    # ln_2
    hooks.append(block.ln_2.register_forward_hook(make_hook(f'layer_{i}.ln_2')))
    # mlp.fc_1 (c_fc)
    hooks.append(block.mlp.c_fc.register_forward_hook(make_hook(f'layer_{i}.mlp.fc_1')))
    # gelu
    hooks.append(block.mlp.act.register_forward_hook(make_hook(f'layer_{i}.mlp.gelu')))
    # mlp.fc_2 (c_proj)
    hooks.append(block.mlp.c_proj.register_forward_hook(make_hook(f'layer_{i}.mlp.fc_2')))
    # full block output (residual)
    hooks.append(block.register_forward_hook(make_hook(f'layer_{i}.residual_out')))

with torch.no_grad():
    model(input_ids)

for h in hooks:
    h.remove()

# Print in same format as your Mila debug output
print(f"\nlenc out: [{captured['lenc_out'][0]:.3f}, {captured['lenc_out'][1]:.3f}]")
print()

for i in range(12):
    for key in [f'layer_{i}.ln_1', f'layer_{i}.fc_qkv_proj', f'layer_{i}.fc_out_proj',
                f'layer_{i}.ln_2', f'layer_{i}.mlp.fc_1', f'layer_{i}.mlp.gelu',
                f'layer_{i}.mlp.fc_2', f'layer_{i}.residual_out']:
        if key in captured:
            mn, mx, shape = captured[key]
            print(f"{key}: [{mn:.3f}, {mx:.3f}] shape={list(shape)}")
    print()

# Print peak residual across all layers to set kResidualAbsLimit
peak = max(abs(captured[f'layer_{i}.residual_out'][1]) for i in range(12))
peak_min = min(captured[f'layer_{i}.residual_out'][0] for i in range(12))
print(f"Peak residual: min={peak_min:.3f}, max={peak:.3f}")
print(f"Suggested kResidualAbsLimit: {peak * 1.5:.1f}")