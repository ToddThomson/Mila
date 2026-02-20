import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Run the full 5-token sequence in one shot
# Token 11 = ',' which appears to be what Mila predicted as first decode token
input_ids = tokenizer.encode("Once upon a time")
print(f"Token ids: {input_ids}")

captured = {}
def make_hook(name):
    def fn(module, input, output):
        t = output if isinstance(output, torch.Tensor) else output[0]
        # Take only last token: shape [1, seq, 768] -> [768]
        last = t[0, -1, :]
        captured[name] = (last.min().item(), last.max().item())
    return fn

hooks = []
for i, block in enumerate(model.transformer.h):
    hooks.append(block.attn.c_proj.register_forward_hook(make_hook(f'layer_{i}.fc_out_proj')))
    hooks.append(block.register_forward_hook(make_hook(f'layer_{i}.residual_out')))

with torch.no_grad():
    out = model(torch.tensor([input_ids]))
    logits = out.logits[0, -1]  # last token position
    top5 = torch.topk(logits, 5)
    print("\nTop 5 predictions after 'Once upon a time':")
    for v, i in zip(top5.values, top5.indices):
        print(f"  {tokenizer.decode([i.item()])!r:15} {v.item():.4f}")

for h in hooks:
    h.remove()

print(f"\nlayer_0 fc_out_proj: {captured['layer_0.fc_out_proj']}")
print(f"layer_0 residual_out: {captured['layer_0.residual_out']}")
print(f"layer_1 residual_out: {captured['layer_1.residual_out']}")
print(f"layer_2 residual_out: {captured['layer_2.residual_out']}")