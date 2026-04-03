import torch
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')

for i in [0, 1]:
    w = model.state_dict()[f'transformer.h.{i}.mlp.c_proj.weight'].T
    print(f"Layer {i} fc_2 after .T: min={w.min():.6f} max={w.max():.6f} mean={w.mean():.6f}")
    print(f"First 5x5:\n{w[:5, :5]}\n")