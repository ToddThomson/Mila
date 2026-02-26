from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2")
tok   = GPT2Tokenizer.from_pretrained("gpt2")

ids = tok.encode("Once upon a time", return_tensors="pt")
out = model.generate(ids, max_new_tokens=64, do_sample=False)  # greedy
print(tok.decode(out[0]))