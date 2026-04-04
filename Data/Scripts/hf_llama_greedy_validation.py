from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

ids = tokenizer.encode("Once upon a time", return_tensors="pt", add_special_tokens=False)
out = model.generate(ids, max_new_tokens=64, do_sample=False)  # greedy
print(tokenizer.decode(out[0]))