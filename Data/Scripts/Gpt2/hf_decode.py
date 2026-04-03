import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model     = GPT2LMHeadModel.from_pretrained( 'gpt2' )
tokenizer = GPT2Tokenizer.from_pretrained( 'gpt2' )
model.eval()

input_ids = tokenizer.encode( "Once upon a time", return_tensors='pt' )
print( f"Token ids: {input_ids.tolist()}" )

with torch.no_grad():
    out = model( input_ids )

logits = out.logits[ 0 ]  # [T, V]

print( "\n=== Token 11 logit at every position ===" )
for pos in range( 4 ):
    print( f"HF token 11 (',') at pos {pos}: {logits[ pos, 11 ].item():.4f}" )

print( "\n=== Top token at every position ===" )
for pos in range( 4 ):
    top_token = logits[ pos ].argmax().item()
    top_logit = logits[ pos, top_token ].item()
    print( f"HF top token at pos {pos}: token={top_token} '{tokenizer.decode([top_token])}' logit={top_logit:.4f}" )

print( f"\n=== Token 11 at pos 3 (expected ~-50.47) ===" )
print( f"HF token 11 at pos 3: {logits[ 3, 11 ].item():.4f}" )