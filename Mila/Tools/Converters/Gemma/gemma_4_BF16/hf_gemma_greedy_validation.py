# HuggingFace greedy reference for Gemma 4 parity (Step 5f).
#
# Uses the chat template required by the IT (instruction-tuned) model so that
# <start_of_turn>/<end_of_turn> boundary tokens are present. Without them the
# model never sees a valid context and degenerates into repetition loops.
#
# Mila side: feed the printed "prompt ids" (which include the template tokens)
# into mila.GemmaModel.generate(prompt_ids, collected.append, max_new_tokens=64)
# and diff collected against "generated ids" below. The Mila tokenizer is validated
# separately (BpeTokenizerGemma suite), so feeding identical ids isolates MODEL
# parity.
#
# REQUIRES CUDA: BF16 on CPU is numerically unstable for a 48-layer model and
# the generated token IDs will not match Mila's BF16 output.

import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if not torch.cuda.is_available():
    print( "ERROR: CUDA is required for this script." )
    print( "  Install a CUDA-enabled PyTorch build:" )
    print( "    pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128" )
    print( f"  Current torch: {torch.__version__}" )
    sys.exit( 1 )

print( f"Device: CUDA (bfloat16)  torch {torch.__version__}  CUDA {torch.version.cuda}" )

model_id = "google/gemma-4-12b-it"

tokenizer = AutoTokenizer.from_pretrained( model_id )
model = AutoModelForCausalLM.from_pretrained( model_id, dtype=torch.bfloat16 ).to( "cuda" ).eval()

# IT models require the chat template; raw prompts produce degenerate output.
# apply_chat_template behaviour varies by transformers version (may return str,
# list[int], or tokenizers.Encoding). Force tokenize=False to always get the
# formatted string, then encode explicitly to guarantee a plain list[int].
prompt = "The capital of France is"
messages = [{"role": "user", "content": prompt}]
prompt_text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)
# add_special_tokens=False avoids a second BOS since the template already includes it.
token_ids = tokenizer.encode( prompt_text, add_special_tokens=False )
ids = torch.tensor( [token_ids], dtype=torch.long ).to( "cuda" )

prompt_ids = ids[0].tolist()
print( "prompt        :", repr( prompt ) )
print( "prompt ids    :", prompt_ids )

with torch.no_grad():
    out = model.generate( ids, max_new_tokens=64, do_sample=False )  # greedy

all_ids = out[0].tolist()
gen_ids = all_ids[ids.shape[1]:]

print( "generated ids :", gen_ids )
print( "all ids       :", all_ids )
print( "text          :", tokenizer.decode( all_ids ) )

# Copy-paste straight into Tests/Dnn/Models/GemmaModel.Parity.Cuda.cpp.
print()
print( "kPromptIds   = {", ", ".join( map( str, prompt_ids ) ), "};" )
print( "kExpectedGen = {", ", ".join( map( str, gen_ids ) ), "};" )