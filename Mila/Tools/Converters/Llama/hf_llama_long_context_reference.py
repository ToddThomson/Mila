# HuggingFace reference for Llama 3.2 1B past its original 8192-token context (ModelFamilyParity.md 8.4, L2).
#
# Every Llama 3.1 and 3.2 checkpoint reads its positions through Llama 3's frequency scaling, whose original context
# is 8192. This records, at FP32, the log-probability HuggingFace gives each next token of the first 10240 tokens of
# the wikitext-2 test split, and a greedy continuation of a short prompt -- what Llama.LongContext.Cuda.cpp compares
# Mila against, band by band on each side of 8192. Needs a CUDA device and the checkpoint in the HuggingFace cache.
#
#   python hf_llama_long_context_reference.py --output ../../../../Data/models/llama/llama32_1b_long_context_reference.safetensors

import argparse
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


MODEL = 'meta-llama/Llama-3.2-1B'
SEQUENCE_TOKENS = 10240
GREEDY_PROMPT = 'The history of the printing press begins in'
GREEDY_TOKENS = 24

# The head runs over this many rows at a time: 10240 rows of 128256 logits at FP32 would be 5 GiB.
HEAD_ROWS = 512
PREFILL_ROWS = 1024


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--corpus', type=Path,
        default=Path( __file__ ).resolve().parents[ 2 ] / 'Quantization' / 'corpus' / 'wiki.test.raw' )
    parser.add_argument( '--output', type=Path, required=True )
    arguments = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained( MODEL )
    model = AutoModelForCausalLM.from_pretrained( MODEL, dtype=torch.float32, attn_implementation='sdpa' ).cuda().eval()

    # Four characters per token over-reads; the tokenizer adds <|begin_of_text|>.
    text = arguments.corpus.read_text( encoding='utf-8' )[ : SEQUENCE_TOKENS * 6 ]
    ids = tokenizer( text, return_tensors='pt' ).input_ids[ :, : SEQUENCE_TOKENS ].cuda()

    assert ids.shape[ 1 ] == SEQUENCE_TOKENS, f'corpus gave {ids.shape[ 1 ]} tokens'
    assert ids[ 0, 0 ].item() == tokenizer.bos_token_id

    with torch.no_grad():
        # Chunked over HuggingFace's own KV cache: one pass would materialize a 10240-square score matrix per
        # head at FP32, 12.5 GiB.
        cache = DynamicCache()
        hidden_chunks = []

        for start in range( 0, SEQUENCE_TOKENS, PREFILL_ROWS ):
            chunk = ids[ :, start : start + PREFILL_ROWS ]
            outputs = model.model( input_ids=chunk, past_key_values=cache, use_cache=True )
            cache = outputs.past_key_values
            hidden_chunks.append( outputs.last_hidden_state[ 0 ] )

        hidden = torch.cat( hidden_chunks )

        log_probabilities = []

        for start in range( 0, SEQUENCE_TOKENS - 1, HEAD_ROWS ):
            end = min( start + HEAD_ROWS, SEQUENCE_TOKENS - 1 )
            rows = model.lm_head( hidden[ start : end ] ).to( torch.float64 )
            targets = ids[ 0, start + 1 : end + 1 ]
            log_probabilities.append( torch.log_softmax( rows, dim=-1 ).gather( 1, targets.unsqueeze( 1 ) ).squeeze( 1 ) )

        next_token_log_probabilities = torch.cat( log_probabilities )

        prompt = tokenizer( GREEDY_PROMPT, return_tensors='pt' ).input_ids.cuda()
        generated = model.generate( prompt, max_new_tokens=GREEDY_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id )

    arguments.output.parent.mkdir( parents=True, exist_ok=True )

    save_file( {
        'tokens': ids[ 0 ].to( torch.int32 ).cpu(),
        'next_token_log_probabilities': next_token_log_probabilities.to( torch.float32 ).cpu(),
        'greedy_prompt': prompt[ 0 ].to( torch.int32 ).cpu(),
        'greedy_tokens': generated[ 0, prompt.shape[ 1 ] : ].to( torch.int32 ).cpu(),
    }, str( arguments.output ), metadata={
        'model': MODEL,
        'transformers': transformers.__version__,
        'torch': torch.__version__,
    } )

    below = next_token_log_probabilities[ : 8191 ]
    above = next_token_log_probabilities[ 8191 : ]

    print( f'wrote {arguments.output}' )
    print( f'mean negative log-likelihood: positions 1-8191 {-below.mean().item():.6f}, '
           f'8192-{SEQUENCE_TOKENS - 1} {-above.mean().item():.6f}' )
    print( f'greedy: {tokenizer.decode( generated[ 0, prompt.shape[ 1 ] : ] )!r}' )


if __name__ == '__main__':
    main()
