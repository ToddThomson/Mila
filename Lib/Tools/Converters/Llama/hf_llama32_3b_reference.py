# HuggingFace reference for Llama 3.2 3B Instruct, the first model on Metal (MetalBackend.md Phase 4).
#
# Records at FP32, and again at BF16 on the same tokens so the distance HuggingFace's own reduced precision keeps
# from FP32 is on record beside it:
#   - a chat prompt rendered by the model's own template, its FP32 greedy continuation, and the logits at the
#     prompt's last position and at every continued position, the continuation teacher-forced;
#   - the log-probability of each next token of the first 1024 tokens of the wikitext-2 test split.
# Needs a CUDA device with room for the FP32 checkpoint (about 13 GB) and the checkpoint in the HuggingFace cache.
#
#   python hf_llama32_3b_reference.py --output ../../../../Data/models/llama/llama32_3b_instruct_reference.safetensors

import argparse
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = 'meta-llama/Llama-3.2-3B-Instruct'
CHAT = [
    { 'role': 'system', 'content': 'You are a helpful assistant.' },
    { 'role': 'user', 'content': 'Explain in three sentences why the sky is blue.' },
]
# The template writes today's date unless given one; a fixed one keeps the capture reproducible.
DATE = '26 Jul 2024'
GREEDY_TOKENS = 32
SEQUENCE_TOKENS = 1024


def logits_at( model, ids, first ):
    """The logits of every position from `first` on, as FP32 on the host."""
    with torch.no_grad():
        return model( input_ids=ids ).logits[ 0, first : ].to( torch.float32 ).cpu()


def next_token_log_probabilities( model, ids ):
    with torch.no_grad():
        rows = model( input_ids=ids ).logits[ 0, : -1 ].to( torch.float64 )

    return torch.log_softmax( rows, dim=-1 ).gather( 1, ids[ 0, 1 : ].unsqueeze( 1 ) ).squeeze( 1 ).to( torch.float32 ).cpu()


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--corpus', type=Path,
        default=Path( __file__ ).resolve().parents[ 2 ] / 'Quantization' / 'corpus' / 'wiki.test.raw' )
    parser.add_argument( '--output', type=Path, required=True )
    arguments = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained( MODEL )

    prompt = tokenizer.apply_chat_template( CHAT, add_generation_prompt=True, date_string=DATE, return_tensors='pt',
        return_dict=True ).input_ids.cuda()

    # Four characters per token over-reads; the tokenizer adds <|begin_of_text|>.
    text = arguments.corpus.read_text( encoding='utf-8' )[ : SEQUENCE_TOKENS * 6 ]
    sequence = tokenizer( text, return_tensors='pt' ).input_ids[ :, : SEQUENCE_TOKENS ].cuda()

    assert sequence.shape[ 1 ] == SEQUENCE_TOKENS, f'corpus gave {sequence.shape[ 1 ]} tokens'
    assert sequence[ 0, 0 ].item() == tokenizer.bos_token_id

    captured = {}

    for name, dtype in ( ( 'fp32', torch.float32 ), ( 'bf16', torch.bfloat16 ) ):
        model = AutoModelForCausalLM.from_pretrained( MODEL, dtype=dtype, attn_implementation='sdpa' ).cuda().eval()

        if name == 'fp32':
            with torch.no_grad():
                generated = model.generate( prompt, attention_mask=torch.ones_like( prompt ), max_new_tokens=GREEDY_TOKENS,
                    do_sample=False, temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id )

            continuation = generated[ :, prompt.shape[ 1 ] : ]
            tokens = torch.cat( [ prompt, continuation ], dim=1 )

        # The last prompt position, then every continued token but the last, whose successor was never chosen.
        captured[ f'{name}.logits' ] = logits_at( model, tokens[ :, : -1 ], prompt.shape[ 1 ] - 1 )
        captured[ f'{name}.next_token_log_probabilities' ] = next_token_log_probabilities( model, sequence )

        del model
        torch.cuda.empty_cache()

    arguments.output.parent.mkdir( parents=True, exist_ok=True )

    save_file( {
        'prompt': prompt[ 0 ].to( torch.int32 ).cpu(),
        'greedy_tokens': continuation[ 0 ].to( torch.int32 ).cpu(),
        'sequence': sequence[ 0 ].to( torch.int32 ).cpu(),
        **captured,
    }, str( arguments.output ), metadata={
        'model': MODEL,
        'date_string': DATE,
        'transformers': transformers.__version__,
        'torch': torch.__version__,
    } )

    fp32 = captured[ 'fp32.logits' ]
    bf16 = captured[ 'bf16.logits' ]
    relative = ( ( bf16 - fp32 ).norm( dim=1 ) / fp32.norm( dim=1 ) )

    print( f'wrote {arguments.output}' )
    print( f'prompt {prompt.shape[ 1 ]} tokens, greedy: {tokenizer.decode( continuation[ 0 ] )!r}' )
    print( f'HF BF16 against FP32, per step relative L2: worst {relative.max().item():.3e}, mean {relative.mean().item():.3e}' )
    print( f'mean negative log-likelihood over {SEQUENCE_TOKENS} tokens: '
           f'FP32 {-captured[ "fp32.next_token_log_probabilities" ].mean().item():.6f}, '
           f'BF16 {-captured[ "bf16.next_token_log_probabilities" ].mean().item():.6f}' )


if __name__ == '__main__':
    main()
