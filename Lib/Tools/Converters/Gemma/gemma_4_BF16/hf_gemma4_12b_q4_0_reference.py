# HuggingFace reference for Gemma 4 12B on Mila's Q4_0 package, decoded (Notebooks/MetalGemma.md section 1).
#
# The weights are the package's own: every Q4_0 and six-bit tensor decoded to FP32 on the device, so the reference is
# the exact model the format defines, and HuggingFace's Gemma 4 runs it at FP32. The FP32 model is about 48 GB, past the
# PC's memory, so it runs one decoder layer at a time through hf_gemma_layer_stream.py's driver. Records:
#   - a chat prompt rendered by the model's own template at its defaults, the FP32 greedy continuation, and the logits
#     at the prompt's last position and at every continued position, the continuation teacher-forced;
#   - a PG-19 book as the G2 harnesses give it to an instruct model (hf_long_context_loss.py), 2048 tokens, past the
#     sliding window of 1024: the log-probability of each next token, the logits at the last position of every
#     32-token piece, which a prefill in pieces of 32 returns, and the logits either side of the window's edge.
# Raw wikitext cannot judge a precision change on this model (ModelFamilyParity.md 8.2, G1), so the book replaces it.
# Logits are before the final softcap; log-probabilities are after it, as HuggingFace's are. A BF16 run on the same
# tokens records how far HuggingFace's own reduced precision keeps from FP32 on these weights.
#
#   python hf_gemma4_12b_q4_0_reference.py --check
#   python hf_gemma4_12b_q4_0_reference.py --output <Data>/Models/Gemma/gemma4_12b_it_q4_0_reference.safetensors
#   python hf_gemma4_12b_q4_0_reference.py --dtype bfloat16 --tokens-from <the FP32 file> \
#       --output <Data>/Models/Gemma/gemma4_12b_it_q4_0_reference_bf16.safetensors

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert( 0, str( Path( __file__ ).resolve().parent ) )
sys.path.insert( 0, str( Path( __file__ ).resolve().parent.parent / 'gemma_4_26b_moe' ) )
sys.path.insert( 0, str( Path( __file__ ).resolve().parent.parent.parent ) )

import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from hf_gemma_layer_stream import StreamedReference, _text_config
from hf_long_context_loss import join_wraps


MODEL = 'google/gemma-4-12B-it-qat-q4_0-unquantized'
DATA = Path( __file__ ).resolve().parents[ 5 ] / 'Data' / 'Models' / 'Gemma'
PACKAGE = DATA / 'gemma4_12b_it_qat_q4_0.safetensors'
BOOK = Path( __file__ ).resolve().parents[ 5 ] / 'Data' / 'Datasets' / 'PG19' / 'raw' / 'test' / '30312.txt'
BOOK_TURN = '<|turn>user\nContinue this book.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'
# The last position of every piece of a prefill in pieces of this many tokens.
PIECE_TOKENS = 32
CHAT = [
    { 'role': 'system', 'content': 'You are a helpful assistant.' },
    { 'role': 'user', 'content': 'Explain in three sentences why the sky is blue.' },
]
GREEDY_TOKENS = 32
SEQUENCE_TOKENS = 2048
# Rows of the sequence's logits kept around the sliding window's edge at 1024, for a decode across it.
WINDOW_EDGE_FIRST = 992
WINDOW_EDGE_ROWS = 64
GROUP = 32


def decode_int4( codes, scales ):
    """Int4Packing: code j's four bits in byte j / 2, the low nibble for even j; value = ( code - 8 ) * scale."""
    low = ( codes & 0x0F ).to( torch.int16 )
    high = ( codes >> 4 ).to( torch.int16 )
    steps = torch.stack( ( low, high ), dim=-1 ).reshape( codes.shape[ 0 ], -1 ) - 8

    return ( steps.to( torch.float32 ).view( codes.shape[ 0 ], -1, GROUP ) * scales.to( torch.float32 ).unsqueeze( -1 ) ).reshape(
        codes.shape[ 0 ], -1 )


def decode_int6( codes, scales ):
    """Int6Packing: low nibbles as Int4's, then code j's high two bits in byte K / 2 + j / 4 at bit 2 ( j % 4 ); value = ( code - 32 ) * scale."""
    columns = codes.shape[ 1 ] * 4 // 3
    nibbles = codes[ :, : columns // 2 ]
    tops = codes[ :, columns // 2 : ].to( torch.int16 )
    low = torch.stack( ( nibbles & 0x0F, nibbles >> 4 ), dim=-1 ).reshape( codes.shape[ 0 ], columns ).to( torch.int16 )
    high = torch.stack( [ ( tops >> ( 2 * k ) ) & 0x3 for k in range( 4 ) ], dim=-1 ).reshape( codes.shape[ 0 ], columns )
    steps = ( low | ( high << 4 ) ) - 32

    return ( steps.to( torch.float32 ).view( codes.shape[ 0 ], -1, GROUP ) * scales.to( torch.float32 ).unsqueeze( -1 ) ).reshape(
        codes.shape[ 0 ], -1 )


class DecodedPackage:
    """Mila's Q4_0 package, read as HuggingFace's text checkpoint: Mila's names and fused tensors mapped back, codes decoded."""

    def __init__( self, path, config, device, int4=decode_int4, int6=decode_int6 ):
        self.file = safe_open( str( path ), framework='pt' )
        self.config = config
        self.device = device
        self.int4 = int4
        self.int6 = int6
        quantization = self.file.metadata().get( 'mila_quantization' )

        if quantization != 'q4_0':
            raise ValueError( f'{path}: weights are {quantization!r}, not q4_0' )

    def _raw( self, name ):
        return self.file.get_tensor( name ).to( self.device )

    def _int4( self, name ):
        return self.int4( self._raw( name ), self._raw( f'{name}_scale' ) )

    def table_rows( self, row_ids ):
        codes = self.file.get_slice( 'temb.wte' )
        scales = self.file.get_slice( 'temb.wte_scale' )
        rows = [ self.int6( codes[ i : i + 1 ].to( self.device ), scales[ i : i + 1 ].to( self.device ) ) for i in map( int, row_ids ) ]

        return torch.cat( rows )

    def tensor( self, name ):
        if name == 'model.embed_tokens.weight':
            return self.int6( self._raw( 'temb.wte' ), self._raw( 'temb.wte_scale' ) )

        if name == 'model.norm.weight':
            return self._raw( 'rmsn_final.weight' ).float()

        raise KeyError( name )

    def rows( self, name, row_ids ):
        assert name == 'model.embed_tokens.weight', name

        return self.table_rows( row_ids )

    def state_dict_for( self, prefix ):
        index = int( prefix.split( '.' )[ 2 ] )
        mila = f'tf_layer_{index}'
        config = self.config
        is_global = config.layer_types[ index ] == 'full_attention'
        key_equals_value = is_global and config.attention_k_eq_v
        head_dim = config.global_head_dim if is_global else config.head_dim
        kv_heads = config.num_global_key_value_heads if key_equals_value else config.num_key_value_heads

        qkv = self._int4( f'{mila}.qkv_proj.weight' )
        q_rows = config.num_attention_heads * head_dim
        kv_rows = kv_heads * head_dim
        expected = q_rows + kv_rows * ( 1 if key_equals_value else 2 )

        if qkv.shape[ 0 ] != expected:
            raise ValueError( f'{mila}.qkv_proj: {qkv.shape[ 0 ]} rows, the config gives {expected}' )

        gate_up = self._int4( f'{mila}.ffn.mlp.fc_gate_up.weight' )
        hidden = config.intermediate_size

        state = {
            'input_layernorm.weight': self._raw( f'{mila}.input_norm.weight' ).float(),
            'self_attn.q_proj.weight': qkv[ : q_rows ],
            'self_attn.k_proj.weight': qkv[ q_rows : q_rows + kv_rows ],
            'self_attn.q_norm.weight': self._raw( f'{mila}.q_norm.weight' ).float(),
            'self_attn.k_norm.weight': self._raw( f'{mila}.k_norm.weight' ).float(),
            'self_attn.o_proj.weight': self._int4( f'{mila}.o_proj.weight' ),
            'post_attention_layernorm.weight': self._raw( f'{mila}.post_attn_norm.weight' ).float(),
            'pre_feedforward_layernorm.weight': self._raw( f'{mila}.ffn.pre_norm.weight' ).float(),
            'mlp.gate_proj.weight': gate_up[ : hidden ],
            'mlp.up_proj.weight': gate_up[ hidden : ],
            'mlp.down_proj.weight': self._int4( f'{mila}.ffn.mlp.fc_down.weight' ),
            'post_feedforward_layernorm.weight': self._raw( f'{mila}.ffn.post_norm.weight' ).float(),
            'layer_scalar': self._raw( f'{mila}.layer_scalar' ).float(),
        }

        if not key_equals_value:
            state[ 'self_attn.v_proj.weight' ] = qkv[ q_rows + kv_rows : ]

        # Mila writes a unit weight for v_norm, which HuggingFace's has none of; anything else would be a different model.
        v_norm = self._raw( f'{mila}.v_norm.weight' ).float()

        if not torch.all( v_norm == 1 ):
            raise ValueError( f'{mila}.v_norm.weight is not a unit weight' )

        return state


def release( device ):
    gc.collect()

    if str( device ).startswith( 'cuda' ):
        torch.cuda.empty_cache()


@torch.no_grad()
def final_hidden( driver, token_ids ):
    """The final norm's output at every position: the driver's own layer loop, kept whole rather than at the last position."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    ids = list( token_ids )
    embedded = driver.weights.rows( 'model.embed_tokens.weight', ids ).to( device=driver.device, dtype=driver.dtype )
    scale = torch.tensor( driver.config.hidden_size ** 0.5 ).to( device=driver.device, dtype=driver.dtype )
    hidden = ( embedded * scale ).view( 1, len( ids ), -1 )
    position_ids = torch.arange( len( ids ), device=driver.device ).unsqueeze( 0 )
    masks = driver._masks( hidden, position_ids )
    position_embeddings = { kind: driver.rotary( hidden, position_ids, kind ) for kind in set( driver.config.layer_types ) }

    for i in range( driver.config.num_hidden_layers ):
        kind = driver.config.layer_types[ i ]
        layer = driver._build_layer( i )
        hidden = layer( hidden, None, shared_kv_states={}, position_embeddings=position_embeddings[ kind ],
            attention_mask=masks[ kind ], position_ids=position_ids, past_key_values=None )
        del layer
        release( driver.device )

    with torch.device( 'meta' ):
        norm = Gemma4RMSNorm( driver.config.hidden_size, eps=driver.config.rms_norm_eps )

    norm.load_state_dict( { 'weight': driver.weights.tensor( 'model.norm.weight' ).to( driver.device, driver.dtype ) }, assign=True )

    return norm( hidden )[ 0 ]


def softcapped( logits, cap ):
    return torch.tanh( logits / cap ) * cap


@torch.no_grad()
def project( normed, table, rows=256 ):
    """Logits before the softcap, a block of positions at a time, on the device."""
    return torch.cat( [ ( normed[ i : i + rows ] @ table.T ) for i in range( 0, normed.shape[ 0 ], rows ) ] )


@torch.no_grad()
def next_token_log_probabilities( normed, table, ids, cap, rows=256 ):
    out = []

    for i in range( 0, normed.shape[ 0 ] - 1, rows ):
        block = softcapped( normed[ i : min( i + rows, normed.shape[ 0 ] - 1 ) ] @ table.T, cap ).double()
        targets = ids[ i + 1 : i + 1 + block.shape[ 0 ] ].unsqueeze( 1 )
        out.append( torch.log_softmax( block, dim=-1 ).gather( 1, targets ).squeeze( 1 ).float().cpu() )

    return torch.cat( out )


def check( config, device ):
    """The decoder against the unquantized QAT weights the package was quantized from, and against itself with the nibbles swapped."""
    root = Path( snapshot_download( MODEL, allow_patterns=[ 'model.safetensors' ], local_files_only=True ) )
    unquantized = safe_open( str( root / 'model.safetensors' ), framework='pt' )
    prefix = next( n for n in unquantized.keys() if n.endswith( 'embed_tokens.weight' ) )[ : -len( 'embed_tokens.weight' ) ]

    def swapped( codes, scales ):
        return decode_int4( ( codes << 4 ) | ( codes >> 4 ), scales )

    failures = 0

    for name, package in ( ( 'decoded', DecodedPackage( PACKAGE, config, device ) ),
                           ( 'nibbles swapped', DecodedPackage( PACKAGE, config, device, int4=swapped ) ) ):
        for index in ( 0, 5 ):
            state = package.state_dict_for( f'model.layers.{index}.' )

            for key in ( 'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight' ):
                want = unquantized.get_tensor( f'{prefix}layers.{index}.{key}' ).to( device ).float()
                relative = ( ( state[ key ] - want ).norm() / want.norm() ).item()
                bad = ( relative > 0.1 ) if name == 'decoded' else ( relative < 0.5 )
                failures += bad
                print( f'  {"FAIL" if bad else "OK  "} {name:<16} layer {index} {key:<26} relative L2 {relative:.3e}' )

            release( device )

    table = DecodedPackage( PACKAGE, config, device ).table_rows( range( 0, 262144, 4099 ) )
    want = unquantized.get_tensor( f'{prefix}embed_tokens.weight' )[ 0 : 262144 : 4099 ].to( device ).float()
    relative = ( ( table - want ).norm() / want.norm() ).item()
    failures += relative > 0.05
    print( f'  {"FAIL" if relative > 0.05 else "OK  "} decoded          table rows               relative L2 {relative:.3e}' )
    print( f'\nCheck: {"PASSED" if failures == 0 else f"FAILED ({failures})"}' )

    return 0 if failures == 0 else 1


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--book', type=Path, default=BOOK )
    parser.add_argument( '--output', type=Path )
    parser.add_argument( '--check', action='store_true', help='Check the decoder against the unquantized weights, then exit' )
    parser.add_argument( '--device', default='cuda' )
    parser.add_argument( '--dtype', default='float32', choices=( 'float32', 'bfloat16' ) )
    parser.add_argument( '--tokens-from', type=Path,
        help='A BF16 run teacher-forces the FP32 capture\'s prompt and greedy tokens, read from this file' )
    arguments = parser.parse_args()

    root = Path( snapshot_download( MODEL, allow_patterns=[ '*.json', '*.jinja' ], local_files_only=True ) )
    config = _text_config( json.loads( ( root / 'config.json' ).read_text( encoding='utf-8' ) ) )

    if arguments.check:
        sys.exit( check( config, arguments.device ) )

    if not arguments.output:
        parser.error( '--output is required unless --check is given' )

    tokenizer = transformers.AutoTokenizer.from_pretrained( root )
    prompt = tokenizer.apply_chat_template( CHAT, add_generation_prompt=True, return_dict=True )[ 'input_ids' ]
    print( f'prompt {len( prompt )} tokens: {tokenizer.decode( prompt )!r}', flush=True )

    # <bos>, which Gemma's tokenizer does not add, then the book in a model turn with its line wraps joined.
    with open( arguments.book, 'rb' ) as f:
        text = join_wraps( f.read( SEQUENCE_TOKENS * 8 ).decode( 'utf-8', errors='ignore' ) )

    sequence = ( [ tokenizer.bos_token_id ] + tokenizer( BOOK_TURN + text, add_special_tokens=False )[ 'input_ids' ] )[ : SEQUENCE_TOKENS ]

    assert len( sequence ) == SEQUENCE_TOKENS, f'corpus gave {len( sequence )} tokens'

    dtype = torch.float32 if arguments.dtype == 'float32' else torch.bfloat16
    name = 'fp32' if arguments.dtype == 'float32' else 'bf16'
    weights = DecodedPackage( PACKAGE, config, arguments.device )
    driver = StreamedReference( config, weights, 'model.', arguments.device, dtype )
    cap = config.final_logit_softcapping
    started = time.time()

    # Greedy by re-running the whole prefix per token, one layer resident; the softcap does not move an argmax.
    ids = list( prompt )
    continuation = []

    if name == 'bf16':
        if not arguments.tokens_from:
            parser.error( '--dtype bfloat16 needs --tokens-from, the FP32 capture' )

        with safe_open( str( arguments.tokens_from ), framework='pt' ) as captured:
            if captured.get_tensor( 'prompt' ).tolist() != prompt or captured.get_tensor( 'sequence' ).tolist() != sequence:
                raise ValueError( f'{arguments.tokens_from}: its prompt or sequence is not this run\'s' )

            continuation = captured.get_tensor( 'greedy_tokens' ).tolist()

    for step in range( GREEDY_TOKENS if name == 'fp32' else 0 ):
        logits = driver.run( ids, config.num_hidden_layers, progress=False )[ 'logits' ]
        top = torch.topk( logits, 2 )
        continuation.append( int( top.indices[ 0 ] ) )
        ids.append( continuation[ -1 ] )
        print( f'step {step}: {continuation[ -1 ]} {tokenizer.decode( continuation[ -1 : ] )!r} '
               f'margin {float( top.values[ 0 ] - top.values[ 1 ] ):.4f}  {time.time() - started:.0f} s', flush=True )

        if continuation[ -1 ] in ( 1, 106 ):
            break

    table = weights.tensor( 'model.embed_tokens.weight' ).to( dtype )
    tokens = prompt + continuation
    normed = final_hidden( driver, tokens[ : -1 ] )
    logits = project( normed[ len( prompt ) - 1 : ], table ).float().cpu()
    del normed
    release( arguments.device )
    print( f'teacher-forced logits {tuple( logits.shape )}  {time.time() - started:.0f} s', flush=True )

    sequence_ids = torch.tensor( sequence, device=arguments.device )
    normed = final_hidden( driver, sequence )
    log_probabilities = next_token_log_probabilities( normed, table, sequence_ids, cap )
    edge = project( normed[ WINDOW_EDGE_FIRST : WINDOW_EDGE_FIRST + WINDOW_EDGE_ROWS ], table ).float().cpu()
    pieces = project( normed[ PIECE_TOKENS - 1 :: PIECE_TOKENS ], table ).float().cpu()
    print( f'sequence {SEQUENCE_TOKENS}: mean negative log-likelihood {-log_probabilities.mean().item():.6f}  '
           f'{time.time() - started:.0f} s', flush=True )

    arguments.output.parent.mkdir( parents=True, exist_ok=True )
    save_file( {
        'prompt': torch.tensor( prompt, dtype=torch.int32 ),
        'greedy_tokens': torch.tensor( continuation, dtype=torch.int32 ),
        'sequence': torch.tensor( sequence, dtype=torch.int32 ),
        f'{name}.logits': logits.contiguous(),
        f'{name}.next_token_log_probabilities': log_probabilities.contiguous(),
        f'{name}.window_edge_logits': edge.contiguous(),
        f'{name}.piece_logits': pieces.contiguous(),
    }, str( arguments.output ), metadata={
        'model': MODEL,
        'weights': PACKAGE.name,
        'book': arguments.book.name,
        'piece_tokens': str( PIECE_TOKENS ),
        'window_edge_first': str( WINDOW_EDGE_FIRST ),
        'final_logit_softcapping': str( cap ),
        'transformers': transformers.__version__,
        'torch': torch.__version__,
    } )

    print( f'wrote {arguments.output}' )
    print( f'greedy: {tokenizer.decode( continuation )!r}' )


if __name__ == '__main__':
    main()
