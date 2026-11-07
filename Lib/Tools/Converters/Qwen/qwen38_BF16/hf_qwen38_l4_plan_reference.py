# HuggingFace reference for Qwen 3.8's four-layer packed build, decoded (Notebooks/MetalQwen.md section 1.2).
#
# The weights are the ones Mila runs under the package's per-role plan: every codebook tensor decoded as stored, and the
# roles the plan holds in FP4 but the package stores in BF16 (the attention layer's projections, the head) quantized to
# FP4 as Mila's load does and decoded back. HuggingFace's Qwen 3.5 text stack, truncated to the package's four layers, runs
# them one layer resident at a time through hf_qwen_layer_stream.py's driver. Records, as Gemma's capture does
# (hf_gemma4_12b_q4_0_reference.py):
#   - a chat prompt rendered by the model's own template at its defaults, the FP32 greedy continuation, and the logits at
#     the prompt's last position and at every continued position, the continuation teacher-forced;
#   - a PG-19 book in a user turn, 2048 tokens: the log-probability of each next token, and the logits at the last position
#     of every 32-token piece, which a prefill in pieces of 32 returns.
# A BF16 run on the same tokens records how far HuggingFace's own reduced precision keeps from FP32 on these weights.
#
#   python hf_qwen38_l4_plan_reference.py --check
#   python hf_qwen38_l4_plan_reference.py --output <Data>/Models/Qwen/qwen38_27b_l4_2p9bit_reference.safetensors
#   python hf_qwen38_l4_plan_reference.py --dtype bfloat16 --tokens-from <the FP32 file> \
#       --output <Data>/Models/Qwen/qwen38_27b_l4_2p9bit_reference_bf16.safetensors

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert( 0, str( Path( __file__ ).resolve().parent ) )
sys.path.insert( 0, str( Path( __file__ ).resolve().parents[ 1 ] ) )
sys.path.insert( 0, str( Path( __file__ ).resolve().parents[ 2 ] ) )
sys.path.insert( 0, str( Path( __file__ ).resolve().parents[ 2 ] / 'Gemma' / 'gemma_4_BF16' ) )

import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from common import apply_transform, expand_qwen_tensor_map
from hf_qwen_layer_stream import StreamedReference, TEXT_PREFIX, ShardedWeights
from hf_long_context_loss import join_wraps


MODEL = 'Qwen/Qwen3.8-27B'
ROOT = Path( __file__ ).resolve().parents[ 5 ]
PACKAGE = ROOT / 'Data' / 'Models' / 'Qwen' / 'qwen38_27b_l4_2p9bit.safetensors'
BOOK = ROOT / 'Data' / 'Datasets' / 'PG19' / 'raw' / 'test' / '30312.txt'
CHAT = [
    { 'role': 'system', 'content': 'You are a helpful assistant.' },
    { 'role': 'user', 'content': 'Explain in three sentences why the sky is blue.' },
]
BOOK_REQUEST = 'Continue this book.'
PIECE_TOKENS = 32
GREEDY_TOKENS = 32
SEQUENCE_TOKENS = 2048
FP4_GROUP = 128
FP4_LEVELS = ( 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 )
FP4_BREAKPOINTS = ( 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0 )


# Rows decoded at once: the head's 248320 rows whole would not fit the card beside their intermediates.
BLOCK_ROWS = 8192


def fp4_round_trip( weight ):
    """FP4 E2M1 per group of 128 as Mila quantizes at load (Tools/Quantization/packing.py, CudaFp4WeightQuantization.cu), decoded back."""
    # On the host: CUDA's torch divides by a scalar through its reciprocal, which moves a scale by its last bit from the one
    # Mila's quantizer and packing.py compute.
    host = weight.cpu()

    return torch.cat( [ fp4_round_trip_block( host[ i : i + BLOCK_ROWS ] ) for i in range( 0, host.shape[ 0 ], BLOCK_ROWS ) ] ).to( weight.device )


def fp4_round_trip_block( weight ):
    rows, columns = weight.shape
    grouped = weight.float().reshape( rows, columns // FP4_GROUP, FP4_GROUP )
    absmax = grouped.abs().amax( dim=2 )
    scales = torch.where( absmax > 0, absmax / 6.0, torch.ones_like( absmax ) )
    normalized = grouped * ( 1.0 / scales ).unsqueeze( -1 )
    breakpoints = torch.tensor( FP4_BREAKPOINTS, device=weight.device )
    magnitude = torch.bucketize( normalized.abs(), breakpoints, right=True )
    levels = torch.tensor( FP4_LEVELS, device=weight.device )[ magnitude ]
    signed = torch.where( normalized < 0, -levels, levels )

    return ( signed * scales.unsqueeze( -1 ) ).reshape( rows, columns )


def decode_codebook( low_plane, high_plane, scales, table, group ):
    """CodebookPacking.ixx: code j's low two bits in byte j / 4 at bit 2 ( j % 4 ), its high bit in byte j / 8 at bit j % 8."""
    return torch.cat( [ decode_codebook_block( low_plane[ i : i + BLOCK_ROWS ], None if high_plane is None else high_plane[ i : i + BLOCK_ROWS ],
        scales[ i : i + BLOCK_ROWS ], table, group ) for i in range( 0, low_plane.shape[ 0 ], BLOCK_ROWS ) ] )


def decode_codebook_block( low_plane, high_plane, scales, table, group ):
    rows = low_plane.shape[ 0 ]
    columns = low_plane.shape[ 1 ] * 4
    low = low_plane.to( torch.int64 )
    codes = torch.stack( [ ( low >> ( 2 * k ) ) & 0x3 for k in range( 4 ) ], dim=-1 ).reshape( rows, columns )

    if high_plane is not None:
        high = high_plane.to( torch.int64 )
        codes = codes | ( torch.stack( [ ( high >> k ) & 0x1 for k in range( 8 ) ], dim=-1 ).reshape( rows, columns ) << 2 )

    values = table.float()[ codes ]

    return ( values.view( rows, -1, group ) * scales.float().unsqueeze( -1 ) ).reshape( rows, columns )


class DecodedPackage:
    """Mila's packed Qwen build read as HuggingFace's text checkpoint: each tensor decoded as Mila runs it, mapped back by the
    converter's own map. Every map transform takes rows, so where a Mila row came from is found by running the transform on
    row numbers."""

    def __init__( self, path, checkpoint, text_config, device, fp4=fp4_round_trip, codebook=decode_codebook ):
        self.file = safe_open( str( path ), framework='pt' )
        self.device = device
        self.fp4 = fp4
        self.codebook = codebook
        metadata = self.file.metadata()

        if metadata.get( 'mila_quantization' ) != 'codebook':
            raise ValueError( f'{path}: weights are {metadata.get( "mila_quantization" )!r}, not the codebook plan' )

        self.policies = json.loads( metadata[ 'mila_codebook_policies' ] )
        self.config = json.loads( metadata[ 'mila_config' ] )
        self.text_config = text_config
        self.head_dim = text_config.head_dim
        self.shapes = ShardedWeights( checkpoint )
        key_dim = text_config.linear_num_key_heads * text_config.linear_key_head_dim
        value_dim = text_config.linear_num_value_heads * text_config.linear_value_head_dim
        self.mappings = expand_qwen_tensor_map( self.config[ 'num_layers' ], text_config.full_attention_interval,
            key_dim, value_dim, prefix=TEXT_PREFIX )
        self.cache = {}

    def _raw( self, name ):
        return self.file.get_tensor( name ).to( self.device )

    def mila_tensor( self, mila ):
        """One Mila tensor as Mila computes with it, FP32."""
        stem = mila.removesuffix( '.weight' )
        policy = self.policies.get( stem, 'NoWeightQuant' )
        stored = self._raw( mila )

        if policy.startswith( 'PerGroupCodebook' ):
            group = int( policy.split( '<' )[ 1 ].rstrip( '>' ) )
            high = self._raw( f'{stem}.weight_high_plane' ) if policy.startswith( 'PerGroupCodebook3' ) else None

            return self.codebook( stored, high, self._raw( f'{stem}.weight_scale' ), self._raw( f'{stem}.weight_codebook' ), group )

        if policy == 'PerGroupFp4<128>':
            if stored.dtype != torch.bfloat16:
                raise ValueError( f'{mila}: FP4 role stored as {stored.dtype}; only BF16 quantized at load is handled here' )

            return self.fp4( stored )

        if policy != 'NoWeightQuant':
            raise ValueError( f'{mila}: policy {policy} is not decoded here' )

        return stored.float()

    def _source_shape( self, name ):
        handle = self.shapes._handle( self.shapes._weight_map[ name ] )

        return handle.get_slice( name ).get_shape()

    def tensor( self, name ):
        """A HuggingFace tensor, assembled from every Mila tensor any of its rows went to."""
        if name in self.cache:
            return self.cache[ name ]

        built = None

        for mapping in self.mappings:
            if name not in mapping.sources:
                continue

            # Each source's rows numbered, the source's index in the high digits, and run through the map's own transform.
            numbered = []

            for index, source in enumerate( mapping.sources ):
                shape = self._source_shape( source )
                rows = torch.arange( shape[ 0 ], dtype=torch.float64 ) + index * 1e7
                numbered.append( rows.view( -1, *( [ 1 ] * ( len( shape ) - 1 ) ) ).expand( *shape ).contiguous() )

            origin = apply_transform( mapping, numbered, self.head_dim )
            origin = origin.reshape( origin.shape[ 0 ], -1 )[ :, 0 ]
            which = torch.div( origin, 1e7, rounding_mode='floor' ).long()
            source_rows = ( origin - which * 1e7 ).long()
            mine = which == mapping.sources.index( name )

            if not bool( mine.any() ):
                continue

            decoded = self.mila_tensor( mapping.mila )

            if built is None:
                built = torch.full( self._source_shape( name ), float( 'nan' ), device=self.device )

            built.view( built.shape[ 0 ], -1 )[ source_rows[ mine ].to( self.device ) ] = \
                decoded.reshape( decoded.shape[ 0 ], -1 )[ mine.nonzero().squeeze( 1 ).to( self.device ) ]

        if built is None:
            raise KeyError( name )

        if torch.isnan( built ).any():
            raise ValueError( f'{name}: some rows map from no Mila tensor' )

        self.cache[ name ] = built

        return built

    def has( self, name ):
        return any( name in mapping.sources for mapping in self.mappings )

    def get( self, name ):
        return self.tensor( name )

    def get_rows( self, name, row_ids ):
        assert name == f'{TEXT_PREFIX}embed_tokens.weight', name
        table = self.file.get_slice( 'temb.wte' )

        return torch.cat( [ table[ int( i ) : int( i ) + 1 ] for i in row_ids ] ).float().to( self.device )

    def state_dict_for( self, prefix ):
        self.cache.clear()
        names = sorted( { source for mapping in self.mappings for source in mapping.sources if source.startswith( prefix ) } )

        return { name[ len( prefix ) : ]: self.tensor( name ) for name in names }


def release( device ):
    gc.collect()

    if str( device ).startswith( 'cuda' ):
        torch.cuda.empty_cache()


@torch.no_grad()
def final_hidden( driver, token_ids ):
    """The final norm's output at every position: the driver's own layer loop (StreamedReference.run), kept whole."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    ids = list( token_ids )
    hidden = driver.weights.get_rows( f'{TEXT_PREFIX}embed_tokens.weight', ids ).to( device=driver.device, dtype=driver.dtype ).view( 1, len( ids ), -1 )
    position_ids = torch.arange( len( ids ), device=driver.device ).view( 1, 1, -1 ).expand( 4, 1, -1 )
    text_position_ids = position_ids[ 0 ]
    causal_mask, linear_mask = driver._masks( hidden, text_position_ids )
    position_embeddings = driver.rotary( hidden, position_ids[ 1: ] )

    for i in range( driver.config.num_hidden_layers ):
        kind = driver.config.layer_types[ i ]
        layer = driver._build_layer( i )
        hidden = layer( hidden, position_embeddings=position_embeddings,
            attention_mask=linear_mask if kind == 'linear_attention' else causal_mask,
            position_ids=text_position_ids, past_key_values=None )
        del layer
        release( driver.device )

    with torch.device( 'meta' ):
        norm = Qwen3_5RMSNorm( driver.config.hidden_size, eps=driver.config.rms_norm_eps )

    norm.load_state_dict( { 'weight': driver.weights.get( f'{TEXT_PREFIX}norm.weight' ).to( driver.device, driver.dtype ) }, assign=True )

    return norm( hidden )[ 0 ]


@torch.no_grad()
def project( normed, head, rows=256 ):
    return torch.cat( [ normed[ i : i + rows ] @ head.T for i in range( 0, normed.shape[ 0 ], rows ) ] )


@torch.no_grad()
def next_token_log_probabilities( normed, head, ids, rows=256 ):
    out = []

    for i in range( 0, normed.shape[ 0 ] - 1, rows ):
        block = ( normed[ i : min( i + rows, normed.shape[ 0 ] - 1 ) ] @ head.T ).double()
        targets = ids[ i + 1 : i + 1 + block.shape[ 0 ] ].unsqueeze( 1 )
        out.append( torch.log_softmax( block, dim=-1 ).gather( 1, targets ).squeeze( 1 ).float().cpu() )

    return torch.cat( out )


def text_config_for( checkpoint, layers ):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config_json = json.loads( ( checkpoint / 'config.json' ).read_text( encoding='utf-8' ) )[ 'text_config' ]
    config_json[ 'num_hidden_layers' ] = layers
    config_json[ 'layer_types' ] = config_json[ 'layer_types' ][ : layers ]
    config = Qwen3_5TextConfig( **config_json )
    config._attn_implementation = 'eager'

    return config


def check( checkpoint, config, device ):
    """Each part of the reading against what it must equal exactly, and the negative control that must miss.

    Codebook tensors against Tools/Quantization/packing.py, the layout's normative Python codec; the FP4 round trip against its
    quantize and dequantize; the map back against the BF16 checkpoint on the tensors the plan leaves unquantized. The decoded
    weights' distance from the BF16 checkpoint is reported, not judged: it is the format's own rounding.
    """
    import numpy as np

    sys.path.insert( 0, str( ROOT / 'Lib' / 'Tools' / 'Quantization' ) )
    import packing

    original = ShardedWeights( checkpoint )
    package = DecodedPackage( PACKAGE, checkpoint, config, device )
    failures = 0

    def report( bad, text ):
        nonlocal failures
        failures += bool( bad )
        print( f'  {"FAIL" if bad else "OK  "} {text}' )

    for stem in ( 'tf_layer_0.fc_in_proj_qk', 'tf_layer_0.fc_out_proj', 'tf_layer_3.fc_down' ):
        low = package.file.get_tensor( f'{stem}.weight' ).numpy()
        columns = low.shape[ 1 ] * 4
        three_bit = package.policies[ stem ].startswith( 'PerGroupCodebook3' )
        high = package.file.get_tensor( f'{stem}.weight_high_plane' ).numpy() if three_bit else None
        rows = slice( 0, 64 )
        codes = packing.unpack_three_bit_codes( low[ rows ], high[ rows ], columns ) if three_bit else packing.unpack_two_bit_codes( low[ rows ], columns )
        scale = package.file.get_tensor( f'{stem}.weight_scale' ).numpy().astype( np.float32 )[ rows ]
        table = package.file.get_tensor( f'{stem}.weight_codebook' ).numpy().astype( np.float32 )
        want = table[ codes ] * np.repeat( scale, columns // scale.shape[ 1 ], axis=1 )
        got = package.mila_tensor( f'{stem}.weight' )[ rows ].cpu().numpy()
        report( not np.array_equal( got, want ), f"{stem:<28} decode equals packing.py's" )

    q = original.get( TEXT_PREFIX + 'layers.3.self_attn.q_proj.weight' )[ : 64 ].float()
    packed, scales = packing.quantize_fp4_e2m1( q.numpy() )
    want = packing.dequantize_fp4_e2m1( packed, scales, q.shape[ 1 ] )
    report( not np.array_equal( fp4_round_trip( q.to( device ) ).cpu().numpy(), want ), "FP4 round trip equals packing.py's quantize and dequantize" )

    for key in ( 'layers.0.linear_attn.in_proj_a.weight', 'layers.0.linear_attn.conv1d.weight', 'layers.0.linear_attn.A_log',
                 'layers.0.linear_attn.norm.weight', 'layers.3.self_attn.q_norm.weight', 'norm.weight' ):
        want = original.get( TEXT_PREFIX + key ).to( device ).float()
        report( not torch.equal( package.tensor( TEXT_PREFIX + key ), want ), f'{key:<40} mapped back equals the checkpoint' )

    for key in ( 'layers.0.linear_attn.in_proj_qkv.weight', 'layers.0.linear_attn.out_proj.weight', 'layers.0.mlp.down_proj.weight',
                 'layers.3.self_attn.q_proj.weight', 'layers.3.mlp.gate_proj.weight' ):
        want = original.get( TEXT_PREFIX + key ).to( device ).float()
        relative = ( ( package.tensor( TEXT_PREFIX + key ) - want ).norm() / want.norm() ).item()
        print( f'  ---- {key:<40} decoded weights {relative:.3e} from BF16, relative L2' )

    def no_high_plane( low, high, scales, table, group ):
        return decode_codebook( low, None, scales, table, group )

    dropped = DecodedPackage( PACKAGE, checkpoint, config, device, codebook=no_high_plane )
    decoded = package.mila_tensor( 'tf_layer_0.fc_in_proj_qk.weight' )
    relative = ( ( dropped.mila_tensor( 'tf_layer_0.fc_in_proj_qk.weight' ) - decoded ).norm() / decoded.norm() ).item()
    report( relative < 0.1, f'high plane dropped moves tf_layer_0.fc_in_proj_qk by {relative:.3e}, and must' )

    print( f'\nCheck: {"PASSED" if failures == 0 else f"FAILED ({failures})"}' )

    return 0 if failures == 0 else 1


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--book', type=Path, default=BOOK )
    parser.add_argument( '--output', type=Path )
    parser.add_argument( '--check', action='store_true', help='Check the decode against the BF16 checkpoint, then exit' )
    parser.add_argument( '--device', default='cuda' )
    parser.add_argument( '--dtype', default='float32', choices=( 'float32', 'bfloat16' ) )
    parser.add_argument( '--tokens-from', type=Path,
        help='A BF16 run teacher-forces the FP32 capture\'s prompt and greedy tokens, read from this file' )
    arguments = parser.parse_args()

    checkpoint = Path( snapshot_download( MODEL, allow_patterns=[ '*.json', '*.jinja', '*.txt' ], local_files_only=True ) )
    layers = json.loads( safe_open( str( PACKAGE ), framework='pt' ).metadata()[ 'mila_config' ] )[ 'num_layers' ]
    config = text_config_for( checkpoint, layers )

    if arguments.check:
        sys.exit( check( checkpoint, config, arguments.device ) )

    if not arguments.output:
        parser.error( '--output is required unless --check is given' )

    tokenizer = transformers.AutoTokenizer.from_pretrained( checkpoint )
    prompt = tokenizer.apply_chat_template( CHAT, add_generation_prompt=True, return_dict=True )[ 'input_ids' ]
    print( f'prompt {len( prompt )} tokens: {tokenizer.decode( prompt )!r}', flush=True )

    # The book after a user turn asking for it, in the model's turn as the template opens it, with the line wraps joined.
    with open( arguments.book, 'rb' ) as f:
        text = join_wraps( f.read( SEQUENCE_TOKENS * 8 ).decode( 'utf-8', errors='ignore' ) )

    turn = tokenizer.apply_chat_template( [ { 'role': 'user', 'content': BOOK_REQUEST } ], add_generation_prompt=True, tokenize=False )
    sequence = tokenizer( turn + text, add_special_tokens=False )[ 'input_ids' ][ : SEQUENCE_TOKENS ]

    assert len( sequence ) == SEQUENCE_TOKENS, f'corpus gave {len( sequence )} tokens'

    dtype = torch.float32 if arguments.dtype == 'float32' else torch.bfloat16
    name = 'fp32' if arguments.dtype == 'float32' else 'bf16'
    weights = DecodedPackage( PACKAGE, checkpoint, config, arguments.device )
    driver = StreamedReference( config, weights, arguments.device, dtype )
    started = time.time()

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

    head = weights.get( 'lm_head.weight' ).to( dtype )
    tokens = prompt + continuation
    normed = final_hidden( driver, tokens[ : -1 ] )
    logits = project( normed[ len( prompt ) - 1 : ], head ).float().cpu()
    del normed
    release( arguments.device )
    print( f'teacher-forced logits {tuple( logits.shape )}  {time.time() - started:.0f} s', flush=True )

    sequence_ids = torch.tensor( sequence, device=arguments.device )
    normed = final_hidden( driver, sequence )
    log_probabilities = next_token_log_probabilities( normed, head, sequence_ids )
    pieces = project( normed[ PIECE_TOKENS - 1 :: PIECE_TOKENS ], head ).float().cpu()
    print( f'sequence {SEQUENCE_TOKENS}: mean negative log-likelihood {-log_probabilities.mean().item():.6f}  '
           f'{time.time() - started:.0f} s', flush=True )

    arguments.output.parent.mkdir( parents=True, exist_ok=True )
    save_file( {
        'prompt': torch.tensor( prompt, dtype=torch.int32 ),
        'greedy_tokens': torch.tensor( continuation, dtype=torch.int32 ),
        'sequence': torch.tensor( sequence, dtype=torch.int32 ),
        f'{name}.logits': logits.contiguous(),
        f'{name}.next_token_log_probabilities': log_probabilities.contiguous(),
        f'{name}.piece_logits': pieces.contiguous(),
    }, str( arguments.output ), metadata={
        'model': MODEL,
        'weights': PACKAGE.name,
        'layers': str( layers ),
        'book': arguments.book.name,
        'piece_tokens': str( PIECE_TOKENS ),
        'transformers': transformers.__version__,
        'torch': torch.__version__,
    } )

    print( f'wrote {arguments.output}' )
    print( f'greedy: {tokenizer.decode( continuation )!r}' )


if __name__ == '__main__':
    main()
