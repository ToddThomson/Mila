# HuggingFace reference for the Qwen 4 components and network (Qwen4.md 8.1, Phase 0).
#
# Builds a tiny Qwen4ExpForCausalLM from transformers' qwen4_exp and draws every weight at random -- every
# norm away from its identity, the PLE convolution away from the zeros the reference initializes it to -- so
# a misplaced norm or a skipped convolution cannot hide. The model is saved as a HuggingFace checkpoint and
# converted with Qwen4/convert_weights.py at FP32 and BF16. No network, no GPU, no 27B checkpoint.
#
# The geometry exercises what a real checkpoint at short context would hide:
#   - indexer budget 8 at compress ratio 4, so QSA selects 2 of up to 8 blocks inside the 32-token capture;
#   - an n-gram base near 1,000, so hash collisions and the floor-mod are exercised;
#   - PLE on 1-based layer 2 (index 1), not layer 0; EOS tokens inside the sequence, one ending a chunk;
#   - full_attention_interval 4, so both mixer kinds appear;
#   - 16 n-gram shards, so a lexical shard order differs from the numeric one (shard_10 sorts before shard_2);
#   - a 64-wide indexer head under a 32-dimension rotary width, so the indexer's partial rotation shows.
#
# Two variants, so either 27B is covered (Qwen4.md 3.5). --variant moe is the reference as written: a routed
# MoE with a gated shared expert on every layer. --variant dense substitutes the reference's own SwiGLU MLP
# for every MoE block -- the shape a dense 27B most plausibly has, since qwen4_exp has no dense layer.
#
# One capture holds three runs of the same weights:
#   chunked  the prompt prefilled in two chunks through the cache, then one-token decode steps. Every
#            component's input and output is recorded per step: n-gram ids, PLE, each gated residual,
#            each mixer, the indexer's selected sets, the MLP, each layer's stream, the final mixer, logits.
#   full     the whole sequence in one uncached pass: logits at every position, and the chunked run's
#            agreement with it, which is the reference checking its own cache.
#   dense    the whole sequence with the indexer's budget lifted past the sequence length: dense causal
#            attention over the same weights, the Phase 4 gate's reference (Qwen4.md 8.3).
#
# Beside the capture, each PLE layer's assembled n-gram table (`weights.layer<i>.ple.ngram_table`) and a text file
# of its int64 hash constants, which no tensor dtype Mila's tests read can hold.
#
#   python hf_qwen4_tiny_reference.py --variant moe --output-dir ../../../../Data/Models/Qwen4/qwen4_tiny_moe
#   python hf_qwen4_tiny_reference.py --variant dense --output-dir ../../../../Data/Models/Qwen4/qwen4_tiny_dense

import argparse
import sys
from pathlib import Path

sys.path.insert( 0, str( Path( __file__ ).resolve().parent ) )

import torch
import transformers
from safetensors.torch import save_file
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM, Qwen4ExpTextMLP

from convert_weights import convert_qwen4


# The prompt prefills as two chunks; the first ends on an EOS, so the n-gram history crosses a chunk
# boundary at a segment boundary.
PREFILL_CHUNKS = ( 13, 11 )
DECODE_TOKENS = 8
EOS_TOKEN_ID = 5
EOS_POSITIONS = ( 9, 12, 27 )

DENSE_INTERMEDIATE_SIZE = 512


def tiny_config() -> Qwen4ExpTextConfig:
    config = Qwen4ExpTextConfig(
        vocab_size=256,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        hidden_act='silu',
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_parameters={
            'rope_type': 'default',
            'rope_theta': 10000000.0,
            'partial_rotary_factor': 0.25,
            'mrope_section': [ 6, 5, 5 ],
            'mrope_interleaved': True,
        },
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        output_gate_type='sigmoid',
        hc_count=4,
        hc_lowrank=32,
        ple_layer_ids=[ 2 ],
        ple_embed_dim=256,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=16,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=64,
        indexer_budget=8,
        indexer_compress_ratio=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        norm_topk_prob=True,
        eos_token_id=EOS_TOKEN_ID )

    config._attn_implementation = 'eager'

    return config


def substitute_dense_feedforward( model: Qwen4ExpForCausalLM, config: Qwen4ExpTextConfig ):
    for layer in model.model.layers:
        layer.mlp = Qwen4ExpTextMLP( config, intermediate_size=DENSE_INTERMEDIATE_SIZE )


def randomize( model: torch.nn.Module, seed: int ):
    """Every floating tensor drawn; the int64 n-gram constants are left as the reference derives them.

    Qwen's norms scale by (1 + weight), so their weights are drawn around zero. The DeltaNet's gated norm
    scales by its raw weight, so it is drawn around one. A_log keeps the decay in the range the reference
    initializes it to.
    """
    generator = torch.Generator().manual_seed( seed )

    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if not tensor.is_floating_point():
                continue

            if name.endswith( 'linear_attn.norm.weight' ):
                tensor.copy_( 0.5 + torch.rand( tensor.shape, generator=generator ) )
            elif 'norm' in name and name.endswith( 'weight' ):
                tensor.copy_( torch.rand( tensor.shape, generator=generator ) - 0.5 )
            elif name.endswith( 'A_log' ):
                tensor.copy_( ( 1.0 + 15.0 * torch.rand( tensor.shape, generator=generator ) ).log() )
            elif name.endswith( 'conv1d.weight' ):
                tensor.copy_( torch.randn( tensor.shape, generator=generator ) * 0.3 )
            else:
                tensor.copy_( torch.randn( tensor.shape, generator=generator ) * 0.05 )


def sequence_ids( total: int, vocab_size: int ):
    ids = [ ( i * 37 + 11 ) % vocab_size for i in range( total ) ]

    for position in EOS_POSITIONS:
        ids[ position ] = EOS_TOKEN_ID

    # A stray EOS from the formula would move a segment boundary the capture does not name.
    for position, token in enumerate( ids ):
        if token == EOS_TOKEN_ID and position not in EOS_POSITIONS:
            raise ValueError( f'sequence formula produced EOS at position {position}' )

    return ids


class Capture:
    """Records named tensors per step, batch axis dropped, floats as FP32 and ids as int32."""

    def __init__( self ):
        self.tensors = {}
        self.step = None

    def put( self, name: str, tensor: torch.Tensor ):
        key = f'{self.step}.{name}'

        if key in self.tensors:
            raise KeyError( f'{key} recorded twice' )

        value = tensor.detach()[ 0 ]

        if value.is_floating_point():
            value = value.to( torch.float32 )
        else:
            value = value.to( torch.int32 )

        self.tensors[ key ] = value.contiguous().clone()


def selected_mask( mask: torch.Tensor ) -> torch.Tensor:
    """The indexer's additive mask as 1 where a key is selected: [batch, query, key]."""
    selected = mask == 0 if mask.is_floating_point() else mask

    return selected[ :, 0 ].to( torch.int32 )


def register_hooks( model: Qwen4ExpForCausalLM, capture: Capture ):
    handles = []
    text_model = model.model

    def output_hook( name ):
        return lambda module, inputs, output: capture.put( name, output )

    handles.append( text_model.embed_tokens.register_forward_hook( output_hook( 'embedding' ) ) )

    for index, layer in enumerate( text_model.layers ):
        prefix = f'layer{index}'

        def layer_input( module, inputs, prefix=prefix ):
            capture.put( f'{prefix}.input', inputs[ 0 ] )

        handles.append( layer.register_forward_pre_hook( layer_input ) )
        handles.append( layer.register_forward_hook( output_hook( f'{prefix}.output' ) ) )

        for residual_name in ( 'attn_residual', 'mlp_residual' ):
            residual = layer.attn_hyper_connection if residual_name == 'attn_residual' else layer.mlp_hyper_connection

            def residual_hook( module, inputs, output, name=f'{prefix}.{residual_name}' ):
                mixed, stream, injection = output
                capture.put( f'{name}.stream', stream )
                capture.put( f'{name}.mixed', mixed )
                capture.put( f'{name}.injection', injection )

            handles.append( residual.register_forward_hook( residual_hook ) )

        if layer.layer_type == 'linear_attention':
            handles.append( layer.linear_attn.register_forward_hook( output_hook( f'{prefix}.mixer' ) ) )
        else:
            def attention_hook( module, inputs, output, prefix=prefix ):
                capture.put( f'{prefix}.mixer', output[ 0 ] )

            def indexer_hook( module, inputs, output, prefix=prefix ):
                capture.put( f'{prefix}.indexer.selected', selected_mask( output ) )

            handles.append( layer.self_attn.register_forward_hook( attention_hook ) )
            handles.append( layer.self_attn.indexer.register_forward_hook( indexer_hook ) )

        handles.append( layer.mlp.register_forward_hook( output_hook( f'{prefix}.mlp' ) ) )

        if layer.ple is not None:
            def ngram_ids_hook( module, inputs, prefix=prefix ):
                capture.put( f'{prefix}.ple.ngram_ids', inputs[ 0 ] )

            ngram = layer.ple.ple_embedding
            handles.append( ngram.ngram_embedding.register_forward_pre_hook( ngram_ids_hook ) )
            handles.append( ngram.register_forward_hook( output_hook( f'{prefix}.ple.ngram_embedding' ) ) )
            handles.append( layer.ple.register_forward_hook( output_hook( f'{prefix}.ple.output' ) ) )

    def mixer_input( module, inputs ):
        capture.put( 'final_mixer.stream', inputs[ 0 ] )

    handles.append( text_model.hyper_connection_mixer.register_forward_pre_hook( mixer_input ) )
    handles.append( text_model.hyper_connection_mixer.register_forward_hook( output_hook( 'final_mixer.mixed' ) ) )

    return handles


def run_chunked( model, tokens: torch.Tensor, capture: Capture ):
    """The prompt in chunks through the cache, then one token per decode step; returns each step's logits."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache( config=model.config )
    steps = []
    start = 0

    for index, length in enumerate( PREFILL_CHUNKS ):
        steps.append( ( f'prefill{index}', start, start + length ) )
        start += length

    for index in range( DECODE_TOKENS ):
        steps.append( ( f'decode{index}', start, start + 1 ) )
        start += 1

    logits = []

    for step, begin, end in steps:
        capture.step = step
        output = model( input_ids=tokens[ :, begin : end ], past_key_values=cache, use_cache=True )
        cache = output.past_key_values
        capture.put( 'logits', output.logits )
        logits.append( output.logits[ 0 ] )

    capture.step = None

    return steps, torch.cat( logits, dim=0 )


def lift_indexer_budget( model, total: int ):
    """Every indexer keeps every complete block: QSA reduces to dense causal attention."""
    for layer in model.model.layers:
        if layer.layer_type != 'linear_attention':
            indexer = layer.self_attn.indexer
            indexer.token_budget = total
            indexer.block_topk = total


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--seed', type=int, default=20261008 )
    parser.add_argument( '--variant', choices=[ 'moe', 'dense' ], default='moe' )
    parser.add_argument( '--output-dir', type=Path, required=True )
    parser.add_argument( '--skip-conversion', action='store_true',
        help='Write the checkpoint and the capture only' )
    arguments = parser.parse_args()

    output = arguments.output_dir.resolve()
    checkpoint = output / 'hf_checkpoint'

    config = tiny_config()
    torch.manual_seed( arguments.seed )
    model = Qwen4ExpForCausalLM( config )

    if arguments.variant == 'dense':
        substitute_dense_feedforward( model, config )

    model.eval()
    randomize( model, arguments.seed + 1 )

    output.mkdir( parents=True, exist_ok=True )
    model.save_pretrained( checkpoint, safe_serialization=True )

    total = sum( PREFILL_CHUNKS ) + DECODE_TOKENS
    ids = sequence_ids( total, config.vocab_size )
    tokens = torch.tensor( [ ids ], dtype=torch.long )

    capture = Capture()
    handles = register_hooks( model, capture )

    with torch.no_grad():
        steps, chunked_logits = run_chunked( model, tokens, capture )

    for handle in handles:
        handle.remove()

    with torch.no_grad():
        full_logits = model( input_ids=tokens, use_cache=False ).logits[ 0 ]
        lift_indexer_budget( model, total )
        dense_logits = model( input_ids=tokens, use_cache=False ).logits[ 0 ]

    cache_error = ( chunked_logits - full_logits ).abs().max().item()
    selection_effect = ( full_logits - dense_logits ).abs().max().item()

    tensors = dict( capture.tensors )
    tensors[ 'tokens' ] = torch.tensor( ids, dtype=torch.int32 )

    # Each PLE layer's assembled table, and its int64 hash constants as decimal text: Mila's tests read FP32 and
    # INT32 tensors only, and a multiplier does not fit either.
    constants = []

    for index, layer in enumerate( model.model.layers ):
        if layer.ple is None:
            continue

        ngram = layer.ple.ple_embedding
        tensors[ f'weights.layer{index}.ple.ngram_table' ] = ngram.ngram_embedding.weight.detach().to( torch.float32 ).contiguous()

        for key, buffer in ( ( 'multipliers', ngram.layer_multipliers ), ( 'head_vocab_sizes', ngram.ngram_heads_vocab_sizes ),
                             ( 'head_offsets', ngram.ngram_heads_offsets ) ):
            constants.append( f'layer{index}.{key}=' + ','.join( str( int( value ) ) for value in buffer.tolist() ) )

        constants.append( f'layer{index}.table_rows={ngram.ngram_embedding.weight.shape[ 0 ]}' )

    ( output / f'qwen4_tiny_{arguments.variant}_ngram_constants.txt' ).write_text( '\n'.join( constants ) + '\n' )
    tensors[ 'full.logits' ] = full_logits.to( torch.float32 ).contiguous()
    tensors[ 'dense.logits' ] = dense_logits.to( torch.float32 ).contiguous()

    # The budget first binds where a query sees more than budget + compress_ratio - 1 tokens.
    dense_exact_positions = config.indexer_budget + config.indexer_compress_ratio - 1

    save_file( tensors, str( output / f'qwen4_tiny_{arguments.variant}_reference.safetensors' ), metadata={
        'variant': arguments.variant,
        'steps': ','.join( f'{step}:{begin}:{end}' for step, begin, end in steps ),
        'eos_token_id': str( EOS_TOKEN_ID ),
        'dense_exact_positions': str( dense_exact_positions ),
        'seed': str( arguments.seed ),
        'transformers': transformers.__version__,
        'torch': torch.__version__,
    } )

    if not arguments.skip_conversion:
        convert_qwen4( checkpoint.as_posix(), str( output / f'qwen4_tiny_{arguments.variant}_fp32.bin' ), 'float32' )
        convert_qwen4( checkpoint.as_posix(), str( output / f'qwen4_tiny_{arguments.variant}_bf16.bin' ), 'bfloat16' )

    print( f'\nwrote {output}' )
    print( f'{len( tensors )} captured tensors over steps {[ step for step, _, _ in steps ]}' )
    print( f'chunked against full logits, max |difference| {cache_error:.3e}' )
    print( f'selection against dense logits, max |difference| {selection_effect:.3e} '
           f'(zero through position {dense_exact_positions - 1})' )

    # The prefix where the budget cannot bind must be exactly dense; past it, selection must change something,
    # or the capture does not exercise QSA at all.
    prefix_error = ( full_logits[ : dense_exact_positions ] - dense_logits[ : dense_exact_positions ] ).abs().max().item()

    if prefix_error != 0.0:
        raise RuntimeError( f'selection changed logits inside the dense-exact prefix: {prefix_error:.3e}' )

    if selection_effect == 0.0:
        raise RuntimeError( 'selection never changed a logit: the capture does not exercise QSA' )


if __name__ == '__main__':
    main()
