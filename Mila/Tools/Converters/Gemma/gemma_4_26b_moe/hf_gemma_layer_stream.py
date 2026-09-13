#!/usr/bin/env python3
# ============================================================================
# File: hf_gemma_layer_stream.py
# Layer-streamed HuggingFace reference for Gemma 4 26B-A4B (Gemma4MoE.md Phase 8)
# ============================================================================

"""
Produce the HuggingFace reference activations for Gemma 4, one decoder layer at a time.

WHY THIS EXISTS. The 26B-A4B checkpoint is 48 GiB against 31.8 GB of host RAM and a
16 GiB card, so `from_pretrained` cannot run. This driver holds ONE decoder layer
resident: it materializes layer i straight from the shards, runs it on the hidden
state, frees it and moves on. Peak residency is one layer (~1.6 GiB at bfloat16) plus
the tied 1.37 GiB table at the end. Method of record: Specifications/Qwen3.8.md s8.

WHAT IT EMITS. A MILA `.bin` container, read by `PretrainedModelReader`, holding the
last-token hidden state after every layer, after the final norm, and the last-position
logits BEFORE the final softcap (Mila applies the softcap at the sampler).

THE DRIVER MIRRORS `Gemma4TextModel.forward` RATHER THAN REIMPLEMENTING IT:

  1. Both masks come from the model's own calls -- `create_causal_mask` for full
     attention and `create_sliding_window_causal_mask` for sliding. Passing None to an
     eager attention layer attends bidirectionally and still returns a hidden state.
  2. Rotary embeddings are computed per layer type by `Gemma4TextRotaryEmbedding`; the
     global layers use the proportional partial rotation, the sliding ones the default.
  3. The embedding scale is multiplied in the TABLE's dtype, as
     `Gemma4TextScaledWordEmbedding` does, so bfloat16 rounds sqrt(hidden) the same way.
  4. Experts run through the eager `Gemma4TextExperts.forward`.

`--self-test` proves the driver: a small random routed model run BOTH whole and streamed
must agree bitwise, and a negative control that drops the masks must diverge.

Usage:
    python Gemma/gemma_4_26b_moe/hf_gemma_layer_stream.py --self-test

    python Gemma/gemma_4_26b_moe/hf_gemma_layer_stream.py \\
        --model <weights-dir>/gemma/gemma-4-26B-A4B-it \\
        --output <weights-dir>/gemma/gemma4_26b_a4b_ref.bin
"""

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).resolve().parent.parent.parent ) )
sys.path.insert( 0, str( Path( __file__ ).resolve().parent.parent ) )

import argparse
import gc
import json

import numpy as np
import torch

from common import MilaWeightWriter, ShardedCheckpoint


# "The capital of France is" in the instruct chat turn, thinking off -- the prompt the 12B parity test
# uses. The bare sentence is out of distribution for an instruct model: its last-position top tokens are
# " DO" and " CAP" at logit ~5, at FP32 as well as BF16, so a divergence there would be measured on
# noise. Decoded and printed on every run as a check.
DEFAULT_PROMPT_IDS = [ 2, 105, 2364, 107, 818, 5279, 529, 7001, 563, 106, 107, 105, 4368, 107, 100, 45518, 107, 101 ]

TORCH_DTYPE_MAP = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
}


def _text_config( config_json: dict ):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    config = Gemma4TextConfig( **config_json.get( 'text_config', config_json ) )
    config._attn_implementation = 'eager'
    config._experts_implementation = 'eager'

    return config


def _text_prefix( names ) -> str:
    """'model.language_model.' for the multimodal packaging, 'model.' for a text-only one."""
    suffix = 'embed_tokens.weight'
    key = next( ( n for n in sorted( names )
                  if n.endswith( suffix ) and 'vision' not in n and 'audio' not in n ), None )

    if key is None:
        raise KeyError( 'embed_tokens.weight not found in the checkpoint' )

    return key[ : -len( suffix ) ]


# ============================================================================
# The streamed driver
# ============================================================================

class StreamedReference:
    """Runs a Gemma 4 text stack layer by layer, holding one layer at a time.

    `weights` is anything with `tensor(name)`, `rows(name, ids)` and `state_dict_for(prefix)`,
    so the self-test can drive it over an in-memory model with no checkpoint.
    """

    def __init__( self, text_config, weights, prefix, device, dtype ):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        self.config = text_config
        self.weights = weights
        self.prefix = prefix
        self.device = device
        self.dtype = dtype
        self.rotary = Gemma4TextRotaryEmbedding( config=text_config ).to( device )

    def _build_layer( self, layer_index: int ):
        """Construct on the meta device, then install the checkpoint tensors as the parameters."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

        with torch.device( 'meta' ):
            layer = Gemma4TextDecoderLayer( self.config, layer_index )

        state = self.weights.state_dict_for( f'{self.prefix}layers.{layer_index}.' )
        state = { k: v.to( device=self.device, dtype=self.dtype ) for k, v in state.items() }

        missing, unexpected = layer.load_state_dict( state, strict=False, assign=True )

        # A meta tensor left behind produces garbage rather than an error.
        if missing or unexpected:
            raise RuntimeError(
                f'layer {layer_index}: missing={sorted( missing )} unexpected={sorted( unexpected )}' )

        return layer.eval()

    def _masks( self, inputs_embeds, position_ids ):
        """The two masks Gemma4TextModel builds, by the same calls it makes."""
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        mask_kwargs = {
            'config': self.config,
            'inputs_embeds': inputs_embeds,
            'attention_mask': None,
            'past_key_values': None,
            'position_ids': position_ids,
        }

        return {
            'full_attention': create_causal_mask( **mask_kwargs ),
            'sliding_attention': create_sliding_window_causal_mask( **mask_kwargs ),
        }

    @torch.no_grad()
    def run( self, token_ids, num_layers, progress=True ):
        ids = list( token_ids )
        seq_len = len( ids )

        table_name = f'{self.prefix}embed_tokens.weight'
        embedded = self.weights.rows( table_name, ids ).to( device=self.device, dtype=self.dtype )
        scale = torch.tensor( self.config.hidden_size ** 0.5 ).to( device=self.device, dtype=self.dtype )
        hidden = ( embedded * scale ).view( 1, seq_len, -1 )

        position_ids = torch.arange( seq_len, device=self.device ).unsqueeze( 0 )
        masks = self._masks( hidden, position_ids )
        position_embeddings = { layer_type: self.rotary( hidden, position_ids, layer_type )
                                for layer_type in set( self.config.layer_types ) }

        layer_hidden = []

        for i in range( num_layers ):
            layer_type = self.config.layer_types[ i ]
            layer = self._build_layer( i )

            hidden = layer(
                hidden,
                None,
                shared_kv_states={},
                position_embeddings=position_embeddings[ layer_type ],
                attention_mask=masks[ layer_type ],
                position_ids=position_ids,
                past_key_values=None )

            layer_hidden.append( hidden[ 0, -1 ].float().cpu().clone() )

            del layer
            _release( self.device )

            if progress:
                print( f'[HF] layer_{i:02d} {layer_type:<17} {_summarize( layer_hidden[ -1 ] )}', flush=True )

        final_hidden, logits = self._head( hidden )

        return { 'layer_hidden': layer_hidden, 'final_hidden': final_hidden, 'logits': logits }

    @torch.no_grad()
    def _head( self, hidden ):
        """Final norm, then the tied table as the head, on the last position only."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

        with torch.device( 'meta' ):
            norm = Gemma4RMSNorm( self.config.hidden_size, eps=self.config.rms_norm_eps )

        norm.load_state_dict(
            { 'weight': self.weights.tensor( f'{self.prefix}norm.weight' )
                            .to( device=self.device, dtype=self.dtype ) }, assign=True )

        normed = norm( hidden )[ :, -1: ]
        final_hidden = normed[ 0, -1 ].float().cpu().clone()

        head_name = f'{self.prefix}embed_tokens.weight' if self.config.tie_word_embeddings else 'lm_head.weight'
        head = self.weights.tensor( head_name ).to( device=self.device, dtype=self.dtype )
        logits = ( normed @ head.T )[ 0, -1 ].float().cpu().clone()

        del head, norm
        _release( self.device )

        return final_hidden, logits


def _release( device ):
    gc.collect()

    if str( device ).startswith( 'cuda' ):
        torch.cuda.empty_cache()


def _summarize( row: torch.Tensor ) -> str:
    head = ', '.join( f'{v:+.5f}' for v in row[ :3 ].tolist() )

    return ( f'l2={row.norm().item():>11.4f} mean={row.mean().item():>+10.5f} '
             f'min={row.min().item():>+10.4f} max={row.max().item():>+10.4f} head=[{head}]' )


# ============================================================================
# Self-test: the streamed driver against the whole model, on a small random one
# ============================================================================

class InMemoryWeights:
    """The checkpoint interface over a live module's state dict."""

    def __init__( self, state: dict ):
        self._state = state

    def tensor( self, name ):
        return self._state[ name ]

    def rows( self, name, row_ids ):
        return self._state[ name ][ list( row_ids ) ]

    def state_dict_for( self, prefix ):
        return { k[ len( prefix ) : ]: v for k, v in self._state.items() if k.startswith( prefix ) }


def self_test( device: str ) -> int:
    """Prove the driver reproduces Gemma4TextModel exactly on a model small enough to run whole."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
    from hf_gemma_moe_model_reference import randomize, tiny_config

    # The tiny wiring-gate geometry, stretched to six layers of period three so both
    # attention kinds run twice. A mask error in the LAST layer is invisible at the last
    # token, so the global layer at index 2 is followed by position-mixing layers. Ten
    # tokens against a window of four exercises the sliding mask.
    config = tiny_config()
    config.num_hidden_layers = 6
    config.layer_types = [ 'sliding_attention', 'sliding_attention', 'full_attention' ] * 2
    config.sliding_window = 4

    model = Gemma4TextModel( config ).to( device=device, dtype=torch.float32 ).eval()
    randomize( model, 20260913 )

    token_ids = [ ( i * 37 + 5 ) % config.vocab_size for i in range( 10 ) ]
    ids = torch.tensor( [ token_ids ], dtype=torch.long, device=device )

    with torch.no_grad():
        whole = model( input_ids=ids, use_cache=False, output_hidden_states=True )

    # hidden_states[i + 1] is layer i's output, except the last entry, which is the post-norm
    # state -- so the last layer is checked through the norm.
    reference = [ whole.hidden_states[ i + 1 ][ 0, -1 ].float().cpu()
                  for i in range( config.num_hidden_layers - 1 ) ]
    reference_final = whole.last_hidden_state[ 0, -1 ].float().cpu()

    weights = InMemoryWeights( { 'model.' + k: v for k, v in model.state_dict().items() } )

    streamed = StreamedReference( config, weights, 'model.', device, torch.float32 ).run(
        token_ids, config.num_hidden_layers, progress=False )

    failures = 0

    for i, ( got, want ) in enumerate( zip( streamed[ 'layer_hidden' ], reference ) ):
        delta = ( got - want ).abs().max().item()
        print( f'  {"OK  " if delta == 0.0 else "FAIL"} layer_{i:02d} {config.layer_types[ i ]:<17} max|delta|={delta:.3e}' )
        failures += ( delta != 0.0 )

    delta = ( streamed[ 'final_hidden' ] - reference_final ).abs().max().item()
    print( f'  {"OK  " if delta == 0.0 else "FAIL"} final_norm{"":<20} max|delta|={delta:.3e}' )
    failures += ( delta != 0.0 )

    # Negative control: a comparison that cannot fail proves only that both sides ran.
    unmasked = StreamedReference( config, weights, 'model.', device, torch.float32 )
    unmasked._masks = lambda embeds, position_ids: { 'full_attention': None, 'sliding_attention': None }
    broken = unmasked.run( token_ids, config.num_hidden_layers, progress=False )
    broken_delta = ( broken[ 'final_hidden' ] - reference_final ).abs().max().item()

    if broken_delta == 0.0:
        print( '  FAIL negative control: dropping the masks changed nothing' )
        failures += 1
    else:
        print( f'  OK   negative control (no masks) diverges: max|delta|={broken_delta:.3e}' )

    # The window alone: sliding layers given the full causal mask must diverge too, or the
    # comparison cannot see a sliding-window error.
    unwindowed = StreamedReference( config, weights, 'model.', device, torch.float32 )
    full_only = unwindowed._masks
    unwindowed._masks = lambda embeds, position_ids: dict(
        full_only( embeds, position_ids ), sliding_attention=full_only( embeds, position_ids )[ 'full_attention' ] )
    windowless = unwindowed.run( token_ids, config.num_hidden_layers, progress=False )
    windowless_delta = ( windowless[ 'final_hidden' ] - reference_final ).abs().max().item()

    if windowless_delta == 0.0:
        print( '  FAIL negative control: widening the sliding window changed nothing' )
        failures += 1
    else:
        print( f'  OK   negative control (no sliding window) diverges: max|delta|={windowless_delta:.3e}' )

    print( f'\nSelf-test: {"PASSED" if failures == 0 else f"FAILED ({failures})"}' )

    return 0 if failures == 0 else 1


# ============================================================================

def _open_checkpoint( model_dir: str ):
    root = Path( model_dir )
    text_config = _text_config( json.loads( ( root / 'config.json' ).read_text( encoding='utf-8' ) ) )
    weights = ShardedCheckpoint( root )

    return root, text_config, weights, _text_prefix( weights.names() )


def _decode( root: Path, token_ids ):
    tokenizer_path = root / 'tokenizer.json'

    if not tokenizer_path.exists():
        return None

    from tokenizers import Tokenizer

    return Tokenizer.from_file( str( tokenizer_path ) ).decode( list( token_ids ), skip_special_tokens=False )


# Gemma 4 instruct stop tokens: <eos> and <end_of_turn>, the ones GemmaModel stops on.
STOP_TOKEN_IDS = ( 1, 106 )


def run_greedy( model_dir: str, prompt_ids, steps: int, device: str, dtype_name: str ):
    """Greedy continuation by re-running the whole prompt per token -- no KV cache, one layer resident."""
    root, text_config, weights, prefix = _open_checkpoint( model_dir )
    driver = StreamedReference( text_config, weights, prefix, device, TORCH_DTYPE_MAP[ dtype_name ] )

    print( f'Prompt: {_decode( root, prompt_ids )!r}  dtype={dtype_name}\n' )

    ids = list( prompt_ids )
    generated = []

    for step in range( steps ):
        logits = driver.run( ids, text_config.num_hidden_layers, progress=False )[ 'logits' ]
        top = torch.topk( logits, 2 )
        token = int( top.indices[ 0 ] )

        print( f'[HF] step {step}: {token} {_decode( root, [ token ] )!r}  logit {float( top.values[ 0 ] ):.4f}, '
               f'runner-up {int( top.indices[ 1 ] )} at {float( top.values[ 1 ] ):.4f}', flush=True )

        generated.append( token )
        ids.append( token )

        if token in STOP_TOKEN_IDS:
            break

    print( f'\nGenerated: {_decode( root, generated )!r}' )
    print( 'kExpectedGen = { ' + ', '.join( str( t ) for t in generated ) + ' };' )


def run_reference( model_dir: str, output_path: str, prompt_ids, max_layers: int, device: str, dtype_name: str ):
    root, text_config, weights, prefix = _open_checkpoint( model_dir )
    num_layers = text_config.num_hidden_layers if max_layers <= 0 else min( max_layers, text_config.num_hidden_layers )

    decoded = _decode( root, prompt_ids )

    if decoded is not None:
        print( f'Prompt: {decoded!r}' )

    print( f'Checkpoint: {root}  prefix={prefix}' )
    print( f'Layers: {num_layers} of {text_config.num_hidden_layers}, device={device}, dtype={dtype_name}' )
    print( f'Prompt ids: {list( prompt_ids )}\n' )

    driver = StreamedReference( text_config, weights, prefix, device, TORCH_DTYPE_MAP[ dtype_name ] )
    result = driver.run( prompt_ids, num_layers )

    print( f'\n[HF] final_norm{"":<16} {_summarize( result[ "final_hidden" ] )}' )

    top = torch.topk( result[ 'logits' ], 5 )
    print( '[HF] top-5 logits: ' + ', '.join( f'{int( i )}={float( v ):.4f}' for v, i in zip( top.values, top.indices ) ) )

    writer = MilaWeightWriter( Path( output_path ) )
    writer.set_metadata( {
        'kind': 'gemma4_layer_stream_reference',
        'num_layers': num_layers,
        'hidden_size': text_config.hidden_size,
        'vocab_size': text_config.vocab_size,
        'compute_dtype': dtype_name,
        'prompt_ids': list( map( int, prompt_ids ) ),
        'layer_types': list( text_config.layer_types[ :num_layers ] ),
    } )

    writer.add_tensor( 'prompt_ids', np.asarray( prompt_ids, dtype=np.int32 ) )

    for i, row in enumerate( result[ 'layer_hidden' ] ):
        writer.add_tensor( f'hidden_layer_{i}', row.numpy().astype( np.float32 ) )

    writer.add_tensor( 'hidden_final', result[ 'final_hidden' ].numpy().astype( np.float32 ) )
    writer.add_tensor( 'logits_last', result[ 'logits' ].numpy().astype( np.float32 ) )
    writer.write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Layer-streamed HuggingFace reference activations for Gemma 4' )
    parser.add_argument( '--self-test', action='store_true',
        help='Prove the streamed driver against the whole model on a small random one, then exit' )
    parser.add_argument( '--model', type=str, help='Local checkpoint directory' )
    parser.add_argument( '--output', type=str, help='Output path for the MILA reference container' )
    parser.add_argument( '--prompt-ids', type=int, nargs='+', default=DEFAULT_PROMPT_IDS )
    parser.add_argument( '--max-layers', type=int, default=0,
        help='Run only the first N layers -- pairs with the converter\'s --max-layers file' )
    parser.add_argument( '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu' )
    parser.add_argument( '--dtype', type=str, default='bfloat16', choices=list( TORCH_DTYPE_MAP ) )
    parser.add_argument( '--generate', type=int, default=0,
        help='Print this many greedy tokens instead of writing a reference -- the whole prompt is re-run per token' )

    args = parser.parse_args()

    if args.self_test:
        sys.exit( self_test( args.device ) )

    if args.generate > 0:
        if not args.model:
            parser.error( '--generate requires --model' )

        run_greedy( args.model, args.prompt_ids, args.generate, args.device, args.dtype )
        sys.exit( 0 )

    if not args.model or not args.output:
        parser.error( '--model and --output are required unless --self-test or --generate is given' )

    run_reference( args.model, args.output, args.prompt_ids, args.max_layers, args.device, args.dtype )
