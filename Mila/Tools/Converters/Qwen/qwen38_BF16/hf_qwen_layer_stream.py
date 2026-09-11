#!/usr/bin/env python3
# ============================================================================
# File: hf_qwen_layer_stream.py
# Layer-streamed HuggingFace reference for Qwen 3.8 27B (Phase 4)
# ============================================================================

"""
Produce the HuggingFace reference activations for Qwen 3.8 27B, one layer at a time.

WHY THIS EXISTS. The BF16 reference is 50 GiB against 31.8 GB of host RAM and a
12 GiB card, so `from_pretrained` cannot run and the Gemma parity method -- load the
whole model, hook every layer -- is unavailable at this scale. This driver holds ONE
decoder layer resident at a time: it materializes layer i's weights straight from the
shards, runs it on the hidden state, frees it, and moves on. Peak residency is one
layer (~0.7 GiB) plus the 2.54 GiB lm_head at the end.

WHAT IT EMITS. A MILA `.bin` container -- the same format the weight converters write
and `PretrainedModelReader` already reads -- holding the last-token hidden state after
every layer, after the final norm, and the last-position logits. The Mila side asserts
against those numbers rather than against printed digits.

THE DRIVER MIRRORS `Qwen3_5TextModel.forward` RATHER THAN REIMPLEMENTING IT, and the
three places that matters are all silent if got wrong:

  1. The causal mask is built by the model's own `create_causal_mask`. Passing
     `attention_mask=None` to a full-attention layer does NOT fall back to causal
     masking -- `eager_attention_forward` simply adds nothing, and the layer attends
     bidirectionally. The result is a plausible hidden state, not an error.
  2. Linear-attention layers get a DIFFERENT mask from full-attention ones (None here,
     since there is no padding). The 4D causal mask handed to a DeltaNet layer would
     hit `apply_mask_to_padding_states`, which expects a 2D padding mask.
  3. Position ids are expanded to four grids, of which the first is the text grid and
     the other three feed mRoPE. Text-only input makes all three identical, so mRoPE
     degenerates to standard RoPE -- exactly as the spec predicts -- but the SHAPE
     still has to be right or the rotary embedding indexes the wrong axis.

`--self-test` is the control that proves the driver: it builds a small random model,
runs it BOTH whole (`Qwen3_5TextModel` with `output_hidden_states=True`) and streamed,
and requires the two to agree bitwise. Run it before trusting any number this produces.

Usage:
    # Prove the driver first
    python Qwen/qwen38_BF16/hf_qwen_layer_stream.py --self-test

    # Reference for the 4-layer fixture (matches --max-layers 4 on the converter)
    python Qwen/qwen38_BF16/hf_qwen_layer_stream.py --model Qwen/Qwen3.8-27B \\
        --max-layers 4 --output <weights-dir>/qwen/qwen38_ref_l4.bin

    # Full 64-layer reference
    python Qwen/qwen38_BF16/hf_qwen_layer_stream.py --model Qwen/Qwen3.8-27B \\
        --output <weights-dir>/qwen/qwen38_ref.bin
"""

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).parent.parent.parent ) )

import argparse
import gc
import json

import numpy as np
import torch
from safetensors import safe_open

from common import MilaWeightWriter


SUPPORTED_MODELS = [
    'Qwen/Qwen3.8-27B',
]

# The prompt the parity comparison runs on. Deliberately ASCII: the tokenizer's
# NFC normalizer is not modelled in Mila, and the checkpoint and `transformers`
# disagree on mark-bearing scripts, so a non-ASCII prompt would put a tokenizer
# question inside a model measurement. Ids come from Qwen/convert_tokenizer.py.
DEFAULT_PROMPT_IDS = [ 760, 6511, 314, 9338, 369 ]   # "The capital of France is"

TEXT_PREFIX = 'model.language_model.'

TORCH_DTYPE_MAP = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
}


def _check_hf_error( model_name: str, e: Exception ):
    name = type( e ).__name__
    msg = str( e )

    if 'RepositoryNotFound' in name or '404' in msg:
        print( f"\nError: '{model_name}' not found on HuggingFace." )
        print( '  Check the model name and your network connection.' )
        sys.exit( 1 )

    raise e


def resolve_checkpoint( model_name: str ) -> Path:
    local = Path( model_name )

    if local.is_dir():
        return local

    from huggingface_hub import snapshot_download

    print( f'Resolving {model_name} (downloads only what is missing from the hub cache)...' )

    try:
        return Path( snapshot_download(
            model_name,
            allow_patterns=[ 'config.json', '*.safetensors', '*.safetensors.index.json' ] ) )
    except Exception as e:
        _check_hf_error( model_name, e )


# ============================================================================
# Shard access
# ============================================================================

class ShardedWeights:
    """Name-addressed read-only view over a sharded safetensors checkpoint.

    safetensors mmaps, so holding every shard open costs address space rather than
    resident memory, and a tensor is only paged in when it is read.
    """

    def __init__( self, checkpoint: Path ):
        index = json.loads( ( checkpoint / 'model.safetensors.index.json' ).read_text() )
        self._weight_map = index[ 'weight_map' ]
        self._handles = {}
        self._checkpoint = checkpoint

    def _handle( self, shard: str ):
        if shard not in self._handles:
            self._handles[ shard ] = safe_open(
                str( self._checkpoint / shard ), framework='pt', device='cpu' )

        return self._handles[ shard ]

    def has( self, name: str ) -> bool:
        return name in self._weight_map

    def get( self, name: str ) -> torch.Tensor:
        if name not in self._weight_map:
            raise KeyError( f'{name} is not in the checkpoint index' )

        return self._handle( self._weight_map[ name ] ).get_tensor( name )

    def get_rows( self, name: str, row_ids ) -> torch.Tensor:
        """Read only the named rows of a 2-D tensor -- the embedding table is 2.5 GiB."""
        handle = self._handle( self._weight_map[ name ] )
        slice_ = handle.get_slice( name )

        return torch.stack( [ slice_[ int( i ) : int( i ) + 1 ][ 0 ] for i in row_ids ] )

    def state_dict_for( self, prefix: str ) -> dict:
        """Every tensor under `prefix`, keyed relative to it."""
        return { name[ len( prefix ) : ]: self.get( name )
                 for name in self._weight_map if name.startswith( prefix ) }


# ============================================================================
# The streamed driver
# ============================================================================

class StreamedReference:
    """Runs a Qwen 3.5-family text stack layer by layer, holding one layer at a time.

    `weights` is anything with `state_dict_for(prefix)` and `get(name)`, so the
    self-test can drive this over an in-memory random model with no checkpoint.
    """

    def __init__( self, text_config, weights, device, dtype, dump_layer=-1 ):
        self.config = text_config
        self.weights = weights
        self.device = device
        self.dtype = dtype

        # When set, every stage inside this one layer is captured at FULL [T, width]
        # rather than the last token, so the Mila side can attribute a divergence to a
        # stage instead of to a layer.
        self.dump_layer = dump_layer
        self.stages = {}

        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

        self.rotary = Qwen3_5TextRotaryEmbedding( config=text_config ).to( device )

    def _build_layer( self, layer_index: int ):
        """Materialize one decoder layer directly onto the target device.

        Constructed on the meta device so the (slow, large) default initialization
        never allocates, then load_state_dict(assign=True) installs the checkpoint
        tensors as the parameters themselves rather than copying into them.
        """
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

        with torch.device( 'meta' ):
            layer = Qwen3_5DecoderLayer( self.config, layer_index )

        state = self.weights.state_dict_for( f'{TEXT_PREFIX}layers.{layer_index}.' )
        state = { k: v.to( device=self.device, dtype=self.dtype ) for k, v in state.items() }

        missing, unexpected = layer.load_state_dict( state, strict=False, assign=True )

        # A meta parameter left behind would silently produce garbage rather than
        # throw, so an incomplete load has to be an error here.
        if missing or unexpected:
            raise RuntimeError(
                f'layer {layer_index}: missing={sorted( missing )} unexpected={sorted( unexpected )}' )

        return layer.eval()

    def _install_stage_hooks( self, layer, layer_index: int ):
        """Capture every addressable stage inside one decoder layer.

        Two hook kinds are needed. A forward hook gives a submodule's OUTPUT, which covers
        the norms and projections; the stages that live inside `self_attn.forward` and
        `mlp.forward` have no module of their own, so they are read as the INPUT of the
        module that consumes them -- o_proj's input is the gated attention output, and
        down_proj's input is the SwiGLU activation.
        """
        handles = []

        def capture_output( name ):
            def hook( _module, _args, output ):
                tensor = output[ 0 ] if isinstance( output, tuple ) else output
                self.stages[ name ] = tensor.detach()[ 0 ].float().cpu().clone()

            return hook

        def capture_input( name ):
            def hook( _module, args ):
                self.stages[ name ] = args[ 0 ].detach()[ 0 ].float().cpu().clone()

            return hook

        outputs = { 'input_norm': layer.input_layernorm,
                    'post_attn_norm': layer.post_attention_layernorm,
                    'ffn_gate': layer.mlp.gate_proj,
                    'ffn_up': layer.mlp.up_proj,
                    'ffn_down': layer.mlp.down_proj }

        inputs = { 'ffn_act': layer.mlp.down_proj }

        if layer.layer_type == 'full_attention':
            attention = layer.self_attn
            outputs.update( { 'q_proj': attention.q_proj, 'k_proj': attention.k_proj,
                              'v_proj': attention.v_proj, 'q_normed': attention.q_norm,
                              'k_normed': attention.k_norm, 'o_proj': attention.o_proj } )
            inputs[ 'gated' ] = attention.o_proj
        else:
            mixer = layer.linear_attn
            outputs.update( { 'in_proj_qkv': mixer.in_proj_qkv, 'in_proj_z': mixer.in_proj_z,
                              'in_proj_a': mixer.in_proj_a, 'in_proj_b': mixer.in_proj_b,
                              'mixer_norm': mixer.norm, 'out_proj': mixer.out_proj } )
            inputs[ 'mixer_out' ] = mixer.out_proj

        for name, module in outputs.items():
            handles.append( module.register_forward_hook( capture_output( f'{name}' ) ) )

        for name, module in inputs.items():
            handles.append( module.register_forward_pre_hook( capture_input( f'{name}' ) ) )

        return handles

    def _capture_rope( self ):
        """Capture q/k AFTER the rotary embedding, which no module boundary exposes.

        RoPE is a free function called inside `Qwen3_5Attention.forward`, so there is no
        hook point. It also matters: Mila applies the rotation IN PLACE over the q_norm
        output slot, so the slot a naive mapping compares holds post-RoPE values while the
        q_norm hook above holds pre-RoPE ones -- a comparison that reports a 36% divergence
        which is entirely the rotation. Patching the function is the only way to line the
        two up. Restored by the caller.
        """
        from transformers.models.qwen3_5 import modeling_qwen3_5 as reference_module

        original = reference_module.apply_rotary_pos_emb

        def capturing( q, k, cos, sin, unsqueeze_dim=1 ):
            q_out, k_out = original( q, k, cos, sin, unsqueeze_dim )

            # [B, heads, T, head_dim] -> [T, heads * head_dim], the layout Mila's slot holds.
            self.stages[ 'q_roped' ] = q_out.detach()[ 0 ].transpose( 0, 1 ).reshape(
                q_out.shape[ 2 ], -1 ).float().cpu().clone()
            self.stages[ 'k_roped' ] = k_out.detach()[ 0 ].transpose( 0, 1 ).reshape(
                k_out.shape[ 2 ], -1 ).float().cpu().clone()

            return q_out, k_out

        reference_module.apply_rotary_pos_emb = capturing

        return lambda: setattr( reference_module, 'apply_rotary_pos_emb', original )

    def _masks( self, inputs_embeds, position_ids ):
        """The two masks Qwen3_5TextModel builds, by the same call it makes."""
        from transformers.masking_utils import create_causal_mask

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids )

        # No padding in this harness, so the linear-attention arm gets no mask --
        # the same value _update_linear_attn_mask returns for an all-ones mask.
        return causal_mask, None

    @torch.no_grad()
    def run( self, token_ids, num_layers, progress=True ):
        """Forward the prompt through `num_layers` layers, one resident at a time.

        Returns the last-token hidden state after every layer, after the final norm,
        and the last-position logits.
        """
        ids = torch.tensor( [ list( token_ids ) ], dtype=torch.long, device=self.device )
        batch, seq_len = ids.shape

        embed = self.weights.get_rows( f'{TEXT_PREFIX}embed_tokens.weight', token_ids )
        hidden = embed.to( device=self.device, dtype=self.dtype ).view( batch, seq_len, -1 )

        # Verbatim from Qwen3_5TextModel.forward: four grids, the first is the text
        # grid the attention uses for positions, the other three feed mRoPE.
        position_ids = torch.arange( seq_len, device=self.device ).view( 1, 1, -1 ).expand( 4, batch, -1 )
        text_position_ids = position_ids[ 0 ]
        rope_position_ids = position_ids[ 1: ]

        causal_mask, linear_mask = self._masks( hidden, text_position_ids )
        position_embeddings = self.rotary( hidden, rope_position_ids )

        layer_hidden = []

        for i in range( num_layers ):
            layer_type = self.config.layer_types[ i ]
            layer = self._build_layer( i )

            handles = []
            restore_rope = None
            block_input = None

            if i == self.dump_layer:
                handles = self._install_stage_hooks( layer, i )
                block_input = hidden[ 0 ].float().cpu().clone()

                if layer_type == 'full_attention':
                    restore_rope = self._capture_rope()

            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=linear_mask if layer_type == 'linear_attention' else causal_mask,
                position_ids=text_position_ids,
                past_key_values=None )

            layer_hidden.append( hidden[ 0, -1 ].float().cpu().clone() )

            if i == self.dump_layer:
                # The block INPUT too: a stage comparison is only meaningful if both sides
                # started from the same hidden state, and this is what proves they did.
                self.stages[ 'block_input' ] = block_input
                self.stages[ 'block_output' ] = hidden[ 0 ].float().cpu().clone()

            for handle in handles:
                handle.remove()

            if restore_rope is not None:
                restore_rope()

            del layer
            _release( self.device )

            if progress:
                summary = _summarize( layer_hidden[ -1 ] )
                print( f'[HF] layer_{i:02d} {layer_type:<17} {summary}', flush=True )

        final_hidden, logits = self._head( hidden )

        return {
            'layer_hidden': layer_hidden,
            'final_hidden': final_hidden,
            'logits': logits,
        }

    @torch.no_grad()
    def _head( self, hidden ):
        """Final norm and lm_head, on the last position only."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        with torch.device( 'meta' ):
            norm = Qwen3_5RMSNorm( self.config.hidden_size, eps=self.config.rms_norm_eps )

        norm.load_state_dict(
            { 'weight': self.weights.get( f'{TEXT_PREFIX}norm.weight' )
                            .to( device=self.device, dtype=self.dtype ) }, assign=True )

        normed = norm( hidden )[ :, -1: ]
        final_hidden = normed[ 0, -1 ].float().cpu().clone()

        lm_head = self.weights.get( 'lm_head.weight' ).to( device=self.device, dtype=self.dtype )
        logits = ( normed @ lm_head.T )[ 0, -1 ].float().cpu().clone()

        del lm_head, norm
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
    """`ShardedWeights` interface over a live nn.Module, for the self-test."""

    def __init__( self, model, prefix: str ):
        self._state = { prefix + k: v for k, v in model.state_dict().items() }

    def has( self, name ):
        return name in self._state

    def get( self, name ):
        return self._state[ name ]

    def get_rows( self, name, row_ids ):
        return self._state[ name ][ list( row_ids ) ]

    def state_dict_for( self, prefix ):
        return { k[ len( prefix ) : ]: v for k, v in self._state.items() if k.startswith( prefix ) }


def self_test( device: str ) -> int:
    """Prove the driver reproduces Qwen3_5TextModel exactly, on a model small enough to run whole.

    The masks, the position-id expansion and the per-layer-type mask choice are the
    parts a streamed driver gets wrong, and every one of them fails silently -- a
    bidirectional attention still returns a hidden state. This is the only check that
    separates "the driver works" from "the driver runs".
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed( 0 )

    # Small, but structurally the real thing: the 3:1 interleave means both block
    # kinds run. EIGHT layers, not four, for two reasons that both cost a debugging
    # session to find. `output_hidden_states` does not expose the last layer's raw
    # output -- its final entry is the POST-NORM state -- so the last layer can only
    # be checked through the norm. And a causal-mask error in the LAST layer is
    # invisible at the last token, because the final row of a causal mask masks
    # nothing; it only becomes observable once a position-mixing layer runs after it,
    # which the recurrent DeltaNet layers at 4-6 supply.
    config = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        full_attention_interval=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        attn_output_gate=True,
        tie_word_embeddings=False )

    config._attn_implementation = 'eager'

    # float32: the control is a bitwise comparison, and bf16 rounding would make a
    # real ordering difference indistinguishable from noise.
    model = Qwen3_5TextModel( config ).to( device=device, dtype=torch.float32 ).eval()

    token_ids = [ 3, 17, 200, 5, 91, 42 ]
    ids = torch.tensor( [ token_ids ], dtype=torch.long, device=device )

    with torch.no_grad():
        whole = model( input_ids=ids, use_cache=False, output_hidden_states=True )

    # hidden_states[0] is the embedding and [i+1] is the output of layer i, EXCEPT
    # for the final entry, which transformers overwrites with the post-norm state --
    # so the last layer's raw output is not available and is checked through the norm.
    reference = [ whole.hidden_states[ i + 1 ][ 0, -1 ].float().cpu()
                  for i in range( config.num_hidden_layers - 1 ) ]
    reference_final = whole.last_hidden_state[ 0, -1 ].float().cpu()

    weights = InMemoryWeights( model, TEXT_PREFIX )
    weights._state[ 'lm_head.weight' ] = torch.zeros(
        config.vocab_size, config.hidden_size, device=device )

    streamed = StreamedReference( config, weights, device, torch.float32 ).run(
        token_ids, config.num_hidden_layers, progress=False )

    failures = 0

    for i, ( got, want ) in enumerate( zip( streamed[ 'layer_hidden' ], reference ) ):
        delta = ( got - want ).abs().max().item()
        status = 'OK  ' if delta == 0.0 else 'FAIL'
        print( f'  {status} layer_{i:02d} {config.layer_types[ i ]:<17} max|delta|={delta:.3e}' )
        failures += ( delta != 0.0 )

    delta = ( streamed[ 'final_hidden' ] - reference_final ).abs().max().item()
    status = 'OK  ' if delta == 0.0 else 'FAIL'
    print( f'  {status} final_norm{"":<20} max|delta|={delta:.3e}' )
    failures += ( delta != 0.0 )

    # A negative control: without it, a comparison that passes proves only that both
    # sides ran, not that either would notice a difference.
    bidirectional = StreamedReference( config, weights, device, torch.float32 )
    bidirectional._masks = lambda embeds, position_ids: ( None, None )
    broken = bidirectional.run( token_ids, config.num_hidden_layers, progress=False )
    broken_delta = ( broken[ 'final_hidden' ] - reference_final ).abs().max().item()

    if broken_delta == 0.0:
        print( '  FAIL negative control: dropping the causal mask changed nothing, '
               'so this comparison cannot detect a mask error' )
        failures += 1
    else:
        print( f'  OK   negative control (no causal mask) diverges: max|delta|={broken_delta:.3e}' )

    print( f'\nSelf-test: {"PASSED" if failures == 0 else f"FAILED ({failures})"}' )

    return 0 if failures == 0 else 1


# ============================================================================

def run_reference( model_name: str, output_path: str, prompt_ids, max_layers: int,
                   device: str, dtype_name: str, dump_layer: int = -1 ):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    checkpoint = resolve_checkpoint( model_name )
    config_json = json.loads( ( checkpoint / 'config.json' ).read_text() )
    text_config = Qwen3_5TextConfig( **config_json[ 'text_config' ] )
    text_config._attn_implementation = 'eager'

    num_layers = text_config.num_hidden_layers if max_layers <= 0 else min(
        max_layers, text_config.num_hidden_layers )

    dtype = TORCH_DTYPE_MAP[ dtype_name ]

    print( f'Checkpoint: {checkpoint}' )
    print( f'Layers: {num_layers} of {text_config.num_hidden_layers}, '
           f'device={device}, dtype={dtype_name}' )
    print( f'Prompt ids: {list( prompt_ids )}\n' )

    weights = ShardedWeights( checkpoint )
    driver = StreamedReference( text_config, weights, device, dtype, dump_layer )
    result = driver.run( prompt_ids, num_layers )
    result[ 'stages' ] = driver.stages

    print( f'\n[HF] final_norm{"":<16} {_summarize( result[ "final_hidden" ] )}' )

    logits = result[ 'logits' ]
    top = torch.topk( logits, 5 )
    print( '[HF] top-5 logits: ' + ', '.join(
        f'{int( i )}={float( v ):.4f}' for v, i in zip( top.values, top.indices ) ) )

    _write_reference( output_path, prompt_ids, num_layers, text_config, dtype_name, result,
                      dump_layer )


def _write_reference( output_path, prompt_ids, num_layers, text_config, dtype_name, result,
                      dump_layer=-1 ):
    """Emit the reference as a MILA container, which the Mila side already knows how to read."""
    writer = MilaWeightWriter( Path( output_path ) )

    writer.set_metadata( {
        'kind': 'qwen38_layer_stream_reference',
        'num_layers': num_layers,
        'hidden_size': text_config.hidden_size,
        'vocab_size': text_config.vocab_size,
        'compute_dtype': dtype_name,
        'prompt_ids': list( map( int, prompt_ids ) ),
        'layer_types': list( text_config.layer_types[ :num_layers ] ),
        'dumped_stage_layer': dump_layer,
    } )

    writer.add_tensor( 'prompt_ids', np.asarray( prompt_ids, dtype=np.int32 ) )

    for i, row in enumerate( result[ 'layer_hidden' ] ):
        writer.add_tensor( f'hidden_layer_{i}', row.numpy().astype( np.float32 ) )

    writer.add_tensor( 'hidden_final', result[ 'final_hidden' ].numpy().astype( np.float32 ) )
    writer.add_tensor( 'logits_last', result[ 'logits' ].numpy().astype( np.float32 ) )

    # Full [T, width] stages for the dumped layer, so the Mila side can attribute a
    # divergence to a stage. Named stage_<name>; the Mila harness looks them up by name.
    for name, tensor in sorted( result.get( 'stages', {} ).items() ):
        writer.add_tensor( f'stage_{name}', tensor.numpy().astype( np.float32 ) )

    writer.write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Layer-streamed HuggingFace reference activations for Qwen 3.8' )
    parser.add_argument( '--self-test', action='store_true',
        help='Prove the streamed driver against the whole model on a small random one, then exit' )
    parser.add_argument( '--model', type=str,
        help=f'HuggingFace model name (supported: {", ".join( SUPPORTED_MODELS )}) '
             'or a local checkpoint directory' )
    parser.add_argument( '--output', type=str,
        help='Output path for the MILA reference container' )
    parser.add_argument( '--prompt-ids', type=int, nargs='+', default=DEFAULT_PROMPT_IDS,
        help='Prompt token ids (from Qwen/convert_tokenizer.py)' )
    parser.add_argument( '--max-layers', type=int, default=0,
        help='Run only the first N layers -- pairs with the converter\'s --max-layers fixture' )
    parser.add_argument( '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu' )
    parser.add_argument( '--dtype', type=str, default='bfloat16', choices=list( TORCH_DTYPE_MAP ) )
    parser.add_argument( '--dump-layer', type=int, default=-1,
        help='Also capture every stage inside this layer at full [T, width], for stage-level '
             'attribution against the Mila harness' )

    args = parser.parse_args()

    if args.self_test:
        sys.exit( self_test( args.device ) )

    if not args.model or not args.output:
        parser.error( '--model and --output are required unless --self-test is given' )

    run_reference( args.model, args.output, args.prompt_ids, args.max_layers,
                   args.device, args.dtype, args.dump_layer )
