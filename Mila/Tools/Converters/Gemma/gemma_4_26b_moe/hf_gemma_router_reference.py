# HuggingFace router reference for Gemma 4 26B-A4B (Specifications/Gemma4MoE.md, Phase 5).
#
# Runs transformers' Gemma4TextRouter on its own, on one decoder layer's real router tensors, and
# writes the inputs and outputs Mila's Router and RouterOp are gated against. The three router
# tensors are read out of the checkpoint shards by byte range, so the download is the config, the
# shard index and well under a megabyte of weights. No GPU is used.
#
# Captured at FP32 and at BF16. HF ranks probabilities held at model precision, and BF16 rounding
# can tie two experts whose logits differ; an index gate needs both captures to tell a correct op
# from a wrong one. The router logits are recorded too, so the selection op can be gated alone.
#
# Hidden states are seeded Gaussian rows, not real residuals: this pins the arithmetic, not the
# routing statistics of real text.
#
#   python hf_gemma_router_reference.py --layer 5 --tokens 256 \
#       --output ../../../../Data/models/gemma/gemma4_26b_router_layer5_reference.safetensors

import argparse
import json
import struct
from pathlib import Path

import torch
import transformers
from huggingface_hub import HfFileSystem, hf_hub_download
from safetensors.torch import save_file
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRouter

MODEL_ID = "google/gemma-4-26B-A4B-it"

SAFETENSORS_DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}


def read_tensor( filesystem, shard, name ):
    """Read one tensor out of a remote safetensors shard by byte range."""
    with filesystem.open( f"{MODEL_ID}/{shard}", "rb" ) as handle:
        header_length = struct.unpack( "<Q", handle.read( 8 ) )[ 0 ]
        header = json.loads( handle.read( header_length ) )
        entry = header[ name ]
        begin, end = entry[ "data_offsets" ]
        handle.seek( 8 + header_length + begin )
        payload = handle.read( end - begin )

    tensor = torch.frombuffer( bytearray( payload ), dtype=SAFETENSORS_DTYPES[ entry[ "dtype" ] ] )

    return tensor.reshape( entry[ "shape" ] ).clone()


def router_logits( router, hidden_states ):
    # The first half of Gemma4TextRouter.forward, through the module's own submodules, because
    # forward() does not return the logits. main() checks these reproduce forward()'s selection.
    normalized = router.norm( hidden_states )
    scaled = normalized * router.scale * router.scalar_root_size

    return router.proj( scaled )


def capture( router, hidden_states, dtype ):
    router = router.to( dtype )
    hidden = hidden_states.to( dtype )

    with torch.no_grad():
        probabilities, weights, indices = router( hidden )
        logits = router_logits( router, hidden )

    recomputed = torch.nn.functional.softmax( logits, dim=-1 )

    if not torch.equal( recomputed, probabilities ):
        raise RuntimeError( f"{dtype}: logits recomputed outside forward() do not reproduce its probabilities" )

    return {
        "logits": logits.contiguous(),
        "probabilities": probabilities.contiguous(),
        "weights": weights.contiguous(),
        "indices": indices.to( torch.int32 ).contiguous(),
    }


def selected_sets( weights, indices ):
    """Per row, the selected (index -> weight) mapping; the combine is a sum, so order is not part of it."""
    return [ dict( zip( row_indices.tolist(), row_weights.tolist() ) )
             for row_indices, row_weights in zip( indices, weights ) ]


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( "--layer", type=int, default=5 )
    parser.add_argument( "--tokens", type=int, default=256 )
    parser.add_argument( "--seed", type=int, default=20260912 )
    parser.add_argument( "--output", type=Path, required=True )
    arguments = parser.parse_args()

    with open( hf_hub_download( MODEL_ID, "config.json" ) ) as handle:
        config = Gemma4TextConfig.from_dict( json.load( handle )[ "text_config" ] )

    with open( hf_hub_download( MODEL_ID, "model.safetensors.index.json" ) ) as handle:
        weight_map = json.load( handle )[ "weight_map" ]

    prefix = f"model.language_model.layers.{arguments.layer}.router."
    filesystem = HfFileSystem()
    checkpoint = { suffix: read_tensor( filesystem, weight_map[ prefix + suffix ], prefix + suffix )
                   for suffix in ( "proj.weight", "scale", "per_expert_scale" ) }

    router = Gemma4TextRouter( config )
    router.load_state_dict( checkpoint, strict=True )
    router.eval()

    generator = torch.Generator().manual_seed( arguments.seed )
    hidden_states = torch.randn( arguments.tokens, config.hidden_size, generator=generator, dtype=torch.float32 )

    fp32 = capture( router, hidden_states, torch.float32 )
    bf16 = capture( router, hidden_states, torch.bfloat16 )

    tensors = {
        "hidden_states": hidden_states,
        "proj_weight": checkpoint[ "proj.weight" ].to( torch.float32 ).contiguous(),
        "scale": checkpoint[ "scale" ].to( torch.float32 ).contiguous(),
        "per_expert_scale": checkpoint[ "per_expert_scale" ].to( torch.float32 ).contiguous(),
    }

    for precision, captured in ( ( "fp32", fp32 ), ( "bf16", bf16 ) ):
        for name, tensor in captured.items():
            tensors[ f"{precision}.{name}" ] = tensor

    metadata = {
        "model": MODEL_ID,
        "layer": str( arguments.layer ),
        "tokens": str( arguments.tokens ),
        "seed": str( arguments.seed ),
        "num_experts": str( config.num_experts ),
        "top_k_experts": str( config.top_k_experts ),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }

    arguments.output.parent.mkdir( parents=True, exist_ok=True )
    save_file( tensors, str( arguments.output ), metadata=metadata )

    fp32_sets = selected_sets( fp32[ "weights" ].float(), fp32[ "indices" ] )
    bf16_sets = selected_sets( bf16[ "weights" ].float(), bf16[ "indices" ] )
    differing_rows = sum( 1 for a, b in zip( fp32_sets, bf16_sets ) if a.keys() != b.keys() )

    weight_sums = fp32[ "weights" ].sum( dim=-1 )

    print( f"wrote {arguments.output}" )
    print( f"layer {arguments.layer}, {arguments.tokens} tokens, top-{config.top_k_experts} of {config.num_experts}" )
    print( f"rows whose selected expert SET differs between FP32 and BF16: {differing_rows}" )
    print( f"FP32 combine-weight sums: min {weight_sums.min().item():.6f} max {weight_sums.max().item():.6f}" )


if __name__ == "__main__":
    main()
