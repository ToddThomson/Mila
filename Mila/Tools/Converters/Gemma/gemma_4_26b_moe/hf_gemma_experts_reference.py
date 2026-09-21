# HuggingFace expert-bank reference for Gemma 4 26B-A4B (Specifications/Gemma4MoE.md, Phase 6).
#
# Runs transformers' Gemma4TextExperts on seeded synthetic stacked weights and writes the inputs and
# output Mila's MixtureOfExperts is gated against. The arithmetic under test -- the gate-first chunk
# of gate_up_proj, gelu_pytorch_tanh, the weighted combine -- does not depend on trained values, so no
# gigabyte-scale expert bank is downloaded. No network, no GPU.
#
# The class is wrapped in @use_experts_implementation, which can dispatch to a batched kernel. The
# reference is the eager loop in the class's own source, reached through forward.__wrapped__; the
# dispatched result is compared against it and reported, so a divergence between HuggingFace's own
# paths is visible rather than silently becoming the reference.
#
#   python hf_gemma_experts_reference.py \
#       --output ../../../../Data/models/gemma/gemma4_26b_experts_reference.safetensors

import argparse
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( "--tokens", type=int, default=48 )
    parser.add_argument( "--hidden", type=int, default=64 )
    parser.add_argument( "--intermediate", type=int, default=32 )
    parser.add_argument( "--experts", type=int, default=16 )
    parser.add_argument( "--top-k", type=int, default=4 )
    parser.add_argument( "--seed", type=int, default=20260912 )
    parser.add_argument( "--output", type=Path, required=True )
    arguments = parser.parse_args()

    config = Gemma4TextConfig(
        hidden_size=arguments.hidden,
        moe_intermediate_size=arguments.intermediate,
        num_experts=arguments.experts,
        top_k_experts=arguments.top_k,
        hidden_activation="gelu_pytorch_tanh",
        enable_moe_block=True )
    config._experts_implementation = "eager"

    experts = Gemma4TextExperts( config ).eval()
    generator = torch.Generator().manual_seed( arguments.seed )

    with torch.no_grad():
        experts.gate_up_proj.copy_(
            torch.randn( arguments.experts, 2 * arguments.intermediate, arguments.hidden, generator=generator ) * 0.1 )
        experts.down_proj.copy_(
            torch.randn( arguments.experts, arguments.hidden, arguments.intermediate, generator=generator ) * 0.1 )

    hidden_states = torch.randn( arguments.tokens, arguments.hidden, generator=generator )

    # Distinct experts per token, as a router's top-k produces them.
    scores = torch.rand( arguments.tokens, arguments.experts, generator=generator )
    indices = scores.argsort( dim=-1, descending=True )[ :, : arguments.top_k ].contiguous()
    weights = ( torch.rand( arguments.tokens, arguments.top_k, generator=generator ) + 0.1 ).contiguous()

    with torch.no_grad():
        eager = type( experts ).forward.__wrapped__( experts, hidden_states, indices, weights )
        dispatched = experts( hidden_states, indices, weights )

    tensors = {
        "hidden_states": hidden_states.contiguous(),
        "gate_up_proj": experts.gate_up_proj.detach().contiguous(),
        "down_proj": experts.down_proj.detach().contiguous(),
        "indices": indices.to( torch.int32 ).contiguous(),
        "weights": weights,
        "output": eager.contiguous(),
    }

    metadata = {
        "tokens": str( arguments.tokens ),
        "hidden": str( arguments.hidden ),
        "intermediate": str( arguments.intermediate ),
        "experts": str( arguments.experts ),
        "top_k": str( arguments.top_k ),
        "seed": str( arguments.seed ),
        "hidden_activation": config.hidden_activation,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }

    arguments.output.parent.mkdir( parents=True, exist_ok=True )
    save_file( tensors, str( arguments.output ), metadata=metadata )

    print( f"wrote {arguments.output}" )
    print( f"{arguments.tokens} tokens, {arguments.experts} experts x intermediate {arguments.intermediate}, "
           f"top-{arguments.top_k}, hidden {arguments.hidden}" )
    print( f"output max |value| {eager.abs().max().item():.6f}" )
    print( f"dispatched forward equals the eager loop: {torch.equal( eager, dispatched )} "
           f"(max difference {( eager - dispatched ).abs().max().item():.3e})" )


if __name__ == "__main__":
    main()
