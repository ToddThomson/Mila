"""The Qwen 3.8 deployment allocation, as the packer's target table.

This file is the Python counterpart of `Src/Dnn/Components/Transformers/Qwen/
Qwen.PrecisionPlan.ixx`: the same Section 5 rows, expressed as what the offline tool
must produce rather than as what the runtime must instantiate. The two are checked
against each other by name and by policy spelling, because they are the two halves of
one contract -- if they disagree the artifact loads into a model that decodes it wrongly.

THREE CLASSES OF LINEAR TENSOR, and the difference is where the quantization happens:

  codebook  fitted offline against calibration data and PACKED into the artifact.
            Codes cannot be recovered from weights, so this is the only path.
  fp4       written BF16 and quantized at load, per Qwen3.8.md section 8 step 5. FP4's
            level table is format-defined and identical for every tensor, so the load
            path needs no fitted table and the artifact needs no new record kind.
  bf16      never quantized. beta and decay drive the forget gate, where error compounds
            EXPONENTIALLY over the sequence; they are 0.1% of parameters.

The research scheme that maximizes compression is not this. Phase 0 assigned attention
q/k to cb8 and v to cb4 -- two formats inside one fused tensor, which no shared table
reconciles. Section 5 puts full attention at FP4 instead, which fuses trivially.
"""

import sys
from collections import namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Converters"))
from common import expand_qwen_tensor_map

# Mila tensor stem -> the policy that stem is written with. Every `is_linear` stem in
# the Qwen map must appear here: a stem with no entry is a hard error rather than a
# default, so adding a projection to the chassis forces a deliberate bit decision.
QWEN_ARTIFACT_POLICIES = {
    # Feed-forward, both block kinds. gate+up fuse into one tensor and therefore share
    # one codebook; down sits a half step above because it reads post-SwiGLU
    # activations, which carry the heavy outliers.
    "fc_gate_up": "cb4",
    "fc_down": "cb8",

    # Gated DeltaNet. q/k address the state matrix and sit a half step above v, the
    # output gate and o, which carry the magnitude that enters it.
    "fc_in_proj_qk": "cb8",
    "fc_in_proj_v": "cb4",
    "fc_in_proj_z": "cb4",
    "fc_out_proj": "cb4",

    # The recurrence's gating. Never quantized.
    "fc_in_proj_a": "bf16",
    "fc_in_proj_b": "bf16",

    # Full attention, and the head. BF16 in the artifact, FP4 at load.
    "fc_qkv_proj": "fp4",
    "fc_o_proj": "fp4",
    "lm_head": "fp4",
}

# Policy spelling as the C++ side names it, for the artifact's per-tensor policy map.
POLICY_TYPE_NAMES = {
    "cb4": "PerGroupCodebook2<32>",
    "cb8": "PerGroupCodebook3<64>",
    "fp4": "PerGroupFp4<128>",
    "bf16": "NoWeightQuant",
}

CODEBOOK_POLICIES = ("cb4", "cb8")


# One quantization unit: a contiguous row range of one HuggingFace weight.
#
#   module  layer-relative path, what named_modules() reports -- addresses the live
#           module for a Hessian hook and for the write-back.
#   rows    half-open row range, or None for the whole tensor.
#   source  the full checkpoint tensor name -- the only handle that reaches a SHAPE
#           before any layer is built, which the artifact's declaration pass needs.
#
# Both names are carried rather than derived from each other: the derivation depends on
# the checkpoint's text prefix, and doing it twice is how the two drift.
Piece = namedtuple("Piece", "module rows source")


class Target:
    """One Mila linear tensor and the HuggingFace pieces it is quantized from.

    Two shapes of fusion appear and they are not symmetric:

      several modules -> one tensor   fc_gate_up, from gate_proj and up_proj. One
                                      codebook across both, so they are fitted jointly.
      one module -> several tensors   in_proj_qkv, whose row ranges become fc_in_proj_qk
                                      and fc_in_proj_v AT DIFFERENT POLICIES. Legitimate
                                      because GPTQ's column walk compensates each output
                                      row independently, so quantizing a row range in
                                      isolation is identical to quantizing it inside the
                                      whole matrix -- given the same Hessian, which the
                                      two slices share because they share an input.
    """

    def __init__(self, mila_name, policy, pieces):
        self.mila_name = mila_name
        self.policy = policy
        self.pieces = pieces

    @property
    def stem(self):
        return self.mila_name.removesuffix(".weight")

    def __repr__(self):
        return f"Target({self.mila_name}, {self.policy}, {self.pieces})"


def layer_targets(num_layers, full_attention_interval, key_dim, value_dim,
                  prefix="model."):
    """{layer_index: [Target]} for every quantizable linear in the stack, plus a
    "post" key holding lm_head.

    Derived from the converter's own map rather than restated: the packer and the BF16
    converter must name and fuse a tensor identically or the artifact will not load.
    """
    mappings = expand_qwen_tensor_map(num_layers, full_attention_interval,
                                      key_dim, value_dim, prefix)
    targets = {}

    for mapping in mappings:
        if not mapping.is_linear:
            continue

        stem = mapping.mila.removesuffix(".weight")
        short = stem.split(".", 1)[-1] if stem.startswith("tf_layer_") else stem
        policy = QWEN_ARTIFACT_POLICIES.get(short)

        if policy is None:
            raise KeyError(
                f"{mapping.mila}: no bit allocation for this projection. Every linear "
                f"tensor needs a deliberate policy in QWEN_ARTIFACT_POLICIES; there is "
                f"no default.")

        pieces = tuple(
            Piece(_module_path(source, prefix), mapping.rows, source)
            for source in mapping.sources)

        key = (int(stem.split(".", 1)[0].removeprefix("tf_layer_"))
               if stem.startswith("tf_layer_") else "post")
        targets.setdefault(key, []).append(Target(mapping.mila, policy, pieces))

    return targets


def _module_path(source, prefix):
    """'model.layers.7.mlp.gate_proj.weight' -> 'mlp.gate_proj'.

    A layer-relative module path, which is the key a decoder layer's named_modules()
    reports and therefore what a hook or a write-back addresses.
    """
    name = source.removesuffix(".weight")

    for marker in ("layers.",):
        if marker in name:
            after = name.split(marker, 1)[1]

            return after.split(".", 1)[1]

    return name.removeprefix(prefix)


def hessian_modules(targets):
    """Every module path a layer's targets need a Hessian for, deduplicated.

    in_proj_qkv appears in two targets at two policies and is accumulated once.
    """
    modules = []

    for target in targets:
        if target.policy == "bf16":
            continue

        for piece in target.pieces:
            if piece.module not in modules:
                modules.append(piece.module)

    return modules


def describe(targets):
    """One line per target, for the run log -- the allocation as it will be applied."""
    lines = []

    for target in targets:
        pieces = ", ".join(
            piece.module + (f"[{piece.rows[0]}:{piece.rows[1]}]" if piece.rows else "")
            for piece in target.pieces)
        lines.append(f"{target.mila_name:<34} {target.policy:<5} <- {pieces}")

    return lines
