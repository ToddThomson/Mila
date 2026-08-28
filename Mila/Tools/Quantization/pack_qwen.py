#!/usr/bin/env python3
"""Pack Qwen 3.8 into the Section 5 allocation: a quantized Mila artifact, offline.

WHAT THIS PRODUCES, AND WHY IT IS NOT THE PUBLISHED MODEL. One safetensors artifact
carrying every tensor the BF16 converter writes, with the codebook rows replaced by
packed codes, per-group FP16 scales and a fitted per-tensor table. Everything else --
the FP4 rows, the embedding table, the norms -- stays BF16. That is a FITTED SOURCE,
not a finished artifact: the codebooks are the one thing no load can reconstruct, so
this produces them and stops.

ExportArtifact finishes it. `ExportArtifact <this file> <dst> --quantization plan`
loads the model under the Section 5 plan, which uploads these codes as they are and
quantizes the FP4 roles on the way in, then writes what ended up on the device. Every
model Mila publishes is written by that one path; this is the extra step in front of
Qwen's. Packing the FP4 rows here as well would make this a second writer of published
artifacts, which is what that decision rules out.

WHY IT STREAMS. Same reason the BF16 converter does, doubled: the checkpoint is
51.8 GiB against 31.8 GB of host RAM and a 12 GiB card, and GPTQ additionally needs a
[in, in] Hessian per linear -- 1.21 GB for down_proj alone. One decoder layer is
resident at a time, materialized from the shards, calibrated, quantized, emitted, freed.

THE ORDER WITHIN A LAYER IS THE WHOLE METHOD, and getting it wrong is silent:

  1. Accumulate Hessians by running the calibration set through the layer UNQUANTIZED.
  2. Quantize each target against those Hessians, compensating column by column.
  3. Damage the FP4 tensors in place, so the layers downstream calibrate against what
     the deployed network will actually carry rather than against BF16. The damaged
     values are what gets written; quantizing them at export yields the same nibbles
     quantizing the originals would, because FP4 is data-free and idempotent.
  4. Re-run the calibration set through the now-quantized layer to produce the next
     layer's inputs.

Skipping step 4 and reusing the unquantized outputs still runs, still produces an
artifact, and quietly discards most of what sequential compensation buys.

THE ROW-SLICED TARGETS ARE THE ONE STRUCTURAL NOVELTY. `in_proj_qkv` becomes two Mila
tensors at two different policies, which is legitimate because GPTQ's column walk
compensates each output row independently -- so quantizing a row range in isolation is
identical to quantizing it inside the whole matrix, given the same Hessian. The two
slices share one, because they share an input.

`--self-test` is the control. It packs a small random Qwen model end to end and checks
the artifact against three properties, one of which is a negative control. Run it
before trusting an hour-long run; see `self_test` for what each property catches.

Usage:
    python pack_qwen.py --self-test

    python pack_qwen.py --model Qwen/Qwen3.8-27B \\
        --calib-text corpus/wiki.train.raw \\
        --output artifacts/qwen38_27b_2p9bit.safetensors

    # Structural smoke test on the 4 layers the converter's fixture also covers
    python pack_qwen.py --model Qwen/Qwen3.8-27B --max-layers 4 --samples 4 \\
        --seqlen 2048 --calib-text corpus/wiki.train.raw \\
        --output artifacts/qwen38_l4_2p9bit.safetensors

    # Audit an artifact against the checkpoint, without regenerating it
    python pack_qwen.py --model Qwen/Qwen3.8-27B \\
        --verify artifacts/qwen38_27b_2p9bit.safetensors
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# formats sets CUBLAS_WORKSPACE_CONFIG, which must land before any CUDA context exists
# or determinism cannot be enabled. Imported before torch reaches this module.
from formats import (CODEBOOK_PARAMETERS, GENERATOR_SEED, GPTQ_FORMATS,
                     enforce_determinism, fake_fp4_e2m1, fit_codebook_levels,
                     fit_codebook_levels_joint)

import torch

import qwen_plan
from artifact import codebook_tensor_records, pack_codebook_tensor
from fit import gptq_quantize_tensor
from streaming_safetensors import StreamingSafetensorsWriter

_CONVERTERS = Path(__file__).resolve().parents[1] / "Converters"
sys.path.insert(0, str(_CONVERTERS))
sys.path.insert(0, str(_CONVERTERS / "Qwen"))
sys.path.insert(0, str(_CONVERTERS / "Qwen" / "qwen38_BF16"))

from common import apply_transform, expand_qwen_tensor_map
from convert_weights import (ShardedCheckpoint, _resolve_checkpoint, _text_prefix,
                             qwen_mila_metadata, resolve_qwen_geometry)
from hf_qwen_layer_stream import TEXT_PREFIX, InMemoryWeights, StreamedReference


class CheckpointWeights:
    """The StreamedReference weight interface over a `ShardedCheckpoint`, plus shapes.

    Shapes come from the shard headers, so the artifact's whole index can be declared
    before any tensor data is touched -- which is what lets the writer seek.
    """

    def __init__(self, checkpoint):
        self._checkpoint = checkpoint

    def has(self, name):
        return name in self._checkpoint.names()

    def get(self, name):
        return self._checkpoint.tensor(name)

    def shape(self, name):
        return self._checkpoint.shape(name)

    def state_dict_for(self, prefix):
        return {name[len(prefix):]: self._checkpoint.tensor(name)
                for name in self._checkpoint.names() if name.startswith(prefix)}


class InMemoryShapes(InMemoryWeights):
    """InMemoryWeights with the shape query the declaration pass needs."""

    def shape(self, name):
        return tuple(self._state[name].shape)


# ---------------------------------------------------------------------------
# Declaration: every artifact tensor's dtype and shape, from source shapes alone
# ---------------------------------------------------------------------------

def codebook_record_shapes(target, shape_of):
    """{tensor_name: (safetensors_dtype, shape)} for one codebook target.

    Derived from the same arithmetic `Linear::initializeParameters` uses, and checked
    against the produced arrays again at write time by codebook_tensor_records -- once
    here so the file can be laid out, once there so a mismatch names the tensor.
    """
    entries, group_size = CODEBOOK_PARAMETERS[target.policy]
    out_features = 0
    in_features = None

    for piece in target.pieces:
        source = shape_of(piece.source)
        out_features += (piece.rows[1] - piece.rows[0]) if piece.rows else source[0]
        in_features = source[1]

    name = target.stem
    shapes = {
        f"{name}.weight": ("U8", (out_features, in_features * 2 // 8)),
        f"{name}.weight_scale": ("F16", (out_features, in_features // group_size)),
        f"{name}.weight_codebook": ("F32", (entries,)),
    }

    if target.policy == "cb8":
        shapes[f"{name}.weight_high_plane"] = ("U8", (out_features, in_features // 8))

    return shapes


def fp4_record_shapes(target, shape_of):
    """{tensor_name: (safetensors_dtype, shape)} for one FP4 target.

    RETIRED -- no caller. This packer writes its FP4 roles through at BF16 and
    ExportArtifact packs them, so nothing here declares an FP4 record any more. Kept
    beside its codebook sibling because it is the only Python statement of the FP4
    record layout, and the fixture that proves byte parity against the CUDA quantizer
    needs exactly this arithmetic.

    Same arithmetic Linear::initializeParameters uses for PerGroupFp4: two nibbles per
    byte along the input axis, one FP32 scale per group of 128. Scales are FP32 and not
    FP16 as the codebook records use -- that is what the FP4 policy allocates, and the
    two formats are read by different code.
    """
    group_size = GPTQ_FORMATS["fp4"][1]
    out_features = 0
    in_features = None

    for piece in target.pieces:
        source = shape_of(piece.source)
        out_features += (piece.rows[1] - piece.rows[0]) if piece.rows else source[0]
        in_features = source[1]

    name = target.stem

    return {
        f"{name}.weight": ("U8", (out_features, in_features // 2)),
        f"{name}.weight_scale": ("F32", (out_features, in_features // group_size)),
    }


def passthrough_shape(mapping, shape_of, head_dim):
    """The shape a mapping's transform produces, from its source shapes alone."""
    shapes = [shape_of(source) for source in mapping.sources]

    if mapping.transform == "rows":
        start, stop = mapping.rows

        return (stop - start, shapes[0][1])

    if mapping.transform == "conv_rows":
        start, stop = mapping.rows

        return (stop - start, shapes[0][-1])

    if mapping.transform == "gated_qkv":
        # The de-interleave rearranges rows without changing their count.
        return (sum(shape[0] for shape in shapes), shapes[0][1])

    if len(shapes) == 1:
        return tuple(shapes[0])

    return (sum(shape[0] for shape in shapes),) + tuple(shapes[0][1:])


# ---------------------------------------------------------------------------
# The streamed packer
# ---------------------------------------------------------------------------

class QwenPacker(StreamedReference):
    """Quantize and emit a Qwen 3.8 stack, one decoder layer resident at a time.

    Inherits `_build_layer` and `_masks` from the Phase 4 reference driver rather than
    reimplementing them. Both are load-bearing and both fail silently: a full-attention
    layer given `attention_mask=None` attends bidirectionally and returns a plausible
    hidden state, and a linear-attention layer given the 4-D causal mask meets a
    padding-mask code path. Sharing them is what keeps a calibration run and the parity
    reference measuring the same model.
    """

    def __init__(self, text_config, weights, device, dtype, geometry, prefix, writer):
        super().__init__(text_config, weights, device, dtype)

        self.geometry = geometry
        self.prefix = prefix
        self.writer = writer
        self.total_params = 0
        self.total_bits = 0.0

    # -- calibration ------------------------------------------------------

    def calibration_hidden(self, samples):
        """Embed every calibration sample once, on the host.

        The table is read whole and dropped: 2.54 GiB against reading 65,536 rows one
        `get_slice` call at a time, which is what the parity driver does for a
        five-token prompt and would be pathological here.
        """
        table = self.weights.get(f"{TEXT_PREFIX}embed_tokens.weight")
        hidden = [table[sample].to(dtype=self.dtype).unsqueeze(0) for sample in samples]

        del table
        _release(self.device)

        return hidden

    def _forward_context(self, seq_len):
        """The masks and rotary embeddings for a fixed calibration length.

        Every sample is the same length, so this is computed once rather than per
        sample -- and per layer type, since the two kinds take different masks.
        """
        probe = torch.zeros(1, seq_len, self.config.hidden_size,
                            device=self.device, dtype=self.dtype)

        position_ids = torch.arange(seq_len, device=self.device).view(1, 1, -1).expand(4, 1, -1)
        text_position_ids = position_ids[0]

        causal_mask, linear_mask = self._masks(probe, text_position_ids)
        position_embeddings = self.rotary(probe, position_ids[1:])

        del probe

        return {"causal": causal_mask, "linear": linear_mask,
                "position_ids": text_position_ids,
                "position_embeddings": position_embeddings}

    def _run_layer(self, layer, layer_type, hidden_states, context, collect):
        """Forward every calibration sample through one layer.

        `collect` replaces each sample's hidden state with this layer's output; a
        Hessian pass passes False and leaves the inputs alone, because it must not
        advance the stack until the layer is quantized.
        """
        mask = context["linear"] if layer_type == "linear_attention" else context["causal"]

        with torch.no_grad():
            for index, hidden in enumerate(hidden_states):
                output = layer(
                    hidden.to(self.device),
                    position_embeddings=context["position_embeddings"],
                    attention_mask=mask,
                    position_ids=context["position_ids"],
                    past_key_values=None)

                if collect:
                    tensor = output[0] if isinstance(output, tuple) else output
                    hidden_states[index] = tensor.detach().cpu()

                del output

    def _accumulate_hessians(self, layer, layer_type, hidden_states, context, modules):
        """A [in, in] FP32 Hessian per module, over the calibration set."""
        by_path = dict(layer.named_modules())
        hessians = {}
        counts = {}
        handles = []

        def make_hook(path):
            def hook(_module, inputs):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()

                if path not in hessians:
                    hessians[path] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                    counts[path] = 0

                hessians[path] += x.T @ x
                counts[path] += x.shape[0]

            return hook

        for path in modules:
            handles.append(by_path[path].register_forward_pre_hook(make_hook(path)))

        self._run_layer(layer, layer_type, hidden_states, context, collect=False)

        for handle in handles:
            handle.remove()

        return hessians, counts

    # -- quantization -----------------------------------------------------

    @staticmethod
    def _piece_view(by_path, piece):
        """The weight rows one piece of a target owns, as a view onto the module."""
        weight = by_path[piece.module].weight.data

        return weight if piece.rows is None else weight[piece.rows[0]:piece.rows[1]]

    def _quantize_target(self, layer_index, layer, target, hessians, counts):
        """Quantize one Mila tensor in place. Returns its artifact records, or None."""
        by_path = dict(layer.named_modules())

        if target.policy == "bf16":
            return None

        if target.policy == "fp4":
            # Quantize-dequantize with no compensation, in place, so the layers
            # downstream calibrate against the damage the deployed network carries.
            # Compensating would optimize for a quantizer that never sees these codes.
            #
            # No records: the damaged view is written through at BF16 and the export
            # pass packs it. Returning None is what routes it to _emit_passthrough.
            group_size = GPTQ_FORMATS["fp4"][1]

            for piece in target.pieces:
                view = self._piece_view(by_path, piece)
                damaged, bits = fake_fp4_e2m1(view.float(), group_size)
                view.copy_(damaged.to(view.dtype))
                self.total_params += view.numel()
                self.total_bits += bits * view.numel()

            return None

        levels, group_size, divisor, bits = GPTQ_FORMATS[target.policy]
        entries = CODEBOOK_PARAMETERS[target.policy][0]
        views = [self._piece_view(by_path, piece) for piece in target.pieces]

        # One codebook per Mila tensor, so a fused tensor's sources are fitted jointly.
        # They share an input axis and therefore an importance vector; only the fit is
        # shared, and the joint fit is strictly less expressive than two separate ones
        # -- measured at 0.77% on the Llama proxy, against a 35% margin.
        first_module = target.pieces[0].module
        importance = torch.diag(hessians[first_module]) / counts[first_module]

        if len(views) > 1:
            levels = fit_codebook_levels_joint(views, entries, group_size,
                                               importance=importance)
        else:
            levels = fit_codebook_levels(views[0], entries, group_size,
                                         importance=importance)

        packed = []

        for piece, view in zip(target.pieces, views):
            quantized, codes, scale_bits = gptq_quantize_tensor(
                view, hessians[piece.module], levels, group_size, divisor)

            packed.append(pack_codebook_tensor(
                f"tf_layer_{layer_index}.{piece.module}", target.policy, quantized,
                codes, scale_bits, levels, group_size))

            view.copy_(quantized.to(view.dtype))
            self.total_params += view.numel()
            self.total_bits += bits * view.numel()

        return codebook_tensor_records(target.stem, packed, levels, group_size,
                                       views[0].shape[1])

    # -- emission ---------------------------------------------------------

    def _emit_passthrough(self, mappings, resolve_source):
        """Write the tensors that are not codebook records, at BF16.

        Read from the LIVE layer rather than from the shards: the FP4-at-load tensors
        were damaged in place above, and it is the damaged values the artifact must
        carry so that what Mila quantizes at load is what the calibration saw.
        """
        for mapping in mappings:
            sources = [resolve_source(name) for name in mapping.sources]
            tensor = apply_transform(mapping, sources, self.geometry["head_dim"])
            array = tensor.to(torch.bfloat16).contiguous().cpu().view(torch.uint16).numpy()

            self.writer.write(mapping.mila, array, bfloat16=True)

    def run(self, samples, targets, layer_mappings, outer_mappings, progress=True):
        """Walk the stack: calibrate, quantize, emit, advance."""
        hidden_states = self.calibration_hidden(samples)
        context = self._forward_context(len(samples[0]))
        num_layers = self.geometry["num_hidden_layers"]

        for layer_index in range(num_layers):
            started = time.time()
            layer_type = self.config.layer_types[layer_index]
            layer = self._build_layer(layer_index)
            layer_targets = targets[layer_index]

            modules = qwen_plan.hessian_modules(layer_targets)
            hessians, counts = self._accumulate_hessians(
                layer, layer_type, hidden_states, context, modules)

            packed_stems = set()

            for target in layer_targets:
                records = self._quantize_target(
                    layer_index, layer, target, hessians, counts)

                if records is None:
                    continue

                packed_stems.add(target.stem)

                for name, array in records.items():
                    self.writer.write(name, array)

            hessians.clear()
            counts.clear()
            _release(self.device)

            prefix = f"{TEXT_PREFIX}layers.{layer_index}."
            by_path = dict(layer.named_modules())

            def resolve_source(name, by_path=by_path, prefix=prefix):
                relative = name[len(prefix):].removesuffix(".weight")

                if relative in by_path:
                    return by_path[relative].weight.data

                # A_log and dt_bias are parameters of the mixer itself, not of a module.
                owner, _, attribute = relative.rpartition(".")

                return getattr(by_path[owner], attribute).data

            self._emit_passthrough(
                [mapping for mapping in layer_mappings[layer_index]
                 if mapping.mila.removesuffix(".weight").split(".", 1)[-1]
                 not in {stem.split(".", 1)[-1] for stem in packed_stems}],
                resolve_source)

            # The next layer must calibrate against what the deployed network carries,
            # so the advance runs on the QUANTIZED layer.
            self._run_layer(layer, layer_type, hidden_states, context, collect=True)

            del layer, by_path
            _release(self.device)

            if progress:
                # Peak allocation, reset per layer: the whole design rests on one layer
                # being resident at a time, and a climbing peak is what a leak looks
                # like here -- the Llama gate's reserved pool crept 8.5 GB to 11.5 GB
                # across 28 layers before empty_cache() was added per layer.
                peak = (torch.cuda.max_memory_allocated() / 1024**3
                        if str(self.device).startswith("cuda") else 0.0)

                print(f"  layer {layer_index + 1}/{num_layers} {layer_type:<17} "
                      f"{time.time() - started:6.1f}s  peak {peak:5.2f} GiB", flush=True)

                if str(self.device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()

        self._emit_outer(outer_mappings)

    def _emit_outer(self, mappings):
        """The tensors outside the decoder stack: the table, the final norm, the head.

        All three pass through at BF16. lm_head is an FP4 role, but it is not even
        damaged here: nothing inside the model reads its output, so damaging it would
        alter no calibration, and the export quantizes the original to the same nibbles
        it would have quantized the damaged copy to. The embedding table is quantized by
        no policy at all.
        """
        self._emit_passthrough(mappings, self.weights.get)


def _release(device):
    gc.collect()

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Driving one run
# ---------------------------------------------------------------------------

def verify_layer_kinds(text_config, geometry):
    """The tensor map's idea of which layers are full attention must be the model's.

    The map derives it arithmetically from `full_attention_interval`; the model carries
    `layer_types`, built by the config and consumed by `Qwen3_5DecoderLayer`. Nothing
    connects the two, and a disagreement is silent and total: the packer would look for
    `linear_attn.in_proj_qkv` on a layer holding `self_attn.q_proj`, or quantize an
    attention projection under a DeltaNet policy.
    """
    interval = geometry["full_attention_interval"]

    for index in range(geometry["num_hidden_layers"]):
        from_map = ((index + 1) % interval) == 0
        from_model = text_config.layer_types[index] == "full_attention"

        if from_map != from_model:
            raise RuntimeError(
                f"layer {index}: the tensor map says "
                f"{'full attention' if from_map else 'DeltaNet'} at interval {interval}, "
                f"the model says {text_config.layer_types[index]}")


def build_plan(geometry, prefix):
    """(targets, layer_mappings, outer_mappings) for the whole stack."""
    key_dim = geometry["linear_num_key_heads"] * geometry["linear_head_dim"]
    value_dim = geometry["linear_num_value_heads"] * geometry["linear_head_dim"]
    num_layers = geometry["num_hidden_layers"]
    interval = geometry["full_attention_interval"]

    targets = qwen_plan.layer_targets(num_layers, interval, key_dim, value_dim, prefix)
    mappings = expand_qwen_tensor_map(num_layers, interval, key_dim, value_dim, prefix)

    layer_mappings = {index: [] for index in range(num_layers)}
    outer_mappings = []

    for mapping in mappings:
        if mapping.mila.startswith("tf_layer_"):
            index = int(mapping.mila.split(".", 1)[0].removeprefix("tf_layer_"))
            layer_mappings[index].append(mapping)
        else:
            outer_mappings.append(mapping)

    return targets, layer_mappings, outer_mappings


def declare_artifact(writer, targets, layer_mappings, outer_mappings, shape_of):
    """Declare every tensor the run will write, before any of it exists."""
    packed_stems = set()

    for layer_targets in targets.values():
        for target in layer_targets:
            if target.policy not in qwen_plan.CODEBOOK_POLICIES:
                continue

            shapes = codebook_record_shapes(target, shape_of)

            packed_stems.add(target.stem)

            for name, (dtype, shape) in shapes.items():
                writer.declare(name, dtype, shape)

    for mappings in list(layer_mappings.values()) + [outer_mappings]:
        for mapping in mappings:
            if mapping.mila.removesuffix(".weight") in packed_stems:
                continue

            writer.declare(mapping.mila, "BF16", passthrough_shape(mapping, shape_of, 0))


def calibration_samples(text, tokenizer, count, seqlen):
    token_ids = tokenizer(text, return_tensors="pt").input_ids[0]

    if token_ids.numel() < seqlen * 2:
        raise ValueError(
            f"calibration text is {token_ids.numel()} tokens, too short for "
            f"{count} samples of {seqlen}")

    generator = torch.Generator().manual_seed(GENERATOR_SEED)
    starts = torch.randint(0, token_ids.numel() - seqlen, (count,), generator=generator)

    return [token_ids[start:start + seqlen] for start in starts]


def pack(model_name, output_path, calib_text_path, device, samples, seqlen,
         max_layers):
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    root = _resolve_checkpoint(model_name)
    config_json = json.loads((root / "config.json").read_text(encoding="utf-8"))
    geometry = resolve_qwen_geometry(config_json, max_layers)

    checkpoint = ShardedCheckpoint(root)
    prefix = _text_prefix(checkpoint)

    # _build_layer addresses the checkpoint through the reference driver's constant, so
    # a packaging it does not expect must fail here rather than half-load a layer.
    if prefix != TEXT_PREFIX:
        raise RuntimeError(
            f"checkpoint text prefix is '{prefix}' but the streamed driver addresses "
            f"'{TEXT_PREFIX}'; hf_qwen_layer_stream.TEXT_PREFIX has to follow")

    text_config = Qwen3_5TextConfig(**config_json.get("text_config", config_json))
    text_config._attn_implementation = "eager"

    verify_layer_kinds(text_config, geometry)

    weights = CheckpointWeights(checkpoint)
    targets, layer_mappings, outer_mappings = build_plan(geometry, prefix)

    print(f"Checkpoint: {root}")
    print(f"Layers: {geometry['num_hidden_layers']} of {text_config.num_hidden_layers}, "
          f"device={device}")
    print(f"Calibration: {samples} samples x {seqlen} tokens\n")

    print("Allocation (layer 0 is DeltaNet, layer "
          f"{geometry['full_attention_interval'] - 1} is full attention):")

    for line in qwen_plan.describe(targets[0] + targets[geometry["full_attention_interval"] - 1]
                                   + targets["post"]):
        print(f"  {line}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = Path(calib_text_path).read_text(encoding="utf-8")
    token_samples = calibration_samples(text, tokenizer, samples, seqlen)

    metadata = qwen_mila_metadata(geometry, "bfloat16",
                                  model_name.rsplit("/", 1)[-1].replace(".", "_").replace("-", "_"))

    writer = StreamingSafetensorsWriter(output_path,
                                        artifact_metadata(metadata, targets))
    declare_artifact(writer, targets, layer_mappings, outer_mappings, weights.shape)

    print(f"\n  {len(writer._declared)} artifact tensors, "
          f"{writer.payload_bytes() / 1024**3:.2f} GiB payload\n")

    packer = QwenPacker(text_config, weights, torch.device(device), torch.bfloat16,
                        geometry, prefix, writer)

    started = time.time()

    with writer:
        packer.run(token_samples, targets, layer_mappings, outer_mappings)

    average = packer.total_bits / packer.total_params if packer.total_params else 0.0

    print(f"\nPacked {packer.total_params / 1e9:.3f} B quantized parameters at "
          f"{average:.3f} average bits/weight")
    print(f"  {time.time() - started:.0f}s -> {output_path}")


def artifact_metadata(config_metadata, targets):
    """__metadata__ for the artifact: the architecture, the scheme, the per-tensor map.

    `mila_quantization` is one string and this artifact carries three policies, so the
    string names the scheme and the map names what each tensor actually is. Which of
    the two a Mila loader trusts is decided by the load path, which does not exist yet.

    Every quantizable linear is named, `NoWeightQuant` ones included, so that a loader
    reading the map never has to decide what an absent entry means.

    Built from the plan BEFORE the run rather than accumulated during it. The two
    produce the same map, but the header is written first, so accumulating would mean
    rewriting a 14 GB file to change 30 KB of it -- an extra 28 GB of I/O and a fresh
    way to fail after two hours of work.
    """
    policies = {}

    for layer_targets in targets.values():
        for target in layer_targets:
            policies[target.stem] = qwen_plan.POLICY_TYPE_NAMES[target.policy]

    return {
        "mila_config": json.dumps(config_metadata),
        "mila_quantization": "codebook",
        "mila_codebook_policies": json.dumps(policies),
    }


# ---------------------------------------------------------------------------
# Verifying a produced artifact
# ---------------------------------------------------------------------------

def verify(model_name, artifact_path, max_layers):
    """Audit a packed artifact against the checkpoint it came from.

    Separate from the run because the run takes hours and this takes minutes: an
    artifact that has been sitting on disk for a week can be re-checked without
    regenerating it. Three properties, each catching a different failure:

      completeness  every tensor the BF16 converter emits is present, with the
                    codebook rows replaced by their packed companions at the shapes
                    the policy implies.
      pass-through  the tensors quantization never touches -- norms, the convolution
                    splits, A_log, dt_bias, the table, the final norm -- are
                    byte-identical to the checkpoint. This is what proves the
                    rearranging transforms survived the packer.
      damage        the FP4-at-load tensors are present, BF16, and NOT identical to
                    the checkpoint. Without it a packer that silently skipped the
                    in-place damage passes everything above.
    """
    from safetensors import safe_open

    root = _resolve_checkpoint(model_name)
    checkpoint = ShardedCheckpoint(root)
    prefix = _text_prefix(checkpoint)
    geometry = resolve_qwen_geometry(
        json.loads((root / "config.json").read_text(encoding="utf-8")), max_layers)

    weights = CheckpointWeights(checkpoint)
    targets, layer_mappings, outer_mappings = build_plan(geometry, prefix)

    codebook_stems, fp4_names = {}, set()

    for layer_targets in targets.values():
        for target in layer_targets:
            if target.policy in qwen_plan.CODEBOOK_POLICIES:
                codebook_stems[target.stem] = target
            elif target.policy == "fp4":
                fp4_names.add(target.mila_name)

    failures = 0
    print(f"Artifact: {artifact_path}")
    print(f"Checkpoint: {root}\n")

    with safe_open(artifact_path, framework="pt") as handle:
        present = set(handle.keys())
        expected = set()

        for mappings in list(layer_mappings.values()) + [outer_mappings]:
            for mapping in mappings:
                stem = mapping.mila.removesuffix(".weight")

                if stem in codebook_stems:
                    expected.update(
                        codebook_record_shapes(codebook_stems[stem], weights.shape))
                else:
                    expected.add(mapping.mila)

        missing = expected - present
        extra = present - expected

        if missing or extra:
            print(f"  FAIL completeness: missing={sorted(missing)[:4]} "
                  f"unexpected={sorted(extra)[:4]}")
            failures += 1
        else:
            print(f"  OK   completeness: {len(present)} tensors, "
                  f"{len(codebook_stems)} packed Mila linears")

        wrong_shape = []

        for stem, target in codebook_stems.items():
            for name, (dtype, shape) in codebook_record_shapes(target, weights.shape).items():
                actual = handle.get_slice(name).get_shape()

                if tuple(actual) != tuple(shape):
                    wrong_shape.append(f"{name}: {tuple(actual)} != {tuple(shape)}")

        if wrong_shape:
            print(f"  FAIL packed shapes: {wrong_shape[:3]}")
            failures += 1
        else:
            print(f"  OK   packed shapes match what Linear allocates for each policy")

        quantized_stems = set(codebook_stems) | {name.removesuffix(".weight")
                                                 for name in fp4_names}
        checked, mismatched = 0, []

        for mappings in list(layer_mappings.values()) + [outer_mappings]:
            for mapping in mappings:
                if mapping.mila.removesuffix(".weight") in quantized_stems:
                    continue

                reference = apply_transform(
                    mapping, [checkpoint.tensor(name) for name in mapping.sources],
                    geometry["head_dim"]).to(torch.bfloat16)
                produced = handle.get_tensor(mapping.mila)
                checked += 1

                if not torch.equal(produced.view(torch.uint16),
                                   reference.view(torch.uint16)):
                    mismatched.append(mapping.mila)

        if mismatched:
            print(f"  FAIL pass-through: {len(mismatched)} differ, e.g. {mismatched[:3]}")
            failures += 1
        else:
            print(f"  OK   pass-through: {checked} untouched tensors are identical "
                  f"to the checkpoint")

        undamaged = []

        for name in sorted(fp4_names):
            produced = handle.get_tensor(name)
            mapping = next(mapping
                           for mappings in list(layer_mappings.values()) + [outer_mappings]
                           for mapping in mappings if mapping.mila == name)
            reference = apply_transform(
                mapping, [checkpoint.tensor(source) for source in mapping.sources],
                geometry["head_dim"]).to(torch.bfloat16)

            # lm_head is deliberately written through unchanged: nothing reads its
            # output inside the model, so damaging it would alter no calibration.
            if name != "lm_head.weight" and torch.equal(produced.view(torch.uint16),
                                                        reference.view(torch.uint16)):
                undamaged.append(name)

        if undamaged:
            print(f"  FAIL damage: FP4-at-load tensors were never damaged: "
                  f"{undamaged[:3]}")
            failures += 1
        else:
            print(f"  OK   damage: {len(fp4_names) - 1} FP4-at-load tensors carry "
                  f"quantization damage, lm_head written through by design")

    print(f"\nVerify: {'PASSED' if failures == 0 else f'FAILED ({failures})'}")

    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test(device):
    """Pack a small random Qwen model end to end and check three properties.

    Each catches a different way the packer can run and still be wrong:

      completeness  the artifact carries exactly the tensor set the BF16 converter
                    emits, modulo the companions a codebook tensor adds. A missing
                    tensor is a load failure much later, against a 14 GB file.
      fidelity      every BF16 pass-through tensor equals the transform applied to the
                    live layer, so the rearrangements (the q_proj de-interleave, the
                    DeltaNet row splits) came through the packer unchanged.
      damage        the packed tensors do NOT dequantize to the original checkpoint
                    weights. This is the negative control: every other check passes
                    just as well if quantization silently did nothing.

    The codebook records are additionally verified bit-for-bit against the weights the
    model carries, inside pack_codebook_tensor, on every tensor of every run.
    """
    import tempfile

    from safetensors import safe_open
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(0)

    # The config consumes full_attention_interval into layer_types and does not keep
    # it, so the interval is named here and verify_layer_kinds holds the two together.
    interval = 4

    # Eight layers for the same reason the reference driver's self-test uses eight:
    # both block kinds must run, and the 3:1 interleave puts the first full-attention
    # layer at index 3.
    #
    # Widths are the smallest that satisfy every group size, and FP4's 128 is the binding
    # one even though nothing here packs FP4: it applies to hidden_size (qkv and the head
    # read it) and to num_attention_heads * head_dim (o_proj reads that), and the export
    # pass that finishes this artifact quantizes one block per 128-column group. At the
    # earlier 64 the fitted source packed cleanly and could never be exported.
    config = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        full_attention_interval=interval,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        attn_output_gate=True,
        tie_word_embeddings=False)

    config._attn_implementation = "eager"

    model = Qwen3_5TextModel(config).to(device=device, dtype=torch.bfloat16).eval()

    weights = InMemoryShapes(model, TEXT_PREFIX)
    weights._state["lm_head.weight"] = torch.randn(
        config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16)

    original = {name: tensor.clone() for name, tensor in weights._state.items()}

    geometry = {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "full_attention_interval": interval,
        "linear_num_key_heads": config.linear_num_key_heads,
        "linear_num_value_heads": config.linear_num_value_heads,
        "linear_head_dim": config.linear_key_head_dim,
        "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
    }

    verify_layer_kinds(config, geometry)
    targets, layer_mappings, outer_mappings = build_plan(geometry, TEXT_PREFIX)

    directory = Path(tempfile.mkdtemp())
    path = directory / "self_test.safetensors"

    writer = StreamingSafetensorsWriter(path)
    declare_artifact(writer, targets, layer_mappings, outer_mappings, weights.shape)

    packer = QwenPacker(config, weights, torch.device(device), torch.bfloat16,
                        geometry, TEXT_PREFIX, writer)

    samples = [torch.randint(0, config.vocab_size, (32,)) for _ in range(2)]

    with writer:
        packer.run(samples, targets, layer_mappings, outer_mappings, progress=False)

    failures = 0

    # -- completeness --
    expected = set()

    for mappings in list(layer_mappings.values()) + [outer_mappings]:
        for mapping in mappings:
            expected.add(mapping.mila)

    codebook_stems = {target.stem
                      for layer_targets in targets.values()
                      for target in layer_targets
                      if target.policy in qwen_plan.CODEBOOK_POLICIES}

    # FP4 roles are packed too, so they are no longer pass-through -- they carry a
    # scale companion and, unlike the codebook roles, no table: FP4 is data-free.
    fp4_stems = {target.stem
                 for layer_targets in targets.values()
                 for target in layer_targets
                 if target.policy == "fp4"}

    packed_stems = codebook_stems | fp4_stems

    for stem in codebook_stems:
        expected.update({f"{stem}.weight", f"{stem}.weight_scale",
                         f"{stem}.weight_codebook"})

    for stem in fp4_stems:
        expected.update({f"{stem}.weight", f"{stem}.weight_scale"})

    # framework="pt": numpy has no bfloat16, so the pass-through tensors can only be
    # read back as torch ones.
    with safe_open(path, framework="pt") as handle:
        written = set(handle.keys())

        missing = expected - written
        extra = written - expected - {f"{stem}.weight_high_plane" for stem in codebook_stems}

        if missing or extra:
            print(f"  FAIL completeness: missing={sorted(missing)[:4]} "
                  f"unexpected={sorted(extra)[:4]}")
            failures += 1
        else:
            print(f"  OK   completeness: {len(written)} tensors, "
                  f"{len(codebook_stems)} packed Mila linears")

        # -- fidelity --
        #
        # Every pass-through tensor, layer ones included: the transforms worth checking
        # are all inside layers -- the q_proj de-interleave and the two DeltaNet row
        # splits have no counterpart in any other family. Comparable after the run only
        # because load_state_dict(assign=True) installs these very tensors as the
        # layer's parameters, so the in-place damage landed in `weights` itself. That
        # holds here and not for a real checkpoint, where the .to() copies to device.
        mismatched = []
        passthrough = [mapping
                       for mappings in list(layer_mappings.values()) + [outer_mappings]
                       for mapping in mappings
                       if mapping.mila.removesuffix(".weight") not in packed_stems]

        for mapping in passthrough:
            produced = handle.get_tensor(mapping.mila).cpu()
            reference = apply_transform(
                mapping, [weights.get(name) for name in mapping.sources],
                config.head_dim).to(torch.bfloat16).cpu()

            # Bit equality on the raw pattern: a pass-through tensor is copied, not
            # computed, so anything but exact agreement is a transform error.
            if not torch.equal(produced.view(torch.uint16), reference.view(torch.uint16)):
                mismatched.append(mapping.mila)

        if mismatched:
            print(f"  FAIL fidelity: {len(mismatched)} differ, e.g. {mismatched[:3]}")
            failures += 1
        else:
            print(f"  OK   fidelity: {len(passthrough)} pass-through tensors "
                  f"match the transform bit-for-bit")

        # -- damage (negative control) --
        unchanged = []

        for stem in sorted(codebook_stems)[:4]:
            table = handle.get_tensor(f"{stem}.weight_codebook")

            # A table whose entries all coincide encodes one value: the fit collapsed
            # and every weight in the tensor decodes to the same number.
            if torch.allclose(table, table[0]):
                unchanged.append(stem)

    # One target of each quantized class, because they are different code paths and a
    # silent no-op in either is invisible to every other check. The FP4 one matters as
    # much as the codebook one: its only purpose is to damage the activations the
    # layers downstream calibrate against.
    def first_piece(layer_index, policy):
        for target in targets[layer_index]:
            if target.policy == policy:
                return target.pieces[0].module

        raise AssertionError(f"layer {layer_index} has no {policy} target")

    probes = {"codebook": (0, first_piece(0, "cb8")),
              "fp4": (3, first_piece(3, "fp4"))}
    still = []
    moved = {}

    for label, (layer_index, module) in probes.items():
        key = f"{TEXT_PREFIX}layers.{layer_index}.{module}.weight"
        delta = (weights.get(key).float() - original[key].float()).abs().max().item()
        moved[label] = delta

        if delta == 0.0:
            still.append(f"{label}: layer {layer_index} {module}")

    if still or unchanged:
        print(f"  FAIL damage: quantization changed nothing for {still}"
              + (f", degenerate tables: {unchanged}" if unchanged else ""))
        failures += 1
    else:
        print("  OK   damage: " + ", ".join(
            f"{label} max|delta|={delta:.3e}" for label, delta in moved.items()))

    print(f"\nSelf-test: {'PASSED' if failures == 0 else f'FAILED ({failures})'}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true",
                        help="pack a small random model end to end and check it, then exit")
    parser.add_argument("--verify", metavar="PATH",
                        help="audit an already-packed artifact against the checkpoint, then exit")
    parser.add_argument("--model", help="HuggingFace model id or a local checkpoint directory")
    parser.add_argument("--output", help="output path for the quantized safetensors artifact")
    parser.add_argument("--calib-text", help="held-out text file for calibration")
    parser.add_argument("--samples", type=int, default=32,
                        help="calibration samples (default: 32)")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="tokens per calibration sample (default: 2048)")
    parser.add_argument("--max-layers", type=int, default=0,
                        help="pack only the first N layers -- a structural smoke test, not a model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    enforce_determinism()
    torch.manual_seed(GENERATOR_SEED)

    if args.self_test:
        sys.exit(self_test(args.device))

    if args.verify:
        if not args.model:
            parser.error("--verify needs --model: the audit is against the checkpoint")

        sys.exit(verify(args.model, Path(args.verify), args.max_layers))

    if not (args.model and args.output and args.calib_text):
        parser.error("--model, --output and --calib-text are required unless --self-test")

    pack(args.model, Path(args.output), args.calib_text, args.device,
         args.samples, args.seqlen, args.max_layers)
