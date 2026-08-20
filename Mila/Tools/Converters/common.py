# ============================================================================
# File: common.py
# Common utilities for weight conversion
# ============================================================================

import struct
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, BinaryIO
import json


# ============================================================================
# HuggingFace -> Mila tensor name map
# ============================================================================

@dataclass(frozen=True)
class TensorMapping:
    """
    One Mila tensor and the HuggingFace tensors it is built from.

    `sources` carries more than one name wherever Mila fuses projections that
    HuggingFace keeps separate; they concatenate along dim 0 in the order given,
    which is the layout the consuming component expects (SwiGLU reads
    fc_gate_up as [gate | up], GQA reads fc_qkv_proj as [Q | K | V]).

    `is_linear` marks a Linear weight rather than a norm or an embedding. The
    converter does not need the distinction -- it writes every tensor the same
    way -- but a quantizing packer does, since only Linear weights are weight
    quantization targets.

    `transform` names the rearrangement between the sources and the Mila tensor,
    for the families where concatenation is not the whole story; `rows` is the
    half-open row range the row-selecting transforms take. Both are applied by
    apply_transform, so a packer rearranges exactly as the converter does.
    """
    mila: str
    sources: Tuple[str, ...]
    is_linear: bool = False
    transform: str = 'concat'
    rows: Tuple[int, int] = None


# Emitted before the layers.
LLAMA_PRE_LAYER_TENSORS: Tuple[TensorMapping, ...] = (
    TensorMapping('temb.wte', ('model.embed_tokens.weight',)),
)

# Emitted once per layer, with '{i}' resolved by expand_llama_tensor_map.
#
# HF names the two norms 'layernorm' historically even though the implementation
# is LlamaRMSNorm. No transposition is needed anywhere: HF nn.Linear is already
# [out, in], which is Mila's convention. There is no positional embedding tensor
# (RoPE is computed) and no attention or MLP bias.
LLAMA_LAYER_TENSORS: Tuple[TensorMapping, ...] = (
    TensorMapping(
        'tf_layer_{i}.rmsn_1.weight',
        ('model.layers.{i}.input_layernorm.weight',)),
    TensorMapping(
        'tf_layer_{i}.fc_qkv_proj.weight',
        ('model.layers.{i}.self_attn.q_proj.weight',
         'model.layers.{i}.self_attn.k_proj.weight',
         'model.layers.{i}.self_attn.v_proj.weight'),
        is_linear=True),
    TensorMapping(
        'tf_layer_{i}.fc_out_proj.weight',
        ('model.layers.{i}.self_attn.o_proj.weight',),
        is_linear=True),
    TensorMapping(
        'tf_layer_{i}.rmsn_2.weight',
        ('model.layers.{i}.post_attention_layernorm.weight',)),
    TensorMapping(
        'tf_layer_{i}.fc_gate_up.weight',
        ('model.layers.{i}.mlp.gate_proj.weight',
         'model.layers.{i}.mlp.up_proj.weight'),
        is_linear=True),
    TensorMapping(
        'tf_layer_{i}.fc_down.weight',
        ('model.layers.{i}.mlp.down_proj.weight',),
        is_linear=True),
)

# Emitted after the layers. lm_head is written explicitly even when the model
# ties it to the embedding, so Mila's loader needs no tying logic; a caller must
# substitute 'model.embed_tokens.weight' when 'lm_head.weight' is absent from
# the state dict (Llama 3.2 1B/3B tie, Llama 3.1 8B does not).
LLAMA_POST_LAYER_TENSORS: Tuple[TensorMapping, ...] = (
    TensorMapping('rmsn_final.weight', ('model.norm.weight',)),
    TensorMapping('lm_head.weight', ('lm_head.weight',), is_linear=True),
)


def expand_llama_tensor_map(num_layers: int) -> List[TensorMapping]:
    """
    The full HF -> Mila map for a Llama model, with layer indices resolved.

    Sequence order is the order a converter emits, and is kept stable so that a
    rewritten converter can be checked byte-for-byte against an artifact it
    produced earlier. It is not semantic: the MILA container's index carries an
    explicit offset per tensor, and the safetensors writer orders the data
    region itself.
    """
    mappings: List[TensorMapping] = list(LLAMA_PRE_LAYER_TENSORS)

    for i in range(num_layers):
        for mapping in LLAMA_LAYER_TENSORS:
            mappings.append(TensorMapping(
                mapping.mila.format(i=i),
                tuple(source.format(i=i) for source in mapping.sources),
                mapping.is_linear))

    mappings.extend(LLAMA_POST_LAYER_TENSORS)

    return mappings


# ============================================================================
# HuggingFace -> Mila tensor name map: Qwen 3.8
# ============================================================================
#
# Two block kinds alternate (3 Gated DeltaNet : 1 full attention at the published
# `full_attention_interval` of 4), so the per-layer map is chosen per layer rather
# than repeated. '{p}' is the checkpoint's prefix for the text model, which differs
# between the text-only and the conditional-generation packagings; the caller
# resolves it once and every source name is formatted with it.
#
# Three of these mappings rearrange rather than concatenate, and each is a contract
# stated in the consuming block's header rather than a convenience here:
#
#  1. `q_proj` is DOUBLE width and interleaved per head, [q_h0 | gate_h0 | q_h1 | ...],
#     because the reference views it as [..., heads, 2 * head_dim] and chunks the last
#     axis. QwenAttentionBlock's fused projection is [query | gate | key | value] with
#     query and gate as CONTIGUOUS halves, so the query projection is de-interleaved
#     before the three sources concatenate.
#  2. `in_proj_qkv` fuses DeltaNet q, k and v into one [2*key_dim + value_dim] tensor.
#     Mila takes it as two projections -- [q|k] and [v] -- because one tensor cannot
#     carry two storage policies and the precision plan puts q/k a half step above v.
#  3. `conv1d.weight` is split on the same boundary, and drops its singleton axis
#     ([channels, 1, kernel] -> [channels, kernel]). The split is exact rather than an
#     approximation: the convolution is depthwise, so no channel ever reads another.

QWEN_PRE_LAYER_TENSORS: Tuple[TensorMapping, ...] = (
    TensorMapping('temb.wte', ('{p}embed_tokens.weight',)),
)

# The full-attention layers. QK-norm is not a config field in this architecture --
# it is unconditional in the reference and visible only in the tensor names.
QWEN_ATTENTION_LAYER_TENSORS: Tuple[TensorMapping, ...] = (
    TensorMapping(
        'tf_layer_{i}.input_norm.weight',
        ('{p}layers.{i}.input_layernorm.weight',)),
    TensorMapping(
        'tf_layer_{i}.fc_qkv_proj.weight',
        ('{p}layers.{i}.self_attn.q_proj.weight',
         '{p}layers.{i}.self_attn.k_proj.weight',
         '{p}layers.{i}.self_attn.v_proj.weight'),
        is_linear=True, transform='gated_qkv'),
    TensorMapping(
        'tf_layer_{i}.q_norm.weight',
        ('{p}layers.{i}.self_attn.q_norm.weight',)),
    TensorMapping(
        'tf_layer_{i}.k_norm.weight',
        ('{p}layers.{i}.self_attn.k_norm.weight',)),
    TensorMapping(
        'tf_layer_{i}.fc_o_proj.weight',
        ('{p}layers.{i}.self_attn.o_proj.weight',),
        is_linear=True),
)

# The Gated DeltaNet layers. A_log and dt_bias belong to the delta rule itself: it
# derives the decay and the forget gate from them rather than taking g and beta from
# the block, so they load into the GatedDeltaRule component, not into a projection.
QWEN_DELTANET_LAYER_TENSORS: Tuple[TensorMapping, ...] = (
    TensorMapping(
        'tf_layer_{i}.input_norm.weight',
        ('{p}layers.{i}.input_layernorm.weight',)),
    TensorMapping(
        'tf_layer_{i}.fc_in_proj_qk.weight',
        ('{p}layers.{i}.linear_attn.in_proj_qkv.weight',),
        is_linear=True, transform='rows'),
    TensorMapping(
        'tf_layer_{i}.fc_in_proj_v.weight',
        ('{p}layers.{i}.linear_attn.in_proj_qkv.weight',),
        is_linear=True, transform='rows'),
    TensorMapping(
        'tf_layer_{i}.fc_in_proj_z.weight',
        ('{p}layers.{i}.linear_attn.in_proj_z.weight',),
        is_linear=True),
    TensorMapping(
        'tf_layer_{i}.fc_in_proj_a.weight',
        ('{p}layers.{i}.linear_attn.in_proj_a.weight',),
        is_linear=True),
    TensorMapping(
        'tf_layer_{i}.fc_in_proj_b.weight',
        ('{p}layers.{i}.linear_attn.in_proj_b.weight',),
        is_linear=True),
    TensorMapping(
        'tf_layer_{i}.conv_qk.weight',
        ('{p}layers.{i}.linear_attn.conv1d.weight',),
        transform='conv_rows'),
    TensorMapping(
        'tf_layer_{i}.conv_v.weight',
        ('{p}layers.{i}.linear_attn.conv1d.weight',),
        transform='conv_rows'),
    TensorMapping(
        'tf_layer_{i}.delta_rule.A_log',
        ('{p}layers.{i}.linear_attn.A_log',)),
    TensorMapping(
        'tf_layer_{i}.delta_rule.dt_bias',
        ('{p}layers.{i}.linear_attn.dt_bias',)),
    TensorMapping(
        'tf_layer_{i}.norm_gate.weight',
        ('{p}layers.{i}.linear_attn.norm.weight',)),
    TensorMapping(
        'tf_layer_{i}.fc_out_proj.weight',
        ('{p}layers.{i}.linear_attn.out_proj.weight',),
        is_linear=True),
)

# The SwiGLU feed-forward half, identical on both block kinds -- `intermediate_size`
# is one scalar in the config and is confirmed uniform in every shard header.
QWEN_FEEDFORWARD_TENSORS: Tuple[TensorMapping, ...] = (
    TensorMapping(
        'tf_layer_{i}.post_attn_norm.weight',
        ('{p}layers.{i}.post_attention_layernorm.weight',)),
    TensorMapping(
        'tf_layer_{i}.fc_gate_up.weight',
        ('{p}layers.{i}.mlp.gate_proj.weight',
         '{p}layers.{i}.mlp.up_proj.weight'),
        is_linear=True),
    TensorMapping(
        'tf_layer_{i}.fc_down.weight',
        ('{p}layers.{i}.mlp.down_proj.weight',),
        is_linear=True),
)

# lm_head is a top-level tensor rather than a member of the text model, and Qwen 3.8
# is UNTIED, so it is always present and always written.
QWEN_POST_LAYER_TENSORS: Tuple[TensorMapping, ...] = (
    TensorMapping('rmsn_final.weight', ('{p}norm.weight',)),
    TensorMapping('lm_head.weight', ('lm_head.weight',), is_linear=True),
)


def expand_qwen_tensor_map(num_layers: int, full_attention_interval: int,
                           key_dim: int, value_dim: int,
                           prefix: str = 'model.') -> List[TensorMapping]:
    """
    The full HF -> Mila map for a Qwen 3.8 model, with layer indices, the checkpoint
    prefix, and the DeltaNet row splits resolved.

    The row ranges are computed here rather than declared, so the split follows the
    geometry the checkpoint actually has: the reference splits in_proj_qkv as
    [key_dim, key_dim, value_dim], which puts the [q|k] pair in the first 2*key_dim
    rows and v in the rest.
    """
    qk_rows = (0, 2 * key_dim)
    v_rows = (2 * key_dim, 2 * key_dim + value_dim)

    split_rows = {
        'fc_in_proj_qk': qk_rows,
        'conv_qk': qk_rows,
        'fc_in_proj_v': v_rows,
        'conv_v': v_rows,
    }

    def resolve(mapping: TensorMapping, i: int) -> TensorMapping:
        mila = mapping.mila.format(i=i)
        rows = mapping.rows

        if mapping.transform in ('rows', 'conv_rows'):
            rows = split_rows[mila.split('.')[1]]

        return TensorMapping(
            mila,
            tuple(source.format(i=i, p=prefix) for source in mapping.sources),
            mapping.is_linear,
            mapping.transform,
            rows)

    mappings: List[TensorMapping] = [resolve(m, 0) for m in QWEN_PRE_LAYER_TENSORS]

    for i in range(num_layers):
        is_full_attention = ((i + 1) % full_attention_interval) == 0
        mixer = QWEN_ATTENTION_LAYER_TENSORS if is_full_attention else QWEN_DELTANET_LAYER_TENSORS

        for mapping in list(mixer) + list(QWEN_FEEDFORWARD_TENSORS):
            mappings.append(resolve(mapping, i))

    mappings.extend(resolve(m, 0) for m in QWEN_POST_LAYER_TENSORS)

    return mappings


def apply_transform(mapping: TensorMapping, sources: List, head_dim: int = 0):
    """
    Build one Mila tensor from its HuggingFace sources.

    `sources` are torch tensors in map order. `head_dim` is required only by the
    'gated_qkv' transform, which is the one rearrangement that needs to know how the
    rows are grouped.
    """
    import torch

    if mapping.transform == 'concat':
        return sources[0] if len(sources) == 1 else torch.cat(sources, dim=0)

    if mapping.transform == 'rows':
        start, stop = mapping.rows

        return sources[0][start:stop]

    if mapping.transform == 'conv_rows':
        start, stop = mapping.rows

        # [channels, 1, kernel] -> [channels, kernel]: the singleton is torch's
        # depthwise-Conv1d input-group axis and has no counterpart in CausalConv1d.
        return sources[0][start:stop].reshape(stop - start, sources[0].shape[-1])

    if mapping.transform == 'gated_qkv':
        query_gate, key, value = sources

        if head_dim <= 0:
            raise ValueError("the 'gated_qkv' transform needs head_dim")

        # Rows run [head][query|gate][within-head]; the two halves are wanted whole.
        in_features = query_gate.shape[1]
        per_head = query_gate.reshape(-1, 2, head_dim, in_features)
        query = per_head[:, 0].reshape(-1, in_features)
        gate = per_head[:, 1].reshape(-1, in_features)

        return torch.cat([query, gate, key, value], dim=0)

    raise ValueError(f"unknown transform '{mapping.transform}' for {mapping.mila}")


class MilaWeightWriter:
    """
    Writes weights in Mila's binary format.
    
    Format:
        [Header]
        - magic: uint32 (0x4D494C41 = "MILA")
        - version: uint32 (1)
        - num_tensors: uint32
        - metadata_size: uint32
        [Metadata JSON]
        - Architecture config as JSON string
        [Tensor Index]
        - For each tensor:
            - name_length: uint32
            - name: char[name_length]
            - dtype: uint32 (0=float32, 1=float16, 2=bfloat16, 3=int32)
            - ndim: uint32
            - shape: uint32[ndim]
            - offset: uint64 (byte offset to tensor data)
            - nbytes: uint64
        [Tensor Data]
        - Raw tensor bytes (all tensors concatenated)
    """
    
    MAGIC = 0x4D494C41  # "MILA"
    VERSION = 1
    
    DTYPE_MAP = {
        'float32': 0,
        'float16': 1,
        'bfloat16': 2,
        'int32': 3,
    }
    
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.tensors: List[Tuple[str, np.ndarray]] = []
        self.metadata: Dict = {}
    
    def set_metadata(self, metadata: Dict):
        """Set architecture metadata (config, hyperparameters, etc.)"""
        self.metadata = metadata
    
    def add_tensor(self, name: str, data: np.ndarray):
        """Add a tensor to be written"""
        self.tensors.append((name, data))
    
    def write(self):
        """Write all tensors to binary file"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'wb') as f:
            # Write header
            self._write_header(f)
            
            # Write metadata
            metadata_bytes = json.dumps(self.metadata, indent=2).encode('utf-8')
            f.write(struct.pack('I', len(metadata_bytes)))
            f.write(metadata_bytes)
            
            # Calculate tensor offsets
            index_start = f.tell()
            index_size = sum(
                4 + len(name) + 4 + 4 + 4*len(data.shape) + 8 + 8
                for name, data in self.tensors
            )
            data_start = index_start + index_size
            
            # Write tensor index
            current_offset = data_start
            tensor_info = []
            for name, data in self.tensors:
                dtype_code = self._get_dtype_code(data.dtype)
                nbytes = data.nbytes
                
                # Write index entry
                f.write(struct.pack('I', len(name)))
                f.write(name.encode('utf-8'))
                f.write(struct.pack('I', dtype_code))
                f.write(struct.pack('I', len(data.shape)))
                f.write(struct.pack('I' * len(data.shape), *data.shape))
                f.write(struct.pack('Q', current_offset))
                f.write(struct.pack('Q', nbytes))
                
                tensor_info.append((current_offset, data))
                current_offset += nbytes
            
            # Write tensor data
            for offset, data in tensor_info:
                assert f.tell() == offset, f"Offset mismatch: expected {offset}, got {f.tell()}"
                data.tofile(f)
        
        print(f"Wrote {len(self.tensors)} tensors to {self.output_path}")
        print(f"File size: {self.output_path.stat().st_size / 1024**2:.2f} MB")
    
    def _write_header(self, f: BinaryIO):
        """Write file header"""
        f.write(struct.pack('I', self.MAGIC))
        f.write(struct.pack('I', self.VERSION))
        f.write(struct.pack('I', len(self.tensors)))
    
    def _get_dtype_code(self, dtype) -> int:
        """Convert numpy dtype to Mila dtype code"""
        dtype_str = str(dtype)
        if 'float32' in dtype_str:
            return self.DTYPE_MAP['float32']
        elif 'float16' in dtype_str:
            return self.DTYPE_MAP['float16']
        elif dtype == np.uint16:  # bfloat16 stored as uint16
            return self.DTYPE_MAP['bfloat16']
        elif 'int32' in dtype_str:
            return self.DTYPE_MAP['int32']
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")


class MilaStreamingWeightWriter:
    """
    Writes the same container as MilaWeightWriter without ever holding the model.

    MilaWeightWriter keeps every tensor in a list until write(); at 27B that is 54 GB
    of host memory against 32 GB of RAM, so a large model has to be written a tensor
    at a time. The index sits at the FRONT of the file and carries an explicit offset
    per tensor, so every offset must be known before the first data byte is written:
    declare() takes the name, dtype and shape a tensor WILL have, and write() supplies
    the bytes, in declaration order.

    Declared shapes are checked against what arrives. That check is the reason this
    class validates rather than trusts: a wrong shape would otherwise be discovered by
    Mila as an unexplained offset, hours into a conversion.
    """

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.entries: List[Tuple[str, str, Tuple[int, ...]]] = []
        self.metadata: Dict = {}
        self._file: BinaryIO = None
        self._offsets: List[int] = []
        self._written = 0

    def set_metadata(self, metadata: Dict):
        """Set architecture metadata (config, hyperparameters, etc.)"""
        self.metadata = metadata

    def declare(self, name: str, dtype: str, shape: Tuple[int, ...]):
        """Declare a tensor. Every declaration must happen before the first write."""
        if self._file is not None:
            raise RuntimeError('MilaStreamingWeightWriter: declare() after the index was written')

        if dtype not in MilaWeightWriter.DTYPE_MAP:
            raise ValueError(f'Unsupported dtype: {dtype}')

        self.entries.append((name, dtype, tuple(int(d) for d in shape)))

    def total_data_bytes(self) -> int:
        """Payload size the declarations imply -- known before anything is written."""
        return sum(_dtype_size(dtype) * _element_count(shape)
                   for _, dtype, shape in self.entries)

    def __enter__(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, 'wb')

        f = self._file
        f.write(struct.pack('I', MilaWeightWriter.MAGIC))
        f.write(struct.pack('I', MilaWeightWriter.VERSION))
        f.write(struct.pack('I', len(self.entries)))

        metadata_bytes = json.dumps(self.metadata, indent=2).encode('utf-8')
        f.write(struct.pack('I', len(metadata_bytes)))
        f.write(metadata_bytes)

        index_start = f.tell()
        index_size = sum(4 + len(name) + 4 + 4 + 4 * len(shape) + 8 + 8
                         for name, _, shape in self.entries)
        offset = index_start + index_size

        for name, dtype, shape in self.entries:
            nbytes = _dtype_size(dtype) * _element_count(shape)

            f.write(struct.pack('I', len(name)))
            f.write(name.encode('utf-8'))
            f.write(struct.pack('I', MilaWeightWriter.DTYPE_MAP[dtype]))
            f.write(struct.pack('I', len(shape)))
            f.write(struct.pack('I' * len(shape), *shape))
            f.write(struct.pack('Q', offset))
            f.write(struct.pack('Q', nbytes))

            self._offsets.append(offset)
            offset += nbytes

        return self

    def write(self, name: str, data: np.ndarray):
        """Write one declared tensor's bytes. Order and shape must match declare()."""
        if self._written >= len(self.entries):
            raise RuntimeError(f'MilaStreamingWeightWriter: unexpected tensor {name}')

        expected_name, expected_dtype, expected_shape = self.entries[self._written]

        if name != expected_name:
            raise RuntimeError(f'MilaStreamingWeightWriter: expected {expected_name}, got {name}')

        if tuple(data.shape) != expected_shape:
            raise RuntimeError(
                f'{name}: declared shape {expected_shape}, got {tuple(data.shape)}')

        if data.dtype != _expected_numpy_dtype(expected_dtype):
            raise RuntimeError(f'{name}: declared {expected_dtype}, got {data.dtype}')

        if self._file.tell() != self._offsets[self._written]:
            raise RuntimeError(
                f'{name}: offset mismatch -- expected {self._offsets[self._written]}, '
                f'at {self._file.tell()}')

        data.tofile(self._file)
        self._written += 1

    def __exit__(self, exc_type, exc, traceback):
        self._file.close()
        self._file = None

        if exc_type is not None:
            return False

        if self._written != len(self.entries):
            raise RuntimeError(
                f'MilaStreamingWeightWriter: declared {len(self.entries)} tensors, '
                f'wrote {self._written}')

        print(f'Wrote {self._written} tensors to {self.output_path}')
        print(f'File size: {self.output_path.stat().st_size / 1024**3:.2f} GiB')

        return False


def _element_count(shape: Tuple[int, ...]) -> int:
    count = 1

    for dim in shape:
        count *= int(dim)

    return count


def _dtype_size(dtype: str) -> int:
    return 4 if dtype in ('float32', 'int32') else 2


def _expected_numpy_dtype(dtype: str):
    """The numpy dtype a Mila dtype arrives as. BF16 has no numpy type and travels
    as a uint16 view of its bit pattern, which is what Mila's loader reads."""
    return {
        'float32': np.float32,
        'float16': np.float16,
        'bfloat16': np.uint16,
        'int32': np.int32,
    }[dtype]


def convert_dtype(tensor: np.ndarray, target_dtype: str) -> np.ndarray:
    """Convert tensor to target dtype"""
    if target_dtype == 'float32':
        return tensor.astype(np.float32)
    elif target_dtype == 'float16':
        return tensor.astype(np.float16)
    elif target_dtype == 'bfloat16':
        # Convert to bfloat16 (stored as uint16)
        import torch
        t = torch.from_numpy(tensor).to(torch.bfloat16)
        return t.view(torch.uint16).numpy()
    else:
        return tensor
