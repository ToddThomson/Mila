import struct
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
print(f"Prompt: {prompt!r}")
print(f"Token ids: {input_ids[0].tolist()}")
print(f"Tokens: {[tokenizer.decode([t]) for t in input_ids[0].tolist()]}")

# Stores lightweight statistics for debugging.
captured = {}
# Small slices of tensors (rows x first N elements) for quick elementwise inspection.
captured_slices = {}

# how many elements to store/print (columns)
_FIRST_N = 16
# how many rows to show at most
_MAX_ROWS = 4

def stats_last_token(t: torch.Tensor):
    """
    Compute min/max/mean/std for the last token along the sequence dimension.
    Expects t shaped [B, T, D] or [B, D] (single token).
    """
    if t.dim() == 2:
        last = t[0, :]
    else:
        last = t[0, -1, :]
    return {
        "min": last.min().item(),
        "max": last.max().item(),
        "mean": last.mean().item(),
        "std": last.std().item(),
    }

def make_hook(name):
    def fn(module, input, output):
        t = output if isinstance(output, torch.Tensor) else output[0]
        captured[name] = stats_last_token(t)
    return fn

def make_store_first_n_hook(name, n=_FIRST_N):
    """
    Legacy: store first `n` elements of the last-token vector as a Python list
    (detached CPU floats) under captured_slices[name].
    Kept for compatibility with existing registrations that expect last-token only.
    """
    def fn(module, input, output):
        t = output if isinstance(output, torch.Tensor) else output[0]
        if t.dim() == 2:
            last = t[0, :]
        else:
            last = t[0, -1, :]
        vals = last.detach().cpu().tolist()
        captured_slices[name] = [float(x) for x in vals[:n]]
    return fn

def make_store_rows_hook(name, rows=_MAX_ROWS, cols=_FIRST_N):
    """
    Store up to `rows` rows (sequence dimension) and up to `cols` columns (features)
    for the batch item 0. Result is a list-of-lists of floats stored under captured_slices[name].
    """
    def fn(module, input, output):
        t = output if isinstance(output, torch.Tensor) else output[0]
        rows_list = []
        if t.dim() == 2:
            # single-token / single-row case
            row0 = t[0, :cols].detach().cpu().tolist()
            rows_list.append([float(x) for x in row0])
        else:
            seq_len = t.size(1)
            use_rows = min(rows, seq_len)
            for r in range(use_rows):
                vals = t[0, r, :cols].detach().cpu().tolist()
                rows_list.append([float(x) for x in vals])
        captured_slices[name] = rows_list
    return fn

def make_qk_pre_hook(layer_index, kind):
    """
    Hook q_proj/k_proj to capture pre-RoPE Q/K last-token stats.
    (Do not keep full tensors in memory — we only need lightweight stats.)
    """
    def fn(module, input, output):
        t = output if isinstance(output, torch.Tensor) else output[0]
        captured[f"layer_{layer_index}.{kind}_pre_rope"] = stats_last_token(t)
    return fn

# ---------------------------------------------------------------------------
# FNV-1a checksum — exact port of Mila's print_stats debug helper.
# Processes float32 values as 4 little-endian bytes per element.
# ---------------------------------------------------------------------------

def fnv1a_checksum_last_token(t: torch.Tensor) -> int:
    """
    Compute the 64-bit FNV-1a checksum over the last-token vector of t.
    Matches Mila's print_stats bit-for-bit: iterates float32 values in
    sequence order, processing 4 little-endian bytes per element.
    """
    FNV_OFFSET = 1469598103934665603
    FNV_PRIME = 1099511628211
    MASK64 = 0xFFFFFFFFFFFFFFFF

    if t.dim() == 2:
        last = t[0, :]
    else:
        last = t[0, -1, :]

    checksum = FNV_OFFSET

    for val in last.detach().cpu().to(torch.float32):
        bits = struct.unpack('<I', struct.pack('<f', float(val)))[0]

        for byte_idx in range(4):
            b = (bits >> (byte_idx * 8)) & 0xFF
            checksum = ((checksum ^ b) * FNV_PRIME) & MASK64

    return checksum

def stats_and_checksum_last_token(t: torch.Tensor) -> dict:
    """
    Compute min/max/mean/std/checksum for the last-token vector of t.
    std uses the population formula (divide by N, not N-1) to match
    Mila's print_stats which computes sqrt(var_sum / D).
    """
    if t.dim() == 2:
        last = t[0, :]
    else:
        last = t[0, -1, :]

    last_f32 = last.detach().cpu().to(torch.float32)
    n = last_f32.numel()
    mean_val = last_f32.sum().item() / n
    var_val = ((last_f32 - mean_val) ** 2).sum().item() / n

    return {
        "min": last_f32.min().item(),
        "max": last_f32.max().item(),
        "mean": mean_val,
        "std": var_val ** 0.5,
        "checksum": fnv1a_checksum_last_token(t),
    }

def _fmt_mila_stats(stats: dict) -> str:
    """Format a stats dict to match Mila's print_stats output for direct line comparison."""
    return (
        f"min={stats['min']:.6f} max={stats['max']:.6f} "
        f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
        f"checksum=0x{stats['checksum']:016x}"
    )

def make_attn_output_pre_hook(layer_index):
    """
    Capture the attention output before o_proj via a forward pre-hook.
    The input[0] to o_proj is the result of scaled dot-product attention
    reshaped to [B, T, model_dim], which corresponds directly to the
    return value of Mila's attn_->prefill().
    """
    def fn(module, input):
        t = input[0] if isinstance(input, (tuple, list)) else input

        captured[f"layer_{layer_index}.attn_out"] = stats_and_checksum_last_token(t)

        rows_list = []
        seq_len = t.size(1) if t.dim() == 3 else 1
        use_rows = min(_MAX_ROWS, seq_len)

        for r in range(use_rows):
            if t.dim() == 3:
                vals = t[0, r, :_FIRST_N].detach().cpu().tolist()
            else:
                vals = t[0, :_FIRST_N].detach().cpu().tolist()
            rows_list.append([float(x) for x in vals])

        captured_slices[f"layer_{layer_index}.attn_out_rows_first{_FIRST_N}"] = rows_list
    return fn

# Formatting helpers: show numeric outputs with 6 significant digits
def _fmt_num(x: float) -> str:
    return f"{x:.6g}"

def _fmt_stats(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return "{" + ", ".join(f"{k}: {_fmt_num(v)}" for k, v in obj.items()) + "}"
    return str(obj)

def _print_table(name, rows_list):
    """
    Print a simple ASCII table for rows_list (list of lists of floats).
    Shows up to _MAX_ROWS rows and up to _FIRST_N columns.
    This implementation computes column widths from the formatted strings so
    negative values (leading '-') are accounted for correctly.
    """
    if not rows_list:
        print(f"{name}: (no rows captured)")
        return

    # determine column count from captured rows (limit to _FIRST_N)
    cols = max(len(r) for r in rows_list)
    cols = min(cols, _FIRST_N)

    # prepare formatted rows for display (only first _MAX_ROWS)
    formatted_rows = []
    display_rows = rows_list[:_MAX_ROWS]
    for row in display_rows:
        formatted = []
        for j in range(cols):
            val = row[j] if j < len(row) else None
            formatted.append(_fmt_num(val) if val is not None else "")
        formatted_rows.append(formatted)

    # compute column widths considering header labels and formatted numeric strings (includes minus sign)
    col_widths = []
    for j in range(cols):
        header_label = f"C{j}"
        max_cell = max((len(r[j]) for r in formatted_rows), default=0)
        col_widths.append(max(len(header_label), max_cell))

    # build header with column widths
    header_cols = " | ".join(f"{f'C{j}':>{col_widths[j]}}" for j in range(cols))
    header = f"Row | {header_cols}"
    sep = "-" * len(header)

    print(f"\n{name}")
    print(sep)
    print(header)
    print(sep)

    for i, row in enumerate(formatted_rows):
        row_str = " | ".join(f"{row[j]:>{col_widths[j]}}" for j in range(cols))
        print(f"{i:3} | {row_str}")

    if len(rows_list) > _MAX_ROWS:
        print(f"... ({len(rows_list)} rows captured, showing first {_MAX_ROWS})")

hooks = []

# Hook token embeddings (if present) to capture the embedding outputs and first rows.
if hasattr(model.model, "embed_tokens"):
    hooks.append(model.model.embed_tokens.register_forward_hook(
        make_hook("token_embeds")))
    hooks.append(model.model.embed_tokens.register_forward_hook(
        make_store_rows_hook(f"token_embeds_rows_first{_FIRST_N}", rows=_MAX_ROWS, cols=_FIRST_N)))

# Only attach layer hooks for layer 0 (user requested only layer 0).
for i, layer in enumerate(model.model.layers):
    if i != 0:
        continue

    # existing checkpoints (stats)
    if hasattr(layer, "input_layernorm"):
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_hook(f"layer_{i}.rmsn_1")))
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_store_rows_hook(f"layer_{i}.rmsn_1_rows_first{_FIRST_N}", rows=_MAX_ROWS, cols=_FIRST_N)))

    # capture pre-RoPE Q/K by hooking the projection modules that produce them
    if hasattr(layer.self_attn, "q_proj"):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(
            make_qk_pre_hook(i, "q")))
        hooks.append(layer.self_attn.q_proj.register_forward_hook(
            make_store_rows_hook(f"layer_{i}.q_pre_rope_rows_first{_FIRST_N}", rows=_MAX_ROWS, cols=_FIRST_N)))
    if hasattr(layer.self_attn, "k_proj"):
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            make_qk_pre_hook(i, "k")))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            make_store_rows_hook(f"layer_{i}.k_pre_rope_rows_first{_FIRST_N}", rows=_MAX_ROWS, cols=_FIRST_N)))

    # other existing projections and checkpoints
    if hasattr(layer.self_attn, "v_proj"):
        hooks.append(layer.self_attn.v_proj.register_forward_hook(
            make_hook(f"layer_{i}.v_proj")))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(
            make_store_rows_hook(f"layer_{i}.v_proj_rows_first{_FIRST_N}", rows=_MAX_ROWS, cols=_FIRST_N)))

    if hasattr(layer.self_attn, "o_proj"):
        # Pre-hook captures attention output before o_proj — matches Mila attn_->prefill() return.
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
            make_attn_output_pre_hook(i)))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            make_hook(f"layer_{i}.fc_out_proj")))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            make_store_rows_hook(f"layer_{i}.fc_out_proj_rows_first{_FIRST_N}", rows=_MAX_ROWS, cols=_FIRST_N)))

    if hasattr(layer, "post_attention_layernorm"):
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            make_hook(f"layer_{i}.rmsn_2")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            make_store_rows_hook(f"layer_{i}.rmsn_2_rows_first{_FIRST_N}", rows=_MAX_ROWS, cols=_FIRST_N)))

    if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj"):
        hooks.append(layer.mlp.gate_proj.register_forward_hook(
            make_hook(f"layer_{i}.gate_proj")))
        hooks.append(layer.mlp.gate_proj.register_forward_hook(
            make_store_rows_hook(f"layer_{i}.gate_proj_rows_first{_FIRST_N}", rows=_MAX_ROWS, cols=_FIRST_N)))
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "up_proj"):
        hooks.append(layer.mlp.up_proj.register_forward_hook(
            make_hook(f"layer_{i}.up_proj")))
        hooks.append(layer.mlp.up_proj.register_forward_hook(
            make_store_rows_hook(f"layer_{i}.up_proj_rows_first{_FIRST_N}", rows=_MAX_ROWS, cols=_FIRST_N)))
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        hooks.append(layer.mlp.down_proj.register_forward_hook(
            make_hook(f"layer_{i}.fc_down")))
        hooks.append(layer.mlp.down_proj.register_forward_hook(
            make_store_rows_hook(f"layer_{i}.fc_down_rows_first{_FIRST_N}", rows=_MAX_ROWS, cols=_FIRST_N)))

    hooks.append(layer.register_forward_hook(
        make_hook(f"layer_{i}.block_out")))
    hooks.append(layer.register_forward_hook(
        make_store_rows_hook(f"layer_{i}.block_out_rows_first{_FIRST_N}", rows=_MAX_ROWS, cols=_FIRST_N)))

with torch.no_grad():
    out = model(input_ids)
    logits = out.logits[0, -1]
    top5 = torch.topk(logits, 5)

    print("\nTop 5 predictions:")
    for v, idx in zip(top5.values, top5.indices):
        print(f"  {tokenizer.decode([idx.item()])!r:15} {_fmt_num(v.item())}")

for h in hooks:
    h.remove()

# Print concise summary for token embeddings and layer 0 only (stats)
print("\n--- Token embedding checkpoint (last token, min/max/mean/std) ---")
v = captured.get("token_embeds", "not captured")
print(f"  token_embeds: {_fmt_stats(v)}")

print("\n--- Layer 0 checkpoints (last token, min/max/mean/std) ---")
for key in [
    "layer_0.rmsn_1",
    "layer_0.q_pre_rope",
    "layer_0.k_pre_rope",
    "layer_0.v_proj",
    "layer_0.fc_out_proj",
    "layer_0.rmsn_2",
    "layer_0.gate_proj",
    "layer_0.up_proj",
    "layer_0.fc_down",
    "layer_0.block_out",
]:
    val = captured.get(key, "not captured")
    print(f"  {key}: {_fmt_stats(val)}")

# ---------------------------------------------------------------------------
# Attention output comparison — matches Mila's print_stats("attn_out", attn_out).
# Compare this block directly against the Mila console output for layer 0.
# ---------------------------------------------------------------------------
print("\n--- Attention output comparison (layer 0, before o_proj) ---")
print("  Mila label: attn_out")
attn_stats = captured.get("layer_0.attn_out")

if attn_stats:
    print(f"  layer_0.attn_out: {_fmt_mila_stats(attn_stats)}")
else:
    print("  layer_0.attn_out: not captured")

# Print the stored rows (up to first _MAX_ROWS rows) in table form
print(f"\n--- First {_FIRST_N} elements (up to first {_MAX_ROWS} rows) for selected checkpoints ---")
for k, v in captured_slices.items():
    # normalize single-row legacy lists into list-of-lists
    if v and isinstance(v[0], float):
        rows = [v]
    else:
        rows = v
    _print_table(k, rows)