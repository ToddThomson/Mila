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

# ── Constants ────────────────────────────────────────────────────────────────

_FIRST_N = 16
_MAX_ROWS = 4

# ── Capture storage ──────────────────────────────────────────────────────────

captured        = {}
captured_slices = {}

# ── Stats helpers ────────────────────────────────────────────────────────────

def fnv1a_checksum_last_token(t: torch.Tensor) -> int:
    FNV_OFFSET = 1469598103934665603
    FNV_PRIME  = 1099511628211
    MASK64     = 0xFFFFFFFFFFFFFFFF

    last     = t[0, :] if t.dim() == 2 else t[0, -1, :]
    checksum = FNV_OFFSET

    for val in last.detach().cpu().to(torch.float32):
        bits = struct.unpack('<I', struct.pack('<f', float(val)))[0]
        for byte_idx in range(4):
            b = (bits >> (byte_idx * 8)) & 0xFF
            checksum = ((checksum ^ b) * FNV_PRIME) & MASK64

    return checksum

def stats_and_checksum_last_token(t: torch.Tensor) -> dict:
    """
    min/max/mean/std/checksum for the last-token vector.
    std uses population formula (divide by N) to match Mila's print_stats.
    """
    last     = t[0, :] if t.dim() == 2 else t[0, -1, :]
    last_f32 = last.detach().cpu().to(torch.float32)
    n        = last_f32.numel()
    mean_val = last_f32.sum().item() / n
    var_val  = ((last_f32 - mean_val) ** 2).sum().item() / n

    return {
        "min":      last_f32.min().item(),
        "max":      last_f32.max().item(),
        "mean":     mean_val,
        "std":      var_val ** 0.5,
        "checksum": fnv1a_checksum_last_token(t),
    }

def _fmt_mila_stats(stats: dict) -> str:
    return (
        f"min={stats['min']:.6f} max={stats['max']:.6f} "
        f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
        f"checksum=0x{stats['checksum']:016x}"
    )

# ── Hook factories ───────────────────────────────────────────────────────────

def make_stats_hook(name):
    def fn(module, input, output):
        t = output if isinstance(output, torch.Tensor) else output[0]
        captured[name] = stats_and_checksum_last_token(t)
    return fn

def make_store_rows_hook(name, rows=_MAX_ROWS, cols=_FIRST_N):
    def fn(module, input, output):
        t = output if isinstance(output, torch.Tensor) else output[0]
        rows_list = []
        if t.dim() == 2:
            rows_list.append([float(x) for x in t[0, :cols].detach().cpu().tolist()])
        else:
            use_rows = min(rows, t.size(1))
            for r in range(use_rows):
                rows_list.append([float(x) for x in t[0, r, :cols].detach().cpu().tolist()])
        captured_slices[name] = rows_list
    return fn

def make_attn_output_pre_hook(layer_index):
    """Pre-hook on o_proj — matches Mila's attn_->decode() return value."""
    def fn(module, input):
        t = input[0] if isinstance(input, (tuple, list)) else input
        captured[f"layer_{layer_index}.attn_out"] = stats_and_checksum_last_token(t)
        rows_list = []
        use_rows  = min(_MAX_ROWS, t.size(1) if t.dim() == 3 else 1)
        for r in range(use_rows):
            vals = t[0, r, :_FIRST_N] if t.dim() == 3 else t[0, :_FIRST_N]
            rows_list.append([float(x) for x in vals.detach().cpu().tolist()])
        captured_slices[f"layer_{layer_index}.attn_out_rows_first{_FIRST_N}"] = rows_list
    return fn

# ── Post-RoPE K/Q capture via apply_rotary_pos_emb monkey-patch ─────────────

def install_rope_capture():
    """
    Monkey-patch apply_rotary_pos_emb in the Llama modeling module to capture
    post-RoPE Q and K for layer 0. Returns a restore function.
    """
    import transformers.models.llama.modeling_llama as llama_mod

    original_rope = llama_mod.apply_rotary_pos_emb

    def capturing_rope(q, k, cos, sin ):
        q_rot, k_rot = original_rope(q, k, cos, sin )

        # q_rot: [B, n_heads,    T, head_dim]
        # k_rot: [B, n_kv_heads, T, head_dim]
        # Capture last token, head 0, first _FIRST_N elements
        captured["layer_0.k_post_rope"] = k_rot[0, 0, -1, :_FIRST_N].detach().cpu().tolist()
        captured["layer_0.q_post_rope"] = q_rot[0, 0, -1, :_FIRST_N].detach().cpu().tolist()

        # Also capture all KV heads for the last token (first _FIRST_N elements each)
        # so we can compare against Mila's full K cache row at position
        n_kv_heads = k_rot.shape[1]
        k_all_heads = []
        for h in range(n_kv_heads):
            k_all_heads.append([float(x) for x in k_rot[0, h, -1, :_FIRST_N].detach().cpu().tolist()])
        captured_slices["layer_0.k_post_rope_all_heads"] = k_all_heads

        return q_rot, k_rot

    llama_mod.apply_rotary_pos_emb = capturing_rope

    def restore():
        llama_mod.apply_rotary_pos_emb = original_rope

    return restore

# ── Hook registration — decode only, layer 0 ────────────────────────────────

def register_decode_hooks():
    hooks = []
    layer = model.model.layers[0]

    if hasattr(model.model, "embed_tokens"):
        hooks.append(model.model.embed_tokens.register_forward_hook(
            make_stats_hook("token_embeds")))
        hooks.append(model.model.embed_tokens.register_forward_hook(
            make_store_rows_hook(f"token_embeds_rows_first{_FIRST_N}")))

    if hasattr(layer, "input_layernorm"):
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_stats_hook("layer_0.rmsn_1")))
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_store_rows_hook(f"layer_0.rmsn_1_rows_first{_FIRST_N}")))

    if hasattr(layer.self_attn, "q_proj"):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(
            make_stats_hook("layer_0.q_pre_rope")))
        hooks.append(layer.self_attn.q_proj.register_forward_hook(
            make_store_rows_hook(f"layer_0.q_pre_rope_rows_first{_FIRST_N}")))

    if hasattr(layer.self_attn, "k_proj"):
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            make_stats_hook("layer_0.k_pre_rope")))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            make_store_rows_hook(f"layer_0.k_pre_rope_rows_first{_FIRST_N}")))

    if hasattr(layer.self_attn, "v_proj"):
        hooks.append(layer.self_attn.v_proj.register_forward_hook(
            make_stats_hook("layer_0.v_proj")))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(
            make_store_rows_hook(f"layer_0.v_proj_rows_first{_FIRST_N}")))

    if hasattr(layer.self_attn, "o_proj"):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
            make_attn_output_pre_hook(0)))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            make_stats_hook("layer_0.fc_out_proj")))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            make_store_rows_hook(f"layer_0.fc_out_proj_rows_first{_FIRST_N}")))

    if hasattr(layer, "post_attention_layernorm"):
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            make_stats_hook("layer_0.rmsn_2")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            make_store_rows_hook(f"layer_0.rmsn_2_rows_first{_FIRST_N}")))

    if hasattr(layer, "mlp"):
        if hasattr(layer.mlp, "gate_proj"):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(
                make_stats_hook("layer_0.gate_proj")))
            hooks.append(layer.mlp.gate_proj.register_forward_hook(
                make_store_rows_hook(f"layer_0.gate_proj_rows_first{_FIRST_N}")))
        if hasattr(layer.mlp, "up_proj"):
            hooks.append(layer.mlp.up_proj.register_forward_hook(
                make_stats_hook("layer_0.up_proj")))
            hooks.append(layer.mlp.up_proj.register_forward_hook(
                make_store_rows_hook(f"layer_0.up_proj_rows_first{_FIRST_N}")))
        if hasattr(layer.mlp, "down_proj"):
            hooks.append(layer.mlp.down_proj.register_forward_hook(
                make_stats_hook("layer_0.fc_down")))
            hooks.append(layer.mlp.down_proj.register_forward_hook(
                make_store_rows_hook(f"layer_0.fc_down_rows_first{_FIRST_N}")))

    hooks.append(layer.register_forward_hook(
        make_stats_hook("layer_0.block_out")))
    hooks.append(layer.register_forward_hook(
        make_store_rows_hook(f"layer_0.block_out_rows_first{_FIRST_N}")))

    return hooks

# ── Formatting helpers ───────────────────────────────────────────────────────

def _fmt_num(x: float) -> str:
    return f"{x:.6g}"

def _print_table(name, rows_list):
    if not rows_list:
        print(f"{name}: (no rows captured)")
        return

    cols = min(max(len(r) for r in rows_list), _FIRST_N)

    formatted_rows = []
    for row in rows_list[:_MAX_ROWS]:
        formatted = []
        for j in range(cols):
            val = row[j] if j < len(row) else None
            formatted.append(_fmt_num(val) if val is not None else "")
        formatted_rows.append(formatted)

    col_widths = []
    for j in range(cols):
        max_cell = max((len(r[j]) for r in formatted_rows), default=0)
        col_widths.append(max(len(f"C{j}"), max_cell))

    header_cols = " | ".join(f"{f'C{j}':>{col_widths[j]}}" for j in range(cols))
    header      = f"Row | {header_cols}"
    sep         = "-" * len(header)

    print(f"\n{name}")
    print(sep)
    print(header)
    print(sep)

    for i, row in enumerate(formatted_rows):
        row_str = " | ".join(f"{row[j]:>{col_widths[j]}}" for j in range(cols))
        print(f"{i:3} | {row_str}")

    if len(rows_list) > _MAX_ROWS:
        print(f"... ({len(rows_list)} rows captured, showing first {_MAX_ROWS})")

# ── Phase 1: Prefill (no hooks — just populate KV cache) ────────────────────

print(f"\n{'='*72}")
print(f"  PHASE 1: PREFILL  ({input_ids.shape[1]} tokens, hooks disabled)")
print(f"{'='*72}")

with torch.no_grad():
    prefill_out     = model(input_ids, use_cache=True)
    past_key_values = prefill_out.past_key_values
    prefill_logits  = prefill_out.logits

next_token_id = prefill_logits[0, -1, :].argmax().item()
print(f"Prefill top prediction: {tokenizer.decode([next_token_id])!r} (id={next_token_id})")

top5 = torch.topk(prefill_logits[0, -1, :], 5)
print("Top 5 prefill predictions:")
for v, idx in zip(top5.values, top5.indices):
    print(f"  {tokenizer.decode([idx.item()])!r:15} {_fmt_num(v.item())}")

# ── Phase 2: Decode loop — hook step 1 only (position 5, input ' the') ──────

print(f"\n{'='*72}")
print(f"  DECODE LOOP — 2 steps, hooking step 1 only (position 5)")
print(f"{'='*72}")

num_decode_steps = 2
current_token_id = next_token_id

for step in range(num_decode_steps):
    position     = input_ids.shape[1] + step
    decode_input = torch.tensor([[current_token_id]])

    hooks        = []
    restore_rope = None

    if step == 1:
        hooks        = register_decode_hooks()
        restore_rope = install_rope_capture()

    with torch.no_grad():
        decode_out      = model(decode_input, past_key_values=past_key_values, use_cache=True)
        decode_logits   = decode_out.logits
        past_key_values = decode_out.past_key_values

    if step == 1:
        for h in hooks:
            h.remove()
        restore_rope()

    predicted_id    = decode_logits[0, -1, :].argmax().item()
    predicted_token = tokenizer.decode([predicted_id])

    print(f"  step={step} pos={position} in={tokenizer.decode([current_token_id])!r:10} -> {predicted_token!r}")

    current_token_id = predicted_id

# ── Print layer 0 decode checkpoints ────────────────────────────────────────

print(f"\n{'='*72}")
print(f"  DECODE STEP 1 (pos=5, input=' the') — Layer 0 Checkpoints")
print(f"  Compare directly against Mila print_stats output")
print(f"{'='*72}")

checkpoint_keys = [
    "token_embeds",
    "layer_0.rmsn_1",
    "layer_0.q_pre_rope",
    "layer_0.k_pre_rope",
    "layer_0.v_proj",
    "layer_0.attn_out",
    "layer_0.fc_out_proj",
    "layer_0.rmsn_2",
    "layer_0.gate_proj",
    "layer_0.up_proj",
    "layer_0.fc_down",
    "layer_0.block_out",
]

for key in checkpoint_keys:
    val = captured.get(key, "not captured")
    if isinstance(val, dict) and "checksum" in val:
        print(f"  {key}:")
        print(f"    {_fmt_mila_stats(val)}")
    else:
        print(f"  {key}: not captured")

# ── Post-RoPE K/Q comparison ─────────────────────────────────────────────────

print(f"\n{'='*72}")
print(f"  POST-RoPE K/Q (layer 0, head 0, first {_FIRST_N} elements)")
print(f"  Compare k_post_rope[0] against Mila decode.k cache row 5")
print(f"{'='*72}")

k_post = captured.get("layer_0.k_post_rope", "not captured")
q_post = captured.get("layer_0.q_post_rope", "not captured")
print(f"  k_post_rope (head 0): {[f'{x:.6f}' for x in k_post] if isinstance(k_post, list) else k_post}")
print(f"  q_post_rope (head 0): {[f'{x:.6f}' for x in q_post] if isinstance(q_post, list) else q_post}")

print(f"\n--- First {_FIRST_N} elements (up to {_MAX_ROWS} rows) ---")
for k, v in captured_slices.items():
    rows = [v] if (v and isinstance(v[0], float)) else v
    _print_table(k, rows)