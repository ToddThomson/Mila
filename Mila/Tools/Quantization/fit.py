"""Calibration and fitting: what decides the scales, the codebooks and the codes.

Two passes are both called calibration and produce different objects:

  collect_importance  per-input-channel mean squared activation, one vector per linear.
                      A handful of forward passes over the whole model. Feeds the
                      k-means weighting in the codebook fit.
  gptq_apply          a full [in, in] Hessian per linear, accumulated layer by layer.
                      Feeds both the codebook fit (through its diagonal) and the
                      error-compensated column walk.

The second subsumes the first, which is why --gptq does not run collect_importance.
"""

import torch

from artifact import emit_mila_layer, pack_codebook_tensor
from formats import (CODEBOOK_PARAMETERS, FORMATS, GENERATOR_SEED, GPTQ_FORMATS,
                     SCHEMES, as_fp16_storage, fake_codebook, fit_codebook_levels,
                     fit_codebook_levels_joint, guard_scale, nearest_level_index)


def collect_importance(model, tokenizer, texts, device):
    """Per-input-channel mean squared activation for every target linear, via hooks.

    This is the calibration pass -- the analog of llama.cpp's importance matrix.
    A handful of forward passes over ordinary text is enough to expose which
    channels the model drives hard.
    """
    sums = {}
    counts = {}
    hooks = []

    def make_hook(name):
        def hook(module, inputs):
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            sums[name] = sums.get(name, 0) + (x * x).sum(0)
            counts[name] = counts.get(name, 0) + x.shape[0]
        return hook

    target_suffixes = set(SCHEMES["codebook"])
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in target_suffixes and hasattr(module, "weight"):
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    with torch.no_grad():
        for text in texts:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            model(input_ids[:, :1024])

    for hook in hooks:
        hook.remove()

    return {name: sums[name] / counts[name] for name in sums}


# ---------------------------------------------------------------------------
# GPTQ-style error compensation: quantize columns sequentially, folding each
# column's quantization error into the not-yet-quantized columns through the
# inverse Hessian of the layer's calibration activations. The remaining large
# lever after codebooks and calibration (spec section 8, Phase 0).
# ---------------------------------------------------------------------------

class _StopForward(Exception):
    pass


def gptq_quantize_tensor(weight, hessian, levels, group_size, divisor,
                         block_size=128, damp_fraction=0.01):
    """Return (Q, codes, scale_bits): the compensated-quantized weight, its level
    indices [out, in] uint8, and its FP16 scale bits [out, in/group_size] uint16.
    Group scales are computed when the column walk reaches each group, from the
    residual-updated weights -- group sizes of 32/64/128 all divide block_size, so
    a group never straddles a block."""
    W = weight.float().clone()
    out_features, in_features = W.shape
    H = hessian.clone()

    diagonal = torch.arange(in_features, device=H.device)
    dead = torch.diag(H) == 0
    H[diagonal[dead], diagonal[dead]] = 1.0
    W[:, dead] = 0.0
    H[diagonal, diagonal] += damp_fraction * torch.diag(H).mean()

    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    codes = torch.zeros(out_features, in_features, dtype=torch.uint8, device=W.device)
    scale_bits = torch.zeros(out_features, in_features // group_size,
                             dtype=torch.uint16, device=W.device)
    levels = levels.to(W.device)
    scale = None

    for block_start in range(0, in_features, block_size):
        block_end = min(block_start + block_size, in_features)
        count = block_end - block_start
        W1 = W[:, block_start:block_end].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[block_start:block_end, block_start:block_end]

        for i in range(count):
            column = block_start + i
            if column % group_size == 0:
                group_end = min(i + group_size, count)
                absmax = W1[:, i:group_end].abs().amax(1)
                scale = guard_scale(as_fp16_storage(absmax / divisor))
                scale_bits[:, column // group_size] = scale.half().view(torch.uint16)
            w = W1[:, i]
            index = nearest_level_index(w / scale, levels)
            q = levels[index] * scale
            Q1[:, i] = q
            codes[:, column] = index.to(torch.uint8)
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err

        Q[:, block_start:block_end] = Q1
        if block_end < in_features:
            W[:, block_end:] -= Err1 @ Hinv[block_start:block_end, block_end:]

    return Q, codes, scale_bits


def gptq_apply(model, tokenizer, calibration_text, device, args, artifact=None,
               artifact_policies=None):
    """Sequential GPTQ over the decoder stack.

    Per layer: capture the layer's calibration inputs, accumulate per-linear
    Hessians, quantize each target linear with compensation, then recompute the
    layer's outputs with the quantized weights so the next layer calibrates
    against what it will actually see at inference.

    artifact / artifact_policies are filled in place when an artifact is being
    written: the Mila-named packed tensors, and the policy each one carries.
    """
    policy_by_suffix = SCHEMES["codebook"]
    layers = model.model.layers
    num_layers = len(layers)

    token_ids = tokenizer(calibration_text, return_tensors="pt").input_ids[0]
    generator = torch.Generator().manual_seed(GENERATOR_SEED)
    starts = torch.randint(0, max(1, token_ids.numel() - args.gptq_seqlen),
                           (args.gptq_samples,), generator=generator)
    samples = [token_ids[s:s + args.gptq_seqlen].unsqueeze(0).to(device) for s in starts]

    captured = []
    original_forward = layers[0].forward

    def catcher(*call_args, **call_kwargs):
        hidden = call_args[0] if call_args else call_kwargs.pop("hidden_states")
        captured.append((hidden, call_kwargs))
        raise _StopForward

    layers[0].forward = catcher
    with torch.no_grad():
        for sample in samples:
            try:
                model(sample)
            except _StopForward:
                pass
    layers[0].forward = original_forward

    total_params = 0
    total_bits = 0.0
    skipped_layers = []

    for layer_index, layer in enumerate(layers):
        protected = (layer_index < args.protect_first
                     or layer_index >= num_layers - args.protect_last)

        targets = {}
        for name, module in layer.named_modules():
            suffix = name.rsplit(".", 1)[-1]
            if suffix in policy_by_suffix and hasattr(module, "weight"):
                targets[name] = module

        hessians = {}
        counts = {}
        hooks = []

        def make_hook(name):
            def hook(module, inputs):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], device=x.device)
                    counts[name] = 0
                hessians[name] += x.T @ x
                counts[name] += x.shape[0]
            return hook

        for name, module in targets.items():
            hooks.append(module.register_forward_pre_hook(make_hook(name)))
        with torch.no_grad():
            for hidden, kwargs in captured:
                layer(hidden, **kwargs)
        for hook in hooks:
            hook.remove()

        # Mila fuses gate_proj and up_proj into one tensor (tf_layer_N.fc_gate_up), and a
        # codebook is per tensor, so a fused artifact can carry only one table. Fitting the
        # pair jointly is the cheapest way to make the artifact expressible; this measures
        # what that costs. The two projections read the SAME input, so their Hessians and
        # hence the importance vector are identical -- only the fit is shared. Off by
        # default: the recorded Phase 0 numbers are per-HF-linear.
        shared_levels = {}
        if args.fuse_gate_up_codebook and not protected:
            gate_name = next((n for n in targets if n.endswith("gate_proj")), None)
            up_name = next((n for n in targets if n.endswith("up_proj")), None)
            gate_policy = policy_by_suffix.get("gate_proj")

            if gate_name and up_name and gate_policy in CODEBOOK_PARAMETERS:
                k, fit_group = CODEBOOK_PARAMETERS[gate_policy]
                importance = torch.diag(hessians[gate_name]) / counts[gate_name]
                shared = fit_codebook_levels_joint(
                    [targets[gate_name].weight.data, targets[up_name].weight.data],
                    k, fit_group, importance=importance)
                shared_levels[gate_name] = shared
                shared_levels[up_name] = shared

        packed_pieces = {}
        fitted_levels = {}

        for name, module in targets.items():
            policy = "fp4" if protected else policy_by_suffix[name.rsplit(".", 1)[-1]]
            levels, group_size, divisor, bits = GPTQ_FORMATS[policy]
            weight = module.weight.data
            if levels is None and name in shared_levels:
                levels = shared_levels[name]
            if levels is None:
                k = 4 if policy == "cb4" else 8
                importance = torch.diag(hessians[name]) / counts[name]
                levels = fit_codebook_levels(weight, k, group_size,
                                             importance=importance)
            quantized, codes, scale_bits = gptq_quantize_tensor(
                weight, hessians[name], levels, group_size, divisor)
            if artifact is not None and policy in CODEBOOK_PARAMETERS:
                packed_pieces[name] = pack_codebook_tensor(
                    f"model.layers.{layer_index}.{name}", policy, quantized, codes,
                    scale_bits, levels, group_size)
                fitted_levels[name] = levels
            module.weight.data = quantized.to(weight.dtype)
            total_params += weight.numel()
            total_bits += bits * weight.numel()

        if artifact is not None and not protected:
            emit_mila_layer(artifact, artifact_policies, layer_index, targets,
                            packed_pieces, fitted_levels)
        elif artifact is not None:
            skipped_layers.append(layer_index)

        hessians.clear()

        # clear() drops the references but not the reservation: torch's caching allocator
        # keeps freed blocks, so across 28 layers of differently-shaped Hessians and
        # float32 fit transients the reserved pool creeps monotonically -- measured 8.5 GB
        # climbing to 11.5 GB and spilling into shared memory on the last layers of a
        # 12 GiB card. Returning the blocks each layer bounds it, at the cost of some
        # reallocation, and is what lets this gate run on a smaller card at all.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            for i, (hidden, kwargs) in enumerate(captured):
                output = layer(hidden, **kwargs)
                captured[i] = (output[0] if isinstance(output, tuple) else output,
                               kwargs)

        print(f"  layer {layer_index + 1}/{num_layers} done"
              + (" (protected, fp4)" if protected else ""), flush=True)

    average = total_bits / total_params if total_params else 0.0
    print(f"  quantized {total_params / 1e9:.3f} B params, "
          f"{average:.3f} average bits/weight over quantized tensors")
    if model.config.tie_word_embeddings:
        print("  lm_head is tied to embed_tokens on this model: left in BF16")
    if skipped_layers:
        print(f"  artifact carries no codebook record for protected layers "
              f"{skipped_layers}: those stay BF16 and quantize to FP4 at load")
    return total_params, average


def layer_index_of(module_name):
    parts = module_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def apply_scheme(model, scheme_name, importance=None, protect_first=0, protect_last=0,
                 verbose=True):
    """Fake-quantize matching linear weights in place. Returns (params_quantized, avg_bits).

    importance: optional {module_name: per-channel mean squared activation} from
    collect_importance; feeds the weighted codebook fit.
    protect_first / protect_last: decoder layers held at fp4 instead of their
    sub-4-bit policy (the llama.cpp UD move -- edge layers are the sensitive ones).
    """
    policy_by_suffix = SCHEMES[scheme_name]
    total_params = 0
    total_bits = 0.0
    tied = model.config.tie_word_embeddings
    num_layers = model.config.num_hidden_layers

    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        policy = policy_by_suffix.get(suffix)
        if policy is None or not hasattr(module, "weight"):
            continue
        if suffix == "lm_head" and tied:
            if verbose:
                print("  lm_head is tied to embed_tokens on this model: left in BF16")
            continue

        layer = layer_index_of(name)
        if layer is not None and (layer < protect_first or layer >= num_layers - protect_last):
            policy = "fp4"

        weight = module.weight.data
        if policy in CODEBOOK_PARAMETERS:
            k, group_size = CODEBOOK_PARAMETERS[policy]
            channel_importance = importance.get(name) if importance else None
            dequantized, bits = fake_codebook(weight, k, group_size,
                                              importance=channel_importance)
        else:
            dequantized, bits = FORMATS[policy](weight)
        module.weight.data = dequantized.to(weight.dtype)
        total_params += weight.numel()
        total_bits += bits * weight.numel()

    average = total_bits / total_params if total_params else 0.0
    if verbose:
        print(f"  quantized {total_params / 1e9:.3f} B params, "
              f"{average:.3f} average bits/weight over quantized tensors")
    return total_params, average
