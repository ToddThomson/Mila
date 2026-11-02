"""Group nsys cuda_gpu_kern_sum CSVs into attention / GEMM / FP8-plumbing / other.

Prints the share table AND every kernel that fell into 'other' above a threshold,
so a miscategorisation shows up instead of hiding inside the residual.
"""
import csv
import glob
import os
import re
import sys

def csv_dir():
    """Where the nsys CSVs live. Defaults to the working directory.

    This used to default to the script's own directory, which was right while it sat
    beside its inputs in a scratchpad and is wrong now that it lives in the tree.
    """
    return sys.argv[1] if len(sys.argv) > 1 else "."

# Ordered: first match wins.
RULES = [
    # Weight unpacking + activation quantisation: everything that exists only to feed
    # the GEMM a format it accepts. Codebook builds unpack with codebook_dequantize_*
    # rather than dequantize_fp4_to_fp8 -- same role, different name, and leaving it
    # out put 35.5% of Qwen prefill in 'other'.
    ("plumbing", re.compile(
        r"dequantize_fp4_to_fp8|codebook_dequantize|apply_per_token_scales"
        r"|fp8_activation_quantize|fp8_weight_scale|quantize_bf16_to_fp8", re.I)),
    # NOTE: the namespace is `Cuda::Gqa::`, so an earlier `gqa_` pattern MISSED
    # `Gqa::prefill_softmax_bf16_kernel` -- 75% of one config -- and dumped it in
    # 'other'. Match the namespace, not just the kernel-name prefix.
    ("attention", re.compile(
        r"Gqa::|Mha::|gqa_|flash|kvcache|softmax|attention|delta_?net|\bmha_", re.I)),
    ("gemm", re.compile(
        r"nvjet|cutlass|gemm|matvec|\bsgemm|\bdot_kernel|wmma", re.I)),
]


def classify(name):
    for label, rx in RULES:
        if rx.search(name):
            return label
    return "other"


def load(path):
    groups = {"attention": 0.0, "gemm": 0.0, "plumbing": 0.0, "other": 0.0}
    others = []
    total = 0.0

    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name_key = next((k for k in row if k and "Name" in k), None)
            time_key = next((k for k in row if k and "Total Time" in k), None)

            if not name_key or not time_key:
                continue

            try:
                ns = float(row[time_key])
            except (TypeError, ValueError):
                continue

            label = classify(row[name_key])
            groups[label] += ns
            total += ns

            if label == "other":
                others.append((ns, row[name_key][:70]))

    return groups, others, total


def main():
    directory = csv_dir()
    files = sorted(glob.glob(os.path.join(directory, "*_cuda_gpu_kern_sum.csv")))

    if not files:
        print(f"no *_cuda_gpu_kern_sum.csv found in {os.path.abspath(directory)}")
        return 1

    print(f"{'config':<26}{'attention':>11}{'GEMM':>9}{'plumbing':>10}{'other':>8}   total ms")
    print("-" * 76)

    all_others = {}

    for path in files:
        base = os.path.basename(path)
        tag = base[:-len("_cuda_gpu_kern_sum.csv")]

        groups, others, total = load(path)

        if total <= 0:
            print(f"{tag:<26}  (empty)")
            continue

        pct = {k: 100.0 * v / total for k, v in groups.items()}
        print(f"{tag:<26}{pct['attention']:>10.1f}%{pct['gemm']:>8.1f}%"
              f"{pct['plumbing']:>9.1f}%{pct['other']:>7.1f}%{total/1e6:>11.1f}")

        for ns, nm in others:
            if 100.0 * ns / total >= 1.0:
                all_others.setdefault(nm, []).append((tag, 100.0 * ns / total))

    if all_others:
        print("\nUncategorised kernels above 1% (check none of these belong in a group):")
        for nm, hits in sorted(all_others.items()):
            worst = max(h[1] for h in hits)
            print(f"  {worst:5.1f}% max   {nm}")
    else:
        print("\nNo uncategorised kernel exceeds 1%.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
