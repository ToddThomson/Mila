"""Measure Mila and llama.cpp on the same card and write the table the website shows.

Every row runs prefill at 512, 2K, 8K and 32K tokens and generation of 128 tokens at context
depth 0, 8K and 32K, in both engines, pinned to one GPU by UUID. Mila runs through ProfileModel
(generate(), the entry a consumer calls); llama.cpp runs llama-bench with flash attention, every
layer on the GPU and an FP16 KV cache. Each cell is the mean of --runs measured runs after one
warmup in both engines.

Results are written to the JSON file after every cell, so --resume continues an interrupted run.
A cell that fails (weights missing, a deployment Mila refuses, a GGUF llama.cpp cannot load) is
recorded with its reason and stays in the table.

    python benchmark_comparison.py --llama-bench C:/llama.cpp/llama-bench.exe
"""
import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
MODELS = REPOSITORY / "Data" / "Models"

PREFILL_LENGTHS = [512, 2048, 8192, 32768]
GENERATION_DEPTHS = [0, 8192, 32768]
GENERATED_TOKENS = 128

# One row per published model and Mila format. A GGUF is a path under Data/Models or a
# (repository, file) pair in the Hugging Face cache; None means no llama.cpp counterpart yet.
# head_to_head marks rows where both engines run the same weights in the same format.
ROWS = [
    {
        "key": "gemma-4-12b-q4_0",
        "model": "Gemma 4 12B Instruct",
        "family": "gemma",
        "mila_quantization": "q4_0",
        "mila_weights": "Gemma/gemma4_12b_it_qat_q4_0.safetensors",
        "mila_format": "Q4_0 (QAT)",
        "gguf": ("google/gemma-4-12B-it-qat-q4_0-gguf", "gemma-4-12b-it-qat-q4_0.gguf"),
        "llama_cpp_format": "Q4_0 (QAT)",
        "head_to_head": True,
    },
    {
        # Quantized on load from BF16 until the model's Q4_0 expert bank lands; then q4_0, head to head.
        "key": "gemma-4-26b-a4b-fp4",
        "model": "Gemma 4 26B-A4B Instruct",
        "family": "gemma",
        "mila_quantization": "fp4",
        "mila_weights": "Gemma/gemma4_26b_a4b_it_bf16.bin",
        "mila_format": "FP4",
        "gguf": ("google/gemma-4-26B-A4B-it-qat-q4_0-gguf", "gemma-4-26B_q4_0-it.gguf"),
        "llama_cpp_format": "Q4_0 (QAT)",
        "head_to_head": False,
    },
    {
        "key": "llama-3.1-8b-q4_0",
        "model": "Llama 3.1 8B Instruct",
        "family": "llama",
        "mila_quantization": "q4_0",
        "mila_weights": "LLaMa/llama31_8b_instruct_q4_0.safetensors",
        "mila_format": "Q4_0",
        "gguf": "LLaMa/llama31_8b_instruct_q4_0.gguf",
        "llama_cpp_format": "Q4_0",
        "head_to_head": True,
    },
    {
        "key": "llama-3.1-8b-fp4",
        "model": "Llama 3.1 8B Instruct",
        "family": "llama",
        "mila_quantization": "fp4",
        "mila_weights": "LLaMa/llama31_8b_instruct_fp4.safetensors",
        "mila_format": "FP4",
        "gguf": "LLaMa/llama31_8b_instruct_q4_0.gguf",
        "llama_cpp_format": "Q4_0",
        "head_to_head": False,
    },
    {
        "key": "llama-3.2-3b-fp4",
        "model": "Llama 3.2 3B Instruct",
        "family": "llama",
        "mila_quantization": "fp4",
        "mila_weights": "LLaMa/llama32_3b_instruct_fp4.safetensors",
        "mila_format": "FP4",
        "gguf": "LLaMa/llama32_3b_instruct_q4_0.gguf",
        "llama_cpp_format": "Q4_0",
        "head_to_head": False,
    },
    {
        "key": "qwen-3.8-27b-fp4",
        "model": "Qwen 3.8 27B",
        "family": "qwen",
        "mila_quantization": "fp4",
        "mila_weights": "Qwen/qwen38_27b_fp4.safetensors",
        "mila_format": "FP4",
        "gguf": None,
        "llama_cpp_format": None,
        "head_to_head": False,
    },
    {
        "key": "qwen-3.8-27b-cb2-3",
        "model": "Qwen 3.8 27B",
        "family": "qwen",
        "mila_quantization": "plan",
        "mila_weights": "Qwen/qwen38_27b_cb2-3.safetensors",
        "mila_format": "2/3-bit codebook",
        "gguf": None,
        "llama_cpp_format": None,
        "head_to_head": False,
    },
]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llama-bench", default=os.environ.get("MILA_LLAMA_BENCH") or shutil.which("llama-bench"),
                        help="llama-bench executable. Default: $MILA_LLAMA_BENCH, else llama-bench on PATH.")
    parser.add_argument("--profile-model", default=str(REPOSITORY / "out" / "build" / "x64-profile" / "ProfileModel.exe"),
                        help="ProfileModel executable. Default: the x64-profile build.")
    parser.add_argument("--gpu", default="RTX 5060 Ti",
                        help="Substring of the GPU name both engines are pinned to. Default: the 16 GB reference card.")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per cell. Default: 3.")
    parser.add_argument("--rows", help="Comma-separated row keys to run. Default: every row.")
    parser.add_argument("--engines", default="mila,llama.cpp", help="Comma-separated engines. Default: both.")
    parser.add_argument("--output", default="benchmark_comparison.json", help="Results file (JSON).")
    parser.add_argument("--markdown", help="Also write the tables to this Markdown file.")
    parser.add_argument("--resume", action="store_true", help="Keep cells already in --output and run the rest.")
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds before one engine run is abandoned.")
    parser.add_argument("--list", action="store_true", help="Print the rows and exit.")

    return parser.parse_args()


def resolve_gpu(name):
    """UUID and full name of the one GPU whose name contains `name`."""
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout
    matches = [[field.strip() for field in line.split(",")] for line in output.splitlines() if name in line]

    if len(matches) != 1:
        sys.exit(f"--gpu '{name}' matches {len(matches)} GPUs:\n{output}")

    gpu_name, uuid, driver = matches[0]

    return {"name": gpu_name, "uuid": uuid, "driver": driver}


def resolve_gguf(gguf):
    """Path of a row's GGUF, or (None, reason) when it is not on this machine."""
    if gguf is None:
        return None, "no llama.cpp format chosen for this row"

    if isinstance(gguf, str):
        path = MODELS / gguf

        return (path, None) if path.exists() else (None, f"{path} not found")

    repository, filename = gguf

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None, "huggingface_hub is not installed, so the cache cannot be searched"

    try:
        return Path(hf_hub_download(repository, filename, local_files_only=True)), None
    except Exception:
        return None, f"{repository}/{filename} is not in the Hugging Face cache"


def context_for(tokens):
    """The smallest multiple of 1024 that holds `tokens`; llama-bench likewise sizes its context to the test."""
    return max(1024, math.ceil(tokens / 1024) * 1024)


def summary(samples):
    return {
        "mean": statistics.fmean(samples),
        "stddev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples": samples,
    }


def run(command, environment, timeout):
    """(stdout, None) on success, else (None, the last lines of the failure)."""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=timeout,
                                   encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return None, f"no result after {timeout} s"

    if completed.returncode != 0:
        tail = (completed.stderr.strip() or completed.stdout.strip()).splitlines()[-6:]

        return None, "\n".join(tail) or f"exit code {completed.returncode}"

    return completed.stdout, None


def mila_prefill(arguments, row, length, environment):
    command = [
        arguments.profile_model, "--model", row["family"], "--quantization", row["mila_quantization"],
        "--model-path", str(MODELS / row["mila_weights"]), "--phase", "prefill", "--seq-len", str(length),
        "--context-length", str(context_for(length)), "--warmup", "1", "--runs", str(arguments.runs),
    ]
    stdout, error = run(command, environment, arguments.timeout)

    if error:
        return {"error": error, "command": command}

    match = re.search(r"\[prefill\] runs=\d+ .* run_ms=([\d.,]+)", stdout)

    if not match:
        return {"error": "ProfileModel printed no prefill result", "command": command}

    samples = [length / (float(ms) / 1000.0) for ms in match.group(1).split(",")]

    return {**summary(samples), "command": command}


def mila_generation(arguments, row, depth, environment):
    # One token more than llama.cpp generates: the first comes from the prefill's logits, and
    # the rate is read over the GENERATED_TOKENS decode steps between the first and the last.
    tokens = GENERATED_TOKENS + 1
    command = [
        arguments.profile_model, "--model", row["family"], "--quantization", row["mila_quantization"],
        "--model-path", str(MODELS / row["mila_weights"]), "--phase", "decode", "--seq-len", str(max(1, depth)),
        "--tokens", str(tokens), "--ignore-eos", "--context-length", str(context_for(max(1, depth) + tokens)),
        "--warmup", "1", "--runs", str(arguments.runs),
    ]
    stdout, error = run(command, environment, arguments.timeout)

    if error:
        return {"error": error, "command": command}

    generated = [int(value) for value in re.findall(r"\[decode_measured\] .*tokens_generated=(\d+)", stdout)]

    if any(count != tokens for count in generated):
        return {"error": f"generated {generated} tokens per run, expected {tokens}", "command": command}

    match = re.search(r"\[decode\] runs=\d+ .* run_tok_per_s=([\d.,]+)", stdout)

    if not match:
        return {"error": "ProfileModel printed no generation result", "command": command}

    return {**summary([float(value) for value in match.group(1).split(",")]), "command": command}


def llama_bench(arguments, gguf, test_arguments, environment):
    """llama-bench's JSON records for one invocation, or {'error': ...}."""
    command = [
        arguments.llama_bench, "-m", str(gguf), "-ngl", "99", "-fa", "on", "-ctk", "f16", "-ctv", "f16",
        "-r", str(arguments.runs), "-o", "json", *test_arguments,
    ]
    stdout, error = run(command, environment, arguments.timeout)

    if error:
        return {"error": error, "command": command}

    try:
        return {"records": json.loads(stdout), "command": command}
    except json.JSONDecodeError:
        return {"error": "llama-bench printed no JSON", "command": command}


def llama_cpp_cells(arguments, gguf, environment):
    """Every llama.cpp cell for one GGUF: two llama-bench invocations cover the prefill and generation tests."""
    cells = {}
    prefill = llama_bench(arguments, gguf, ["-p", ",".join(map(str, PREFILL_LENGTHS)), "-n", "0"], environment)
    generation = llama_bench(arguments, gguf, ["-p", "0", "-n", str(GENERATED_TOKENS),
                                               "-d", ",".join(map(str, GENERATION_DEPTHS))], environment)

    for kind, result, tests in (("prefill", prefill, PREFILL_LENGTHS), ("generation", generation, GENERATION_DEPTHS)):
        for value in tests:
            cell = f"{kind}:{value}"

            if "error" in result:
                cells[cell] = {"error": result["error"], "command": result["command"]}
                continue

            if kind == "prefill":
                record = next((r for r in result["records"] if r["n_prompt"] == value and r["n_gen"] == 0), None)
            else:
                record = next((r for r in result["records"]
                               if r["n_prompt"] == 0 and r["n_gen"] == GENERATED_TOKENS and r["n_depth"] == value), None)

            if record is None:
                cells[cell] = {"error": "llama-bench reported no such test", "command": result["command"]}
            else:
                cells[cell] = {**summary(record["samples_ts"]), "command": result["command"]}

    return cells


def llama_cpp_version(llama_bench_path):
    completed = subprocess.run([llama_bench_path, "--version"], capture_output=True, text=True)
    match = re.search(r"build (\d+), commit (\w+)", completed.stdout + completed.stderr)

    return f"llama.cpp build {match.group(1)} ({match.group(2)})" if match else "llama.cpp (build unknown)"


def format_cell(mila, llama_cpp):
    def rate(cell):
        if cell is None:
            return "not run"

        if "error" in cell:
            return "failed"

        return f"{cell['mean']:,.0f}"

    text = f"{rate(mila)} / {rate(llama_cpp)}"

    if mila and llama_cpp and "mean" in mila and "mean" in llama_cpp:
        text += f" ({mila['mean'] / llama_cpp['mean']:.2f}x)"

    return text


def markdown(results):
    lines = [
        f"Mila {results['mila_version']} and {results['llama_cpp_version']} on the {results['gpu']['name']} "
        f"(driver {results['gpu']['driver']}). Tokens per second, Mila / llama.cpp (Mila's ratio), mean of "
        f"{results['runs']} runs.",
        "",
    ]

    for kind, title, tests, label in (
        ("prefill", "Prefill", PREFILL_LENGTHS, lambda v: f"{v // 1024}K" if v >= 1024 else str(v)),
        ("generation", f"Generation ({GENERATED_TOKENS} tokens) at context depth", GENERATION_DEPTHS,
         lambda v: f"{v // 1024}K" if v else "0"),
    ):
        lines += [f"### {title}", "",
                  "| Model | Mila | llama.cpp | " + " | ".join(label(v) for v in tests) + " |",
                  "|---|---|---|" + "---|" * len(tests)]

        for row in ROWS:
            row_results = results["rows"].get(row["key"])

            if row_results is None:
                continue

            cells = [format_cell(row_results["mila"].get(f"{kind}:{v}"), row_results["llama.cpp"].get(f"{kind}:{v}"))
                     for v in tests]
            lines.append(f"| {row['model']} | {row['mila_format']} | {row['llama_cpp_format'] or '--'} | "
                         + " | ".join(cells) + " |")

        lines.append("")

    failures = [(key, engine, cell, value["error"])
                for key, row_results in results["rows"].items()
                for engine in ("mila", "llama.cpp")
                for cell, value in row_results[engine].items() if "error" in value]

    if failures:
        lines += ["### Cells without a result", ""]
        lines += [f"- {key}, {engine}, {cell}: {error.splitlines()[-1] if error else ''}"
                  for key, engine, cell, error in failures]
        lines.append("")

    return "\n".join(lines)


def main():
    arguments = parse_arguments()

    if arguments.list:
        for row in ROWS:
            gguf, reason = resolve_gguf(row["gguf"])
            print(f"{row['key']:<22} Mila {row['mila_format']:<18} llama.cpp {row['llama_cpp_format'] or '--':<12} "
                  f"{gguf or reason}")
        return

    engines = set(arguments.engines.split(","))
    selected = [row for row in ROWS if not arguments.rows or row["key"] in arguments.rows.split(",")]

    if "llama.cpp" in engines and not (arguments.llama_bench and Path(arguments.llama_bench).exists()):
        sys.exit("llama-bench not found: pass --llama-bench or set MILA_LLAMA_BENCH.")

    if "mila" in engines and not Path(arguments.profile_model).exists():
        sys.exit(f"ProfileModel not found at {arguments.profile_model}: build it or pass --profile-model.")

    gpu = resolve_gpu(arguments.gpu)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu["uuid"]}
    output = Path(arguments.output)

    results = json.loads(output.read_text()) if arguments.resume and output.exists() else {"rows": {}}
    results.update({
        "gpu": gpu,
        "runs": arguments.runs,
        "mila_version": (REPOSITORY / "Version.txt").read_text().strip(),
        "llama_cpp_version": llama_cpp_version(arguments.llama_bench) if arguments.llama_bench else "llama.cpp",
    })

    def save():
        output.write_text(json.dumps(results, indent=2))

    # llama.cpp results are per GGUF, so a GGUF two rows share is measured once.
    llama_cpp_by_gguf = {}

    for row in selected:
        row_results = results["rows"].setdefault(row["key"], {"mila": {}, "llama.cpp": {}})

        if "mila" in engines:
            for length in PREFILL_LENGTHS:
                cell = f"prefill:{length}"

                if cell not in row_results["mila"] or not arguments.resume:
                    print(f"{row['key']}: Mila prefill {length}", flush=True)
                    row_results["mila"][cell] = mila_prefill(arguments, row, length, environment)
                    save()

            for depth in GENERATION_DEPTHS:
                cell = f"generation:{depth}"

                if cell not in row_results["mila"] or not arguments.resume:
                    print(f"{row['key']}: Mila generation at depth {depth}", flush=True)
                    row_results["mila"][cell] = mila_generation(arguments, row, depth, environment)
                    save()

        if "llama.cpp" in engines:
            gguf, reason = resolve_gguf(row["gguf"])
            wanted = [f"prefill:{v}" for v in PREFILL_LENGTHS] + [f"generation:{v}" for v in GENERATION_DEPTHS]

            if gguf is None:
                row_results["llama.cpp"] = {cell: {"error": reason} for cell in wanted}
            elif arguments.resume and all(cell in row_results["llama.cpp"] for cell in wanted):
                pass
            else:
                if gguf not in llama_cpp_by_gguf:
                    print(f"{row['key']}: llama.cpp {gguf.name}", flush=True)
                    llama_cpp_by_gguf[gguf] = llama_cpp_cells(arguments, gguf, environment)

                row_results["llama.cpp"] = dict(llama_cpp_by_gguf[gguf])

            save()

    table = markdown(results)
    print()
    print(table)

    if arguments.markdown:
        Path(arguments.markdown).write_text(table)


if __name__ == "__main__":
    main()
