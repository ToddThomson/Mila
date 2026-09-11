"""
What the model store holds, from Python.

The store is the only thing a load reads from, and it is the same store Chat and the
Mila Inference Server use -- one directory, several processes. Pull and load are
separate verbs, so nothing here reaches a network: listing, locating and measuring
are filesystem work.

    python store.py
    python store.py --locate gemma-4-12b-it-fp4

A name is the whole key. There is no catalogue and no path: what a model is called
here is what you type, and the record carries the rest -- architecture, whether it is
instruction-tuned, and which blobs are its weights and its tokenizer.
"""

import argparse

import common


def format_bytes(count):
    """GB/MB on 1024-based divisors, matching what Chat prints."""
    if count >= 1024 ** 3:
        return f"{count / 1024 ** 3:.2f} GB"

    if count >= 1024 ** 2:
        return f"{count / 1024 ** 2:.1f} MB"

    return f"{count} bytes"


def show_listing(store):
    models = store.list()

    if not models:
        print("No models are installed.")
        print(f"Store root: {store.root}")

        return

    print(f"{'MODEL':<28} {'ARCHITECTURE':<14} {'SIZE':>10}  ORIGIN")

    for model in sorted(models, key=lambda m: m.name):
        # A model published from this machine has no hub; origin is a field rather
        # than a path precisely so it can change without moving any files.
        origin = f"{model.hub}:{model.owner}/{model.repository}" if model.hub else "local"
        incomplete = "" if model.complete else "  [incomplete -- re-pull]"

        print(f"{model.name:<28} {model.architecture:<14} "
              f"{format_bytes(model.bytes_on_disk):>10}  {origin}{incomplete}")

    usage = store.usage()

    print()
    print(f"{usage.model_count} model(s), {usage.blob_count} blob(s), "
          f"{format_bytes(usage.blob_bytes)} on disk")

    # Deduplication is content addressing, not naming: two models with byte-identical
    # tokenizers share one blob, so the per-model sizes above can sum to more than this.
    if usage.reclaimable_bytes:
        print(f"{format_bytes(usage.reclaimable_bytes)} reclaimable")

    if usage.partial_bytes:
        print(f"{format_bytes(usage.partial_bytes)} in partial transfers (a retry resumes onto them)")


def show_model(store, name):
    model = store.locate(name)

    if model is None:
        print(f"No complete model named '{name}' is installed.")
        print("locate() also returns None when the record survives but a blob does not.")

        return

    print(f"name                {model.name}")
    print(f"architecture        {model.architecture}")
    print(f"variant             {model.variant}")
    print(f"weight quantization {model.weight_quantization or 'none'}")
    print(f"instruct            {model.instruct}")
    print(f"base model          {model.base_model or '-'}")
    print(f"license             {model.license or '-'}")
    print(f"installed           {model.installed_at or '-'}")
    print(f"size                {format_bytes(model.bytes_on_disk)}")
    print()
    print(f"weights             {model.weights_path}")
    print(f"tokenizer           {model.tokenizer_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locate", metavar="NAME",
        help="Show one model in full instead of the listing.")
    parser.add_argument("--root", default="",
        help="Store directory. Default: MILA_CACHE_DIR, then the platform user cache.")
    args = parser.parse_args()

    common.configure_console()

    mila = common.import_mila()
    store = mila.ModelStore(args.root)

    if args.locate:
        show_model(store, args.locate)
    else:
        show_listing(store)


if __name__ == "__main__":
    main()
