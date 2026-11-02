#!/usr/bin/env python3
"""Write the BF16 twin of a packed codebook artifact: each codebook tensor decoded, everything else as stored.

The twin is what QwenModel.Codebook.Cuda.cpp loads through the BF16 path to check the packed path
against: both hold the same weight values, so a gap between them beyond kernel rounding is a fault in
how Mila loads or multiplies the codes. Decoding goes through packing.py, the layout's one Python codec.

  python decode_codebook.py <packed.safetensors> <twin.safetensors>
  python decode_codebook.py <packed.safetensors> <broken.safetensors> --drop-high-plane

--drop-high-plane decodes every three-bit tensor through its low two bits only: the negative control a
comparison has to fail, or it is not looking. Requires numpy, torch and safetensors.
"""
import argparse
import json
import struct

import numpy as np
import torch
from safetensors.torch import save_file

import packing

ARRAY_TYPES = {"U8": np.uint8, "F16": np.float16, "F32": np.float32, "BF16": np.uint16}


def read_header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]

        return json.loads(f.read(size)), 8 + size


def read_array(path, base, entry):
    begin, end = entry["data_offsets"]

    with open(path, "rb") as f:
        f.seek(base + begin)
        raw = f.read(end - begin)

    return np.frombuffer(raw, dtype=ARRAY_TYPES[entry["dtype"]]).reshape(entry["shape"])


def decode(low_plane, high_plane, scale, codebook):
    """codebook[code] * float32(scale[row, column // group]), as BF16 -- the value the packed path multiplies by."""
    columns = low_plane.shape[1] * 4

    if high_plane is None:
        codes = packing.unpack_two_bit_codes(low_plane, columns)
    else:
        codes = packing.unpack_three_bit_codes(low_plane, high_plane, columns)

    group = columns // scale.shape[1]
    values = codebook.astype(np.float32)[codes] * np.repeat(scale.astype(np.float32), group, axis=1)

    return torch.from_numpy(values).to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--drop-high-plane", action="store_true")
    args = parser.parse_args()

    header, base = read_header(args.source)
    metadata = header.pop("__metadata__")
    stems = {name.removesuffix(".weight_codebook") for name in header if name.endswith(".weight_codebook")}
    companions = {f"{stem}.weight{suffix}" for stem in stems for suffix in ("_codebook", "_scale", "_high_plane")}

    tensors = {}

    for name, entry in header.items():
        if name in companions:
            continue

        stem = name.removesuffix(".weight")

        if name.endswith(".weight") and stem in stems:
            high_name = f"{stem}.weight_high_plane"
            high = None

            if high_name in header and not args.drop_high_plane:
                high = read_array(args.source, base, header[high_name])

            tensors[name] = decode(read_array(args.source, base, entry), high,
                                   read_array(args.source, base, header[f"{stem}.weight_scale"]),
                                   read_array(args.source, base, header[f"{stem}.weight_codebook"]))
            print(f"decoded {name} {list(tensors[name].shape)}", flush=True)
        elif entry["dtype"] == "BF16":
            tensors[name] = torch.from_numpy(read_array(args.source, base, entry).copy()).view(torch.bfloat16)
        else:
            raise SystemExit(f"{name}: {entry['dtype']} is neither BF16 nor part of a codebook tensor")

    # "none": the twin is unquantized storage, so the loader quantizes only what the compiled plan asks it to.
    save_file(tensors, args.destination,
              metadata={"mila_config": metadata["mila_config"], "mila_quantization": "none"})
    print(f"wrote {args.destination}: {len(tensors)} tensors, {len(stems)} decoded")


if __name__ == "__main__":
    main()
