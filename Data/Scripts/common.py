# ============================================================================
# File: common.py
# Common utilities for weight conversion
# ============================================================================

import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, BinaryIO
import json


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
        
        print(f"? Wrote {len(self.tensors)} tensors to {self.output_path}")
        print(f"  File size: {self.output_path.stat().st_size / 1024**2:.2f} MB")
    
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

