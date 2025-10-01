# Tensor System Design Summary

This document tracks the design decisions made during the tensor system redesign for Mila.

## Alignment with Project Guidelines

This tensor design directly supports the project's stated priorities from [copilot-instructions.md](.github/copilot-instructions.md):

## Tensor Design Summary

The Mila tensor system achieves device-aware computing through **TMemoryResource** template-based dispatch, enabling zero-overhead device selection at compile time.

### Core Architecture

- **TMemoryResource** encodes storage location and provides ComputeDeviceTag for TensorOps dispatch
- **DeviceContext** provides device binding and resource management  
- **Abstract TensorDataType** prevents device-specific compilation issues

### TensorOps Device Dispatch Mechanism

The **TensorOps** module uses TMemoryResource::ComputeDeviceTag to dispatch to device-specific specializations:

Current priority backends:
- `TensorOps<CpuComputeDeviceTag>` - CPU backend operations (from CpuMemoryResource)
- `TensorOps<CudaComputeDeviceTag>` - CUDA backend operations (from CudaMemoryResource)

Memory resource types and their dispatch tags:
- `CpuMemoryResource::ComputeDeviceTag` -> `CpuComputeDeviceTag`
- `CudaMemoryResource::ComputeDeviceTag` -> `CudaComputeDeviceTag`
- `CudaPinnedMemoryResource::ComputeDeviceTag` -> `CudaComputeDeviceTag`
- `CudaManagedMemoryResource::ComputeDeviceTag` -> `CudaComputeDeviceTag`

### Key Features

- **Compile-time operation dispatch** - TMemoryResource ComputeDeviceTag selects TensorOps backend
- **Type safety** - Invalid device/memory combinations rejected at compile time
- **Device context validation** - Ensures compatibility between memory resources and devices
- **Automatic optimization** - Hardware-specific alignment (CUDA warp, AVX-512)
- **Seamless transfers** - Memory operations preserve tensor metadata across devices
- **Multi-GPU support** - DeviceContext handles proper device binding

### Design Benefits

1. **Zero runtime overhead** - All device dispatch happens at compile time
2. **Extensible** - New backends added via new TensorOps specializations
3. **Type-safe** - Abstract data types work across all device backends
4. **Memory efficient** - Automatic alignment optimization per target hardware
5. **Device-agnostic** - Unified API across heterogeneous compute environments