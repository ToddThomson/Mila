#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace Mila::Dnn::Compute::Cuda
{
    // ============================================================================
    // Templated Type Mapping for Device Code
    // ============================================================================

    template<typename T>
    struct DeviceTypeInfo;

    template<> struct DeviceTypeInfo<float> { using type = float; };
    template<> struct DeviceTypeInfo<__half> { using type = __half; };
    template<> struct DeviceTypeInfo<__nv_bfloat16> { using type = __nv_bfloat16; };
    template<> struct DeviceTypeInfo<__nv_fp8_e4m3> { using type = __nv_fp8_e4m3; };
    template<> struct DeviceTypeInfo<__nv_fp8_e5m2> { using type = __nv_fp8_e5m2; };
    template<> struct DeviceTypeInfo<int8_t> { using type = int8_t; };
    template<> struct DeviceTypeInfo<int16_t> { using type = int16_t; };
    template<> struct DeviceTypeInfo<int32_t> { using type = int32_t; };
    template<> struct DeviceTypeInfo<uint8_t> { using type = uint8_t; };
    template<> struct DeviceTypeInfo<uint16_t> { using type = uint16_t; };
    template<> struct DeviceTypeInfo<uint32_t> { using type = uint32_t; };

    // ============================================================================
    // Scalable Constant Fill Kernels (No Temporary Memory Required)
    // ============================================================================

    template<typename TargetType, typename HostType>
    __global__ void fill_constant_kernel(TargetType* dst, size_t count, HostType host_value) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        // Convert once per thread
        TargetType converted_value = static_cast<TargetType>(host_value);

        // Grid-stride loop for scalability
        for (size_t i = idx; i < count; i += stride) {
            dst[i] = converted_value;
        }
    }

    // ============================================================================
    // Chunked Array Fill Kernels (Memory-Efficient for Large Tensors)
    // ============================================================================

    template<typename TargetType, typename HostType>
    __global__ void fill_array_kernel(TargetType* dst, const HostType* src, size_t count, size_t dst_offset = 0) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        // Grid-stride loop with offset support for chunked processing
        for (size_t i = idx; i < count; i += stride) {
            dst[dst_offset + i] = static_cast<TargetType>(src[i]);
        }
    }

    // ============================================================================
    // Launch Configuration Helpers
    // ============================================================================

    inline dim3 calculateOptimalGrid(size_t count, int blockSize = 256) {
        int deviceId;
        cudaGetDevice(&deviceId);

        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, deviceId);

        // Use occupancy calculator for optimal block count
        int minGridSize, optimalBlockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize,
            fill_constant_kernel<float, float>, 0, blockSize);

        // Calculate grid size ensuring we don't exceed device limits
        int maxGridSize = props.maxGridSize[0];
        int gridSize = (count + blockSize - 1) / blockSize;
        gridSize = min(gridSize, maxGridSize);
        gridSize = min(gridSize, minGridSize * props.multiProcessorCount);

        return dim3(gridSize);
    }

    // ============================================================================
    // Templated Launch Functions (Compile-Time Dispatch)
    // ============================================================================

    template<typename TargetType, typename HostType>
    void launch_constant_fill_typed(void* dst, size_t count, HostType host_value, cudaStream_t stream) {
        using NativeType = typename DeviceTypeInfo<TargetType>::type;

        const int blockSize = 256;
        dim3 grid = calculateOptimalGrid(count, blockSize);
        dim3 block(blockSize);

        fill_constant_kernel<<<grid, block, 0, stream>>>(
            static_cast<NativeType*>(dst), count, host_value);
    }

    template<typename TargetType, typename HostType>
    void launch_array_fill_typed(void* dst, const HostType* host_values, size_t count, cudaStream_t stream) {
        using NativeType = typename DeviceTypeInfo<TargetType>::type;

        // For large arrays, use chunked processing to limit temporary memory usage
        const size_t CHUNK_SIZE = 1024 * 1024;  // 1M elements per chunk

        if (count <= CHUNK_SIZE) {
            // Small array - process in single kernel
            HostType* d_temp;
            cudaMalloc(&d_temp, count * sizeof(HostType));
            cudaMemcpyAsync(d_temp, host_values, count * sizeof(HostType),
                cudaMemcpyHostToDevice, stream);

            const int blockSize = 256;
            dim3 grid = calculateOptimalGrid(count, blockSize);
            dim3 block(blockSize);

            fill_array_kernel<<<grid, block, 0, stream>>>(
                static_cast<NativeType*>(dst), d_temp, count);

            // Cleanup after kernel completion
            cudaStreamAddCallback(stream, [](cudaStream_t, cudaError_t, void* userData) {
                cudaFree(userData);
            }, d_temp, 0);
        }
        else {
            // Large array - process in chunks
            for (size_t offset = 0; offset < count; offset += CHUNK_SIZE) {
                size_t chunk_count = min(CHUNK_SIZE, count - offset);

                HostType* d_temp;
                cudaMalloc(&d_temp, chunk_count * sizeof(HostType));
                cudaMemcpyAsync(d_temp, host_values + offset, chunk_count * sizeof(HostType),
                    cudaMemcpyHostToDevice, stream);

                const int blockSize = 256;
                dim3 grid = calculateOptimalGrid(chunk_count, blockSize);
                dim3 block(blockSize);

                fill_array_kernel<<<grid, block, 0, stream>>>(
                    static_cast<NativeType*>(dst), d_temp, chunk_count, offset);

                // Cleanup after kernel completion
                cudaStreamAddCallback(stream, [](cudaStream_t, cudaError_t, void* userData) {
                    cudaFree(userData);
                }, d_temp, 0);
            }
        }
    }

    // ============================================================================
    // Explicit Template Instantiations for Common Types
    // ============================================================================

    // Integer constant fills
    template void launch_constant_fill_typed<int8_t, int32_t>(void*, size_t, int32_t, cudaStream_t);
    template void launch_constant_fill_typed<int16_t, int32_t>(void*, size_t, int32_t, cudaStream_t);
    template void launch_constant_fill_typed<int32_t, int32_t>(void*, size_t, int32_t, cudaStream_t);
    template void launch_constant_fill_typed<uint8_t, int32_t>(void*, size_t, int32_t, cudaStream_t);
    template void launch_constant_fill_typed<uint16_t, int32_t>(void*, size_t, int32_t, cudaStream_t);
    template void launch_constant_fill_typed<uint32_t, int32_t>(void*, size_t, int32_t, cudaStream_t);

    // Float constant fills
    template void launch_constant_fill_typed<float, float>(void*, size_t, float, cudaStream_t);
    template void launch_constant_fill_typed<__half, float>(void*, size_t, float, cudaStream_t);
    template void launch_constant_fill_typed<__nv_bfloat16, float>(void*, size_t, float, cudaStream_t);
    template void launch_constant_fill_typed<__nv_fp8_e4m3, float>(void*, size_t, float, cudaStream_t);
    template void launch_constant_fill_typed<__nv_fp8_e5m2, float>(void*, size_t, float, cudaStream_t);

    // Integer array fills
    template void launch_array_fill_typed<int8_t, int32_t>(void*, const int32_t*, size_t, cudaStream_t);
    template void launch_array_fill_typed<int16_t, int32_t>(void*, const int32_t*, size_t, cudaStream_t);
    template void launch_array_fill_typed<int32_t, int32_t>(void*, const int32_t*, size_t, cudaStream_t);
    template void launch_array_fill_typed<uint8_t, int32_t>(void*, const int32_t*, size_t, cudaStream_t);
    template void launch_array_fill_typed<uint16_t, int32_t>(void*, const int32_t*, size_t, cudaStream_t);
    template void launch_array_fill_typed<uint32_t, int32_t>(void*, const int32_t*, size_t, cudaStream_t);

    // Float array fills
    template void launch_array_fill_typed<float, float>(void*, const float*, size_t, cudaStream_t);
    template void launch_array_fill_typed<__half, float>(void*, const float*, size_t, cudaStream_t);
    template void launch_array_fill_typed<__nv_bfloat16, float>(void*, const float*, size_t, cudaStream_t);
    template void launch_array_fill_typed<__nv_fp8_e4m3, float>(void*, const float*, size_t, cudaStream_t);
    template void launch_array_fill_typed<__nv_fp8_e5m2, float>(void*, const float*, size_t, cudaStream_t);
}