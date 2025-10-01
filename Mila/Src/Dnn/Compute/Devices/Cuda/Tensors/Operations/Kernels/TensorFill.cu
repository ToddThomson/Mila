#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace Mila::Dnn::Compute
{

// Include tensor data type enumeration

    // Forward declaration of TensorDataType enum - defined in host code
    enum class TensorDataType : int {
        FP32 = 0,
        FP16 = 1,
        BF16 = 2,
        FP8_E4M3 = 3,
        FP8_E5M2 = 4,
        INT8 = 5,
        INT16 = 6,
        INT32 = 7,
        UINT8 = 8,
        UINT16 = 9,
        UINT32 = 10
    };

// ============================================================================
// Templated Type Mapping for Device Code
// ============================================================================

template<TensorDataType T>
struct DeviceTypeMapper;

template<> struct DeviceTypeMapper<TensorDataType::FP32> { using type = float; };
template<> struct DeviceTypeMapper<TensorDataType::FP16> { using type = __half; };
template<> struct DeviceTypeMapper<TensorDataType::BF16> { using type = __nv_bfloat16; };
template<> struct DeviceTypeMapper<TensorDataType::FP8_E4M3> { using type = __nv_fp8_e4m3; };
template<> struct DeviceTypeMapper<TensorDataType::FP8_E5M2> { using type = __nv_fp8_e5m2; };
template<> struct DeviceTypeMapper<TensorDataType::INT8> { using type = int8_t; };
template<> struct DeviceTypeMapper<TensorDataType::INT16> { using type = int16_t; };
template<> struct DeviceTypeMapper<TensorDataType::INT32> { using type = int32_t; };
template<> struct DeviceTypeMapper<TensorDataType::UINT8> { using type = uint8_t; };
template<> struct DeviceTypeMapper<TensorDataType::UINT16> { using type = uint16_t; };
template<> struct DeviceTypeMapper<TensorDataType::UINT32> { using type = uint32_t; };

// ============================================================================
// Scalable Constant Fill Kernels (No Temporary Memory Required)
// ============================================================================

template<typename TargetType, typename HostType>
__global__ void fill_constant_kernel( TargetType* dst, size_t count, HostType host_value ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Convert once per thread
    TargetType converted_value = static_cast<TargetType>(host_value);

    // Grid-stride loop for scalability
    for ( size_t i = idx; i < count; i += stride ) {
        dst[ i ] = converted_value;
    }
}

// ============================================================================
// Chunked Array Fill Kernels (Memory-Efficient for Large Tensors)
// ============================================================================

template<typename TargetType, typename HostType>
__global__ void fill_array_kernel( TargetType* dst, const HostType* src, size_t count, size_t dst_offset = 0 ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Grid-stride loop with offset support for chunked processing
    for ( size_t i = idx; i < count; i += stride ) {
        dst[ dst_offset + i ] = static_cast<TargetType>( src[ i ] );
    }
}

// ============================================================================
// Launch Configuration Helpers
// ============================================================================

inline dim3 calculateOptimalGrid( size_t count, int blockSize = 256 ) {
    // Calculate optimal grid size based on element count and device capabilities
    int deviceId;
    cudaGetDevice( &deviceId );

    cudaDeviceProp props;
    cudaGetDeviceProperties( &props, deviceId );

    // Use occupancy calculator for optimal block count
    int minGridSize, optimalBlockSize;
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &optimalBlockSize,
        fill_constant_kernel<float, float>, 0, blockSize );

    // Calculate grid size ensuring we don't exceed device limits
    int maxGridSize = props.maxGridSize[ 0 ];
    int gridSize = (count + blockSize - 1) / blockSize;
    gridSize = min( gridSize, maxGridSize );
    gridSize = min( gridSize, minGridSize * props.multiProcessorCount );

    return dim3( gridSize );
}

// ============================================================================
// Type-Dispatched Constant Fill Functions
// ============================================================================

template<TensorDataType TargetType, typename HostType>
void launch_constant_fill_typed( void* dst, size_t count, HostType host_value, cudaStream_t stream ) {
    using NativeType = typename DeviceTypeMapper<TargetType>::type;

    const int blockSize = 256;
    dim3 grid = calculateOptimalGrid( count, blockSize );
    dim3 block( blockSize );

    fill_constant_kernel << <grid, block, 0, stream >> > (
        static_cast<NativeType*>(dst), count, host_value
        );
}

template<TensorDataType TargetType, typename HostType>
void launch_array_fill_typed( void* dst, const HostType* host_values, size_t count, cudaStream_t stream ) {
    using NativeType = typename DeviceTypeMapper<TargetType>::type;

    // For large arrays, use chunked processing to limit temporary memory usage
    const size_t CHUNK_SIZE = 1024 * 1024;  // 1M elements per chunk

    if ( count <= CHUNK_SIZE ) {
        // Small array - process in single kernel
        HostType* d_temp;
        cudaMalloc( &d_temp, count * sizeof( HostType ) );
        cudaMemcpyAsync( d_temp, host_values, count * sizeof( HostType ),
            cudaMemcpyHostToDevice, stream );

        const int blockSize = 256;
        dim3 grid = calculateOptimalGrid( count, blockSize );
        dim3 block( blockSize );

        fill_array_kernel << <grid, block, 0, stream >> > (
            static_cast<NativeType*>(dst), d_temp, count
            );

        // Cleanup after kernel completion
        cudaStreamAddCallback( stream, []( cudaStream_t, cudaError_t, void* userData ) {
            cudaFree( userData );
            }, d_temp, 0 );
    }
    else {
        // Large array - process in chunks
        for ( size_t offset = 0; offset < count; offset += CHUNK_SIZE ) {
            size_t chunk_count = min( CHUNK_SIZE, count - offset );

            HostType* d_temp;
            cudaMalloc( &d_temp, chunk_count * sizeof( HostType ) );
            cudaMemcpyAsync( d_temp, host_values + offset, chunk_count * sizeof( HostType ),
                cudaMemcpyHostToDevice, stream );

            const int blockSize = 256;
            dim3 grid = calculateOptimalGrid( chunk_count, blockSize );
            dim3 block( blockSize );

            fill_array_kernel << <grid, block, 0, stream >> > (
                static_cast<NativeType*>(dst), d_temp, chunk_count, offset
                );

            // Cleanup after kernel completion
            cudaStreamAddCallback( stream, []( cudaStream_t, cudaError_t, void* userData ) {
                cudaFree( userData );
                }, d_temp, 0 );
        }
    }
}

// ============================================================================
// C-Style Interface Functions for CudaMemoryResource
// ============================================================================


    void launch_fill_constant_int32_kernel( void* dst, size_t count, int32_t host_value,
        TensorDataType target_type, cudaStream_t stream ) {
        switch ( target_type ) {
            case TensorDataType::INT8:
                launch_constant_fill_typed<TensorDataType::INT8>( dst, count, host_value, stream );
                break;
            case TensorDataType::INT16:
                launch_constant_fill_typed<TensorDataType::INT16>( dst, count, host_value, stream );
                break;
            case TensorDataType::INT32:
                launch_constant_fill_typed<TensorDataType::INT32>( dst, count, host_value, stream );
                break;
            case TensorDataType::UINT8:
                launch_constant_fill_typed<TensorDataType::UINT8>( dst, count, host_value, stream );
                break;
            case TensorDataType::UINT16:
                launch_constant_fill_typed<TensorDataType::UINT16>( dst, count, host_value, stream );
                break;
            case TensorDataType::UINT32:
                launch_constant_fill_typed<TensorDataType::UINT32>( dst, count, host_value, stream );
                break;
            default:
                // Error: unsupported type conversion
                break;
        }
    }

    void launch_fill_constant_float_kernel( void* dst, size_t count, float host_value,
        TensorDataType target_type, cudaStream_t stream ) {
        switch ( target_type ) {
            case TensorDataType::FP32:
                launch_constant_fill_typed<TensorDataType::FP32>( dst, count, host_value, stream );
                break;
            case TensorDataType::FP16:
                launch_constant_fill_typed<TensorDataType::FP16>( dst, count, host_value, stream );
                break;
            case TensorDataType::BF16:
                launch_constant_fill_typed<TensorDataType::BF16>( dst, count, host_value, stream );
                break;
            case TensorDataType::FP8_E4M3:
                launch_constant_fill_typed<TensorDataType::FP8_E4M3>( dst, count, host_value, stream );
                break;
            case TensorDataType::FP8_E5M2:
                launch_constant_fill_typed<TensorDataType::FP8_E5M2>( dst, count, host_value, stream );
                break;
            default:
                // Error: unsupported type conversion
                break;
        }
    }

    void launch_fill_int32_kernel( void* dst, size_t count, const int32_t* host_values,
        TensorDataType target_type, cudaStream_t stream ) {
        switch ( target_type ) {
            case TensorDataType::INT8:
                launch_array_fill_typed<TensorDataType::INT8>( dst, host_values, count, stream );
                break;
            case TensorDataType::INT16:
                launch_array_fill_typed<TensorDataType::INT16>( dst, host_values, count, stream );
                break;
            case TensorDataType::INT32:
                launch_array_fill_typed<TensorDataType::INT32>( dst, host_values, count, stream );
                break;
            case TensorDataType::UINT8:
                launch_array_fill_typed<TensorDataType::UINT8>( dst, host_values, count, stream );
                break;
            case TensorDataType::UINT16:
                launch_array_fill_typed<TensorDataType::UINT16>( dst, host_values, count, stream );
                break;
            case TensorDataType::UINT32:
                launch_array_fill_typed<TensorDataType::UINT32>( dst, host_values, count, stream );
                break;
            default:
                // Error: unsupported type conversion
                break;
        }
    }

    void launch_fill_float_kernel( void* dst, size_t count, const float* host_values,
        TensorDataType target_type, cudaStream_t stream ) {
        switch ( target_type ) {
            case TensorDataType::FP32:
                launch_array_fill_typed<TensorDataType::FP32>( dst, host_values, count, stream );
                break;
            case TensorDataType::FP16:
                launch_array_fill_typed<TensorDataType::FP16>( dst, host_values, count, stream );
                break;
            case TensorDataType::BF16:
                launch_array_fill_typed<TensorDataType::BF16>( dst, host_values, count, stream );
                break;
            case TensorDataType::FP8_E4M3:
                launch_array_fill_typed<TensorDataType::FP8_E4M3>( dst, host_values, count, stream );
                break;
            case TensorDataType::FP8_E5M2:
                launch_array_fill_typed<TensorDataType::FP8_E5M2>( dst, host_values, count, stream );
                break;
            default:
                // Error: unsupported type conversion
                break;
        }
    }
}