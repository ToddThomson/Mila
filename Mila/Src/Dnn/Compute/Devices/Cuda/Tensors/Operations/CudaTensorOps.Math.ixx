/**
 * @file CudaTensorOps.Math.ixx
 * @brief CUDA tensor mathematical operations partition
 */

module;
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <source_location>
#include <cmath>
#include "Kernels/Math.Elementwise.h"

export module Dnn.TensorOps:Math.Cuda;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorOps;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;
import Compute.DeviceTraits;
import Compute.CudaDeviceContext;
import Compute.CudaMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.CudaManagedMemoryResource;
import Cuda.Error;

namespace Mila::Dnn
{
	template<typename TComputeDeviceTag> struct TensorOps; // Forward declaration

    template<>
    struct TensorOps<Compute::CudaComputeDeviceTag> 
    {
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static Tensor<TDataType, TMemoryResource> add( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
            if (a.shape() != b.shape()) {
                throw std::invalid_argument( "Tensor shapes must match for addition" );
            }
            auto context = a.deviceContext();
            
            if (!context || context != b.deviceContext()) {
                throw std::invalid_argument( "Tensors must have the same valid device context" );
            }
            
            Tensor<TDataType, TMemoryResource> result( context, a.shape() );
            
            /*Math::add<TDataType>( a.rawData(), b.rawData(), result.rawData(),
                a.size(), std::dynamic_pointer_cast<CudaDeviceContext>(context) );*/
            
            return result;
        }
    };

    /**
     * @brief CUDA backend mathematical tensor operations
     */
    //export class Math
    //{
    //public:
    //    // ================================================================
    //    // Element-wise Binary Operations
    //    // ================================================================

    //    template<TensorDataType TDataType>
    //    static void add( const void* src1_data, const void* src2_data, void* dst_data,
    //        size_t count, std::shared_ptr<CudaDeviceContext> context ) {
    //        if (!src1_data || !src2_data || !dst_data || count == 0 || !context) {
    //            return;
    //        }

    //        context->makeCurrent();

    //        using NativeType = typename NativeDataTypeMap<TDataType>::type;
    //        const auto* typed_src1 = static_cast<const NativeType*>(src1_data);
    //        const auto* typed_src2 = static_cast<const NativeType*>(src2_data);
    //        auto* typed_dst = static_cast<NativeType*>(dst_data);

    //        constexpr int block = 256;
    //        const int grid = static_cast<int>((count + block - 1) / block);

    //        // Call host-side wrapper function (implemented in .cu file)
    //        launch_elementwise_add_kernel<NativeType>(
    //            typed_src1, typed_src2, typed_dst, count, grid, block, context->getStream() );

    //        cudaError_t status = cudaGetLastError();
    //        cudaCheckStatus( status, std::source_location::current() );
    //    }

    //    template<TensorDataType TDataType>
    //    static void subtract( const void* src1_data, const void* src2_data, void* dst_data,
    //        size_t count, std::shared_ptr<CudaDeviceContext> context ) {
    //        if (!src1_data || !src2_data || !dst_data || count == 0 || !context) {
    //            return;
    //        }

    //        context->makeCurrent();

    //        using NativeType = typename NativeDataTypeMap<TDataType>::type;
    //        const auto* typed_src1 = static_cast<const NativeType*>(src1_data);
    //        const auto* typed_src2 = static_cast<const NativeType*>(src2_data);
    //        auto* typed_dst = static_cast<NativeType*>(dst_data);

    //        constexpr int block = 256;
    //        const int grid = static_cast<int>((count + block - 1) / block);

    //        launch_elementwise_subtract_kernel<NativeType>(
    //            typed_src1, typed_src2, typed_dst, count, grid, block, context->getStream() );

    //        cudaError_t status = cudaGetLastError();
    //        cudaCheckStatus( status, std::source_location::current() );
    //    }

    //    // ... similar pattern for multiply, divide, addScalar, multiplyScalar, relu, sigmoid, tanh

    //    template<TensorDataType TDataType>
    //    static float sum( const void* src_data, size_t count,
    //        std::shared_ptr<CudaDeviceContext> context ) {
    //        if (!src_data || count == 0 || !context) {
    //            return 0.0f;
    //        }

    //        context->makeCurrent();

    //        using NativeType = typename NativeDataTypeMap<TDataType>::type;
    //        const auto* typed_src = static_cast<const NativeType*>(src_data);

    //        constexpr int block = 256;
    //        const int grid = static_cast<int>((count + block - 1) / block);

    //        float* d_partial_sums = nullptr;
    //        cudaError_t status = cudaMallocAsync( reinterpret_cast<void**>(&d_partial_sums),
    //            grid * sizeof( float ), context->getStream() );
    //        cudaCheckStatus( status, std::source_location::current() );

    //        float result = 0.0f;
    //        try {
    //            // Call host-side wrapper for reduction kernel
    //            launch_sum_reduction_kernel<NativeType>(
    //                typed_src, d_partial_sums, count, grid, block,
    //                block * sizeof( float ), context->getStream() );

    //            std::vector<float> h_partial_sums( grid );
    //            status = cudaMemcpyAsync( h_partial_sums.data(), d_partial_sums,
    //                grid * sizeof( float ), cudaMemcpyDeviceToHost, context->getStream() );
    //            cudaCheckStatus( status, std::source_location::current() );

    //            context->synchronize();

    //            for (float partial : h_partial_sums) {
    //                result += partial;
    //            }

    //            cudaFreeAsync( d_partial_sums, context->getStream() );
    //        }
    //        catch (...) {
    //            cudaFreeAsync( d_partial_sums, context->getStream() );
    //            throw;
    //        }

    //        return result;
    //    }

        // Similar pattern for max() reduction
    //;
}