/**
 * @file CudaExecutionContext.ixx
 * @brief CUDA-specific execution context specialization.
 */

module;
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <curand.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#include <memory>
#include <string>
#include <format>
#include <stdexcept>

export module Compute.ExecutionContext:Cuda;

import Compute.ExecutionContextTemplate;
import Compute.IExecutionContext;
import Compute.DeviceId;
import Compute.DeviceType;
import Core.RandomGenerator;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA execution context specialization.
     *
     * Manages CUDA execution resources including streams and library handles.
     * Each context owns an independent CUDA stream and owns the library
     * handles used by callers bound to this context's stream.
     *
     * Thread Safety:
     * - Multiple contexts can safely exist on the same device.
     * - Each context has its own stream for isolated execution.
     * - Library handles are owned by the context and must not be used
     *   concurrently from multiple threads without external synchronization.
     */
    template<> class ExecutionContext<DeviceType::Cuda> : public IExecutionContext
    {
    public:
        /**
         * @brief Constructs CUDA execution context for a specific device.
         *
         * Creates an independent CUDA stream.
         *
         * @param device_id CUDA device identifier.
         * @throws std::invalid_argument If device_id type is not Cuda.
         * @throws std::runtime_error If CUDA stream creation fails.
         */
        explicit ExecutionContext( DeviceId device_id )
            : device_id_( validateDeviceId( device_id ) )
        {
            initializeResources();
        }

        /**
         * @brief Destructor with proper CUDA resource cleanup.
         */
        ~ExecutionContext()
        {
            releaseResources();
        }

        ExecutionContext( const ExecutionContext& ) = delete;
        ExecutionContext& operator=( const ExecutionContext& ) = delete;
        ExecutionContext( ExecutionContext&& ) = delete;
        ExecutionContext& operator=( ExecutionContext&& ) = delete;

        /**
         * @brief Gets the device identifier.
         *
         * @return DeviceId The CUDA device identifier (type + index).
         */
        [[nodiscard]] DeviceId getDeviceId() const noexcept override
        {
            return device_id_;
        }

        /**
         * @brief Synchronizes the CUDA stream.
         *
         * Blocks until all operations submitted to this context's stream complete.
         *
         * @throws std::runtime_error If stream synchronization fails.
         */
        void synchronize() override
        {
            cudaError_t error = cudaStreamSynchronize( stream_ );

            if ( error != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "CUDA stream synchronization failed: {}", cudaGetErrorString( error ) )
                );
            }
        }

        /**
         * @brief Gets the CUDA stream for asynchronous operations.
         *
         * @return cudaStream_t The CUDA stream owned by this context.
         */
        [[nodiscard]] cudaStream_t getStream() const noexcept
        {
            return stream_;
        }

        /**
         * @brief Gets the cuRAND generator bound to this context's stream.
         *
         * Created lazily, seeded from Core::RandomGenerator, and bound to this
         * context's stream. Reuse across calls avoids per-call creation overhead.
         *
         * @return curandGenerator_t The cuRAND generator bound to this stream.
         * @throws std::runtime_error If generator creation or seeding fails.
         */
        [[nodiscard]] curandGenerator_t getCurandGenerator() const
        {
            if ( !curand_initialized_ )
            {
                curandStatus_t status = curandCreateGenerator( &curand_generator_, CURAND_RNG_PSEUDO_DEFAULT );

                if ( status != CURAND_STATUS_SUCCESS )
                {
                    throw std::runtime_error( "Failed to create cuRAND generator" );
                }

                auto seed = static_cast<unsigned long long>( Core::RandomGenerator::getInstance().getSeed() );
                status = curandSetPseudoRandomGeneratorSeed( curand_generator_, seed );

                if ( status != CURAND_STATUS_SUCCESS )
                {
                    curandDestroyGenerator( curand_generator_ );
                    curand_generator_ = nullptr;
                    throw std::runtime_error( "Failed to seed cuRAND generator" );
                }

                status = curandSetStream( curand_generator_, stream_ );

                if ( status != CURAND_STATUS_SUCCESS )
                {
                    curandDestroyGenerator( curand_generator_ );
                    curand_generator_ = nullptr;
                    throw std::runtime_error( "Failed to bind cuRAND generator to stream" );
                }

                curand_initialized_ = true;
            }

            return curand_generator_;
        }

        /**
         * @brief Gets a cuBLASLt handle for this context.
         *
         * Created lazily and owned by this context.
         *
         * @return cublasLtHandle_t The cuBLASLt handle.
         * @throws std::runtime_error If handle creation fails.
         */
        [[nodiscard]] cublasLtHandle_t getCublasLtHandle() const
        {
            if ( !cublas_initialized_ )
            {
                cublasStatus_t status = cublasLtCreate( &cublas_handle_ );

                if ( status != CUBLAS_STATUS_SUCCESS )
                {
                    throw std::runtime_error( "Failed to create cuBLASLt handle" );
                }

                cublas_initialized_ = true;
            }

            return cublas_handle_;
        }

        /**
         * @brief Gets the cuBLASLt workspace buffer.
         *
         * Shared scratch memory for cuBLASLt algorithm execution.
         * Allocated once at context creation, valid for the lifetime of this context.
         * Size is kCublasLtWorkspaceSize bytes.
         *
         * @return void* Pointer to device workspace buffer.
         */
        [[nodiscard]] void* getCublasLtWorkspace() const noexcept
        {
            return cublaslt_workspace_;
        }

        /**
         * @brief Gets the cuBLASLt workspace buffer size in bytes.
         *
         * @return size_t Workspace size in bytes.
         */
        [[nodiscard]] size_t getCublasLtWorkspaceSize() const noexcept
        {
            return cublaslt_workspace_size_;
        }

        /**
         * @brief High-water mark of the device scratch buffer, in bytes.
         *
         * The current size is the high-water by construction: the buffer grows on demand
         * and is never shrunk, so device_scratch_size_ only ever increases until
         * releaseResources(). Reported so the footprint tooling can attribute the gap
         * between what a build allocates and what the driver says was consumed, rather
         * than leaving it as an unexplained margin.
         */
        [[nodiscard]] std::size_t getScratchHighWaterBytes() const noexcept override
        {
            return device_scratch_size_;
        }

        /**
         * @brief Gets or grows a general-purpose device scratch buffer.
         *
         * Used by operations that need a temporary device buffer during forward passes
         * (e.g. FP8->BF16 weight dequantization before a cuBLASLt GEMM). Grown on demand,
         * never shrunk. Because all operations on this context share a single stream, the
         * buffer can be safely reused across sequential ops -- each op finishes before the
         * next one writes to the buffer.
         *
         * Freed in releaseResources().
         *
         * @param required_bytes Minimum number of bytes required.
         * @return void* Device buffer of at least required_bytes.
         * @throws std::runtime_error If allocation fails.
         */
        [[nodiscard]] void* getDeviceScratchBuffer( size_t required_bytes ) const
        {
            if ( required_bytes <= device_scratch_size_ )
                return device_scratch_buf_;

            if ( device_scratch_buf_ )
            {
                cudaFree( device_scratch_buf_ );
                device_scratch_buf_ = nullptr;
                device_scratch_size_ = 0;
            }

            cudaError_t err = cudaMalloc( &device_scratch_buf_, required_bytes );

            if ( err != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "Failed to allocate device scratch buffer: {}",
                        cudaGetErrorString( err ) ) );
            }

            device_scratch_size_ = required_bytes;

            return device_scratch_buf_;
        }

        /**
         * @brief Gets or grows the pinned host staging buffer for Host->Device transfers.
         *
         * Page-locked via cudaHostAlloc so cudaMemcpyAsync from it to device memory
         * is a direct DMA transfer. Grown on demand, never shrunk. Freed in releaseResources().
         *
         * @param required_bytes Minimum number of bytes required.
         * @return void* Pinned host buffer of at least required_bytes.
         * @throws std::runtime_error If allocation fails.
         */
        [[nodiscard]] void* getPinnedStagingBuffer( size_t required_bytes ) const
        {
            if ( required_bytes <= pinned_staging_size_ )
                return pinned_staging_buf_;

            if ( pinned_staging_buf_ )
            {
                cudaFreeHost( pinned_staging_buf_ );
                pinned_staging_buf_ = nullptr;
                pinned_staging_size_ = 0;
            }

            cudaError_t err = cudaHostAlloc( &pinned_staging_buf_, required_bytes, cudaHostAllocDefault );

            if ( err != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "Failed to allocate pinned staging buffer: {}", cudaGetErrorString( err ) ) );
            }

            pinned_staging_size_ = required_bytes;

            return pinned_staging_buf_;
        }

#ifdef USE_CUDNN
        /**
         * @brief Gets a cuDNN handle bound to this context's stream.
         *
         * Created lazily and owned by this context. Bound to the context's stream
         * on each call to ensure correct stream association.
         *
         * @return cudnnHandle_t The cuDNN handle bound to this stream.
         * @throws std::runtime_error If handle creation fails.
         */
        [[nodiscard]] cudnnHandle_t getCudnnHandle() const
        {
            if ( !cudnn_initialized_ )
            {
                cudnnStatus_t status = cudnnCreate( &cudnn_handle_ );

                if ( status != CUDNN_STATUS_SUCCESS )
                {
                    throw std::runtime_error( "Failed to create cuDNN handle" );
                }

                cudnn_initialized_ = true;
            }

            cudnnSetStream( cudnn_handle_, stream_ );

            return cudnn_handle_;
        }
#endif

    private:

        DeviceId device_id_;

        mutable void* device_scratch_buf_{ nullptr };
        mutable size_t device_scratch_size_{ 0 };

        mutable void* pinned_staging_buf_{ nullptr };
        mutable size_t pinned_staging_size_{ 0 };

        mutable void* cublaslt_workspace_{ nullptr };
        mutable size_t cublaslt_workspace_size_{ 0 };
        static constexpr size_t kCublasLtWorkspaceSize = 4ull * 1024 * 1024;

        cudaStream_t stream_{ nullptr };
        bool stream_created_{ false };

        mutable curandGenerator_t curand_generator_{ nullptr };
        mutable bool curand_initialized_{ false };

        mutable cublasLtHandle_t cublas_handle_{ nullptr };
        mutable bool cublas_initialized_{ false };

#ifdef USE_CUDNN
        mutable cudnnHandle_t cudnn_handle_{ nullptr };
        mutable bool cudnn_initialized_{ false };
#endif

        static DeviceId validateDeviceId( DeviceId device_id )
        {
            if ( device_id.type != DeviceType::Cuda )
            {
                throw std::invalid_argument(
                    std::format( "CudaExecutionContext requires Cuda device type, got '{}'",
                                device_id.toString() )
                );
            }

            return device_id;
        }

        void initializeResources()
        {
            // A CUDA stream belongs to whichever device is current when it is created, and
            // nothing guarantees that is this context's device -- device enumeration leaves
            // the last probed device current, so on a multi-GPU host a context for device 0
            // would otherwise get a stream on the last device. Its memory resources bind
            // correctly on every allocation, so the two would disagree and every launch
            // would read pointers belonging to another device. Invisible with one GPU
            // visible, an illegal memory access with two.
            cudaError_t bind_error = cudaSetDevice( device_id_.index );

            if ( bind_error != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "Failed to select CUDA device {} for this context: {}",
                        device_id_.index, cudaGetErrorString( bind_error ) )
                );
            }

            cudaError_t error = cudaStreamCreateWithFlags( &stream_, cudaStreamDefault );

            if ( error != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "Failed to create CUDA stream: {}", cudaGetErrorString( error ) )
                );
            }

            stream_created_ = true;

            cudaError_t ws_error = cudaMalloc( &cublaslt_workspace_, kCublasLtWorkspaceSize );

            if ( ws_error != cudaSuccess )
            {
                cublaslt_workspace_ = nullptr;

                throw std::runtime_error(
                    std::format( "Failed to allocate cuBLASLt workspace: {}",
                        cudaGetErrorString( ws_error ) ) );
            }

            cublaslt_workspace_size_ = kCublasLtWorkspaceSize;
        }

        void releaseResources() noexcept
        {
#ifdef USE_CUDNN
            if ( cudnn_handle_ )
            {
                cudnnDestroy( cudnn_handle_ );
                cudnn_handle_ = nullptr;
            }
#endif

            if ( cublas_handle_ )
            {
                cublasLtDestroy( cublas_handle_ );
                cublas_handle_ = nullptr;
            }

            if ( curand_generator_ )
            {
                curandDestroyGenerator( curand_generator_ );
                curand_generator_ = nullptr;
            }

            if ( cublaslt_workspace_ )
            {
                cudaFree( cublaslt_workspace_ );
                cublaslt_workspace_ = nullptr;
                cublaslt_workspace_size_ = 0;
            }

            if ( device_scratch_buf_ )
            {
                cudaFree( device_scratch_buf_ );
                device_scratch_buf_ = nullptr;
                device_scratch_size_ = 0;
            }

            if ( pinned_staging_buf_ )
            {
                cudaFreeHost( pinned_staging_buf_ );
                pinned_staging_buf_ = nullptr;
                pinned_staging_size_ = 0;
            }

            if ( stream_created_ && stream_ )
            {
                cudaError_t err = cudaStreamSynchronize( stream_ );

                if ( err != cudaSuccess )
                {
                    std::fprintf( stderr,
                        "ExecutionContext: Failed to synchronize CUDA stream: %s\n",
                        cudaGetErrorString( err ) );
                }

                err = cudaStreamDestroy( stream_ );

                if ( err != cudaSuccess )
                {
                    std::fprintf( stderr,
                        "ExecutionContext: Failed to destroy CUDA stream: %s\n",
                        cudaGetErrorString( err ) );
                }

                stream_ = nullptr;
                stream_created_ = false;
            }
        }
    };

    export using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
}