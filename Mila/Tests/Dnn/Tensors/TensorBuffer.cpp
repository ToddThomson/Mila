#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <array>

import Mila;
import Compute.DeviceRegistrar;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorBufferTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Initialize device registrar to register all available devices
            DeviceRegistrar::instance();

            // CPU device uses device_id = -1 or 0
            cpu_device_id_ = 0;

            // Check if CUDA devices are available
            int device_count;
            cudaError_t error = cudaGetDeviceCount( &device_count );

            has_cuda_ = (error == cudaSuccess && device_count > 0);

            if (has_cuda_) {
                cuda_device_id_ = 0; // Use first CUDA device
            }
        }

        void TearDown() override {}

        int cpu_device_id_ = 0;
        int cuda_device_id_ = 0;
        bool has_cuda_ = false;
    };

    // ============================================================================
    // Basic Construction Tests
    // ============================================================================

    TEST_F( TensorBufferTests, CpuBufferConstruction ) {
        // Default construction with size only
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> buffer( cpu_device_id_, 100 );
        EXPECT_EQ( buffer.size(), 100 );
        EXPECT_NE( buffer.data(), nullptr );
        EXPECT_TRUE( buffer.isAligned() );
        EXPECT_FALSE( buffer.empty() );

        // Construction with zero size
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> zero_buffer( cpu_device_id_, 0 );
        EXPECT_EQ( zero_buffer.size(), 0 );
        EXPECT_EQ( zero_buffer.data(), nullptr );
        EXPECT_TRUE( zero_buffer.empty() );
    }

    TEST_F( TensorBufferTests, CudaBufferConstruction ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        TensorBuffer<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> buffer( cuda_device_id_, 100 );
        EXPECT_EQ( buffer.size(), 100 );
        EXPECT_NE( buffer.data(), nullptr );

        // Construction with zero size
        TensorBuffer<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> zero_buffer( cuda_device_id_, 0 );
        EXPECT_EQ( zero_buffer.size(), 0 );
        EXPECT_EQ( zero_buffer.data(), nullptr );
        EXPECT_TRUE( zero_buffer.empty() );
    }

    // ============================================================================
    // Move Semantics Tests
    // ============================================================================

    TEST_F( TensorBufferTests, MoveConstructor ) {
        // Create original buffer
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> original( cpu_device_id_, 100 );
        auto original_ptr = original.data();
        auto original_size = original.size();

        // Move construct
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> moved( std::move( original ) );

        // Verify moved buffer has original's data
        EXPECT_EQ( moved.size(), original_size );
        EXPECT_EQ( moved.data(), original_ptr );

        // Verify original is in valid but empty state
        EXPECT_EQ( original.size(), 0 );
        EXPECT_EQ( original.data(), nullptr );
        EXPECT_TRUE( original.empty() );
    }

    TEST_F( TensorBufferTests, MoveAssignment ) {
        // Create buffers
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> source( cpu_device_id_, 100 );
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> target( cpu_device_id_, 50 );

        auto source_ptr = source.data();
        auto source_size = source.size();

        // Move assign
        target = std::move( source );

        // Verify target has source's data
        EXPECT_EQ( target.size(), source_size );
        EXPECT_EQ( target.data(), source_ptr );

        // Verify source is in valid but empty state
        EXPECT_EQ( source.size(), 0 );
        EXPECT_EQ( source.data(), nullptr );
    }

    TEST_F( TensorBufferTests, MoveAssignmentSelfAssignment ) {
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> buffer( cpu_device_id_, 100 );
        auto original_ptr = buffer.data();
        auto original_size = buffer.size();

        // Self-assignment should be safe
        buffer = std::move( buffer );

        EXPECT_EQ( buffer.size(), original_size );
        EXPECT_EQ( buffer.data(), original_ptr );
    }

    // ============================================================================
    // Resize Operation Tests
    // ============================================================================

    TEST_F( TensorBufferTests, CpuBufferResize ) {
        TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource> buffer( cpu_device_id_, 50 );

        // Resize larger
        buffer.resize( 100 );
        EXPECT_EQ( buffer.size(), 100 );
        EXPECT_NE( buffer.data(), nullptr );

        // Resize smaller
        buffer.resize( 25 );
        EXPECT_EQ( buffer.size(), 25 );
        EXPECT_NE( buffer.data(), nullptr );

        // Resize to zero
        buffer.resize( 0 );
        EXPECT_EQ( buffer.size(), 0 );
        EXPECT_EQ( buffer.data(), nullptr );
        EXPECT_TRUE( buffer.empty() );

        // Resize from zero
        buffer.resize( 10 );
        EXPECT_EQ( buffer.size(), 10 );
        EXPECT_NE( buffer.data(), nullptr );

        // Resize to same size (should be no-op)
        auto original_ptr = buffer.data();
        buffer.resize( 10 );
        EXPECT_EQ( buffer.data(), original_ptr );
    }

    TEST_F( TensorBufferTests, CudaBufferResize ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        TensorBuffer<TensorDataType::INT32, Compute::CudaDeviceMemoryResource> buffer( cuda_device_id_, 50 );
        auto original_ptr = buffer.data();

        // Resize larger
        buffer.resize( 100 );
        EXPECT_EQ( buffer.size(), 100 );
        EXPECT_NE( buffer.data(), original_ptr );

        // Resize smaller
        buffer.resize( 25 );
        EXPECT_EQ( buffer.size(), 25 );

        // Resize to zero
        buffer.resize( 0 );
        EXPECT_EQ( buffer.size(), 0 );
        EXPECT_EQ( buffer.data(), nullptr );
    }

    // ============================================================================
    // Error Handling and Edge Cases
    // ============================================================================

    TEST_F( TensorBufferTests, OverflowExceptions ) {
        // Very large allocation (should throw overflow_error)
        EXPECT_THROW( (TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource>(
            cpu_device_id_, std::numeric_limits<size_t>::max() )),
            std::overflow_error );

        // Overflow in aligned_size calculation in resize
        TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource> buffer( cpu_device_id_, 10 );
        EXPECT_THROW( buffer.resize( std::numeric_limits<size_t>::max() ),
            std::overflow_error );
    }

    // ============================================================================
    // Memory Tracking Tests
    // ============================================================================

    TEST_F( TensorBufferTests, MemoryTracking ) {
        // Test with memory tracking enabled
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource, true> tracked_buffer( cpu_device_id_, 100 );
        EXPECT_EQ( tracked_buffer.size(), 100 );
        EXPECT_NE( tracked_buffer.data(), nullptr );

        tracked_buffer.resize( 200 );
        EXPECT_EQ( tracked_buffer.size(), 200 );
    }

    TEST_F( TensorBufferTests, MemoryTrackingWithAllTypes ) {
        // Integer types with tracking
        TensorBuffer<TensorDataType::INT16, Compute::CpuMemoryResource, true> tracked_int16_buffer( cpu_device_id_, 10 );
        TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource, true> tracked_int32_buffer( cpu_device_id_, 10 );
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource, true> tracked_float_buffer( cpu_device_id_, 10 );

        if (has_cuda_) {
            TensorBuffer<TensorDataType::FP32, Compute::CudaDeviceMemoryResource, true> tracked_cuda_float_buffer( cuda_device_id_, 10 );
            TensorBuffer<TensorDataType::FP16, Compute::CudaDeviceMemoryResource, true> tracked_cuda_half_buffer( cuda_device_id_, 10 );

            tracked_cuda_float_buffer.resize( 20 );
            tracked_cuda_half_buffer.resize( 20 );

            EXPECT_EQ( tracked_cuda_float_buffer.size(), 20 );
            EXPECT_EQ( tracked_cuda_half_buffer.size(), 20 );
        }

        tracked_int16_buffer.resize( 20 );
        tracked_int32_buffer.resize( 20 );
        tracked_float_buffer.resize( 20 );

        EXPECT_EQ( tracked_int16_buffer.size(), 20 );
        EXPECT_EQ( tracked_int32_buffer.size(), 20 );
        EXPECT_EQ( tracked_float_buffer.size(), 20 );
    }

    // ============================================================================
    // Data Type Tests
    // ============================================================================

    TEST_F( TensorBufferTests, DifferentDataTypes ) {
        // Integer types
        TensorBuffer<TensorDataType::INT16, Compute::CpuMemoryResource> int16_buffer( cpu_device_id_, 10 );
        TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource> int32_buffer( cpu_device_id_, 10 );
        TensorBuffer<TensorDataType::UINT16, Compute::CpuMemoryResource> uint16_buffer( cpu_device_id_, 10 );
        TensorBuffer<TensorDataType::UINT32, Compute::CpuMemoryResource> uint32_buffer( cpu_device_id_, 10 );
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> float_buffer( cpu_device_id_, 10 );

        EXPECT_EQ( int16_buffer.size(), 10 );
        EXPECT_EQ( int32_buffer.size(), 10 );
        EXPECT_EQ( uint16_buffer.size(), 10 );
        EXPECT_EQ( uint32_buffer.size(), 10 );
        EXPECT_EQ( float_buffer.size(), 10 );

        if (has_cuda_) {
            TensorBuffer<TensorDataType::INT16, Compute::CudaDeviceMemoryResource> cuda_int16_buffer( cuda_device_id_, 10 );
            TensorBuffer<TensorDataType::INT32, Compute::CudaDeviceMemoryResource> cuda_int32_buffer( cuda_device_id_, 10 );
            TensorBuffer<TensorDataType::UINT16, Compute::CudaDeviceMemoryResource> cuda_uint16_buffer( cuda_device_id_, 10 );
            TensorBuffer<TensorDataType::UINT32, Compute::CudaDeviceMemoryResource> cuda_uint32_buffer( cuda_device_id_, 10 );
            TensorBuffer<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> cuda_float_buffer( cuda_device_id_, 10 );
            TensorBuffer<TensorDataType::FP16, Compute::CudaDeviceMemoryResource> cuda_half_buffer( cuda_device_id_, 10 );
            TensorBuffer<TensorDataType::BF16, Compute::CudaDeviceMemoryResource> cuda_bf16_buffer( cuda_device_id_, 10 );
            TensorBuffer<TensorDataType::FP8_E4M3, Compute::CudaDeviceMemoryResource> cuda_fp8_e4m3_buffer( cuda_device_id_, 10 );
            TensorBuffer<TensorDataType::FP8_E5M2, Compute::CudaDeviceMemoryResource> cuda_fp8_e5m2_buffer( cuda_device_id_, 10 );

            EXPECT_EQ( cuda_int16_buffer.size(), 10 );
            EXPECT_EQ( cuda_int32_buffer.size(), 10 );
            EXPECT_EQ( cuda_uint16_buffer.size(), 10 );
            EXPECT_EQ( cuda_uint32_buffer.size(), 10 );
            EXPECT_EQ( cuda_float_buffer.size(), 10 );
            EXPECT_EQ( cuda_half_buffer.size(), 10 );
            EXPECT_EQ( cuda_bf16_buffer.size(), 10 );
            EXPECT_EQ( cuda_fp8_e4m3_buffer.size(), 10 );
            EXPECT_EQ( cuda_fp8_e5m2_buffer.size(), 10 );

            EXPECT_TRUE( cuda_int16_buffer.isAligned() );
            EXPECT_TRUE( cuda_int32_buffer.isAligned() );
            EXPECT_TRUE( cuda_uint16_buffer.isAligned() );
            EXPECT_TRUE( cuda_uint32_buffer.isAligned() );
            EXPECT_TRUE( cuda_float_buffer.isAligned() );
            EXPECT_TRUE( cuda_half_buffer.isAligned() );
            EXPECT_TRUE( cuda_bf16_buffer.isAligned() );
            EXPECT_TRUE( cuda_fp8_e4m3_buffer.isAligned() );
            EXPECT_TRUE( cuda_fp8_e5m2_buffer.isAligned() );
        }

        EXPECT_TRUE( int16_buffer.isAligned() );
        EXPECT_TRUE( int32_buffer.isAligned() );
        EXPECT_TRUE( uint16_buffer.isAligned() );
        EXPECT_TRUE( uint32_buffer.isAligned() );
        EXPECT_TRUE( float_buffer.isAligned() );
    }

    // ============================================================================
    // Properties and Metadata Tests
    // ============================================================================

    TEST_F( TensorBufferTests, CompileTimeProperties ) {
        using FP32Buffer = TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource>;
        using INT32Buffer = TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource>;

        EXPECT_EQ( FP32Buffer::data_type, TensorDataType::FP32 );
        EXPECT_EQ( INT32Buffer::data_type, TensorDataType::INT32 );
        EXPECT_EQ( FP32Buffer::element_size, sizeof( float ) );
        EXPECT_EQ( INT32Buffer::element_size, sizeof( int32_t ) );
        EXPECT_TRUE( FP32Buffer::is_float_type );
        EXPECT_FALSE( FP32Buffer::is_integer_type );
        EXPECT_FALSE( INT32Buffer::is_float_type );
        EXPECT_TRUE( INT32Buffer::is_integer_type );

        if (has_cuda_) {
            using FP16Buffer = TensorBuffer<TensorDataType::FP16, Compute::CudaDeviceMemoryResource>;
            EXPECT_EQ( FP16Buffer::data_type, TensorDataType::FP16 );
            EXPECT_EQ( FP16Buffer::element_size, sizeof( half ) );
            EXPECT_TRUE( FP16Buffer::is_float_type );
            EXPECT_FALSE( FP16Buffer::is_integer_type );
        }
    }

    TEST_F( TensorBufferTests, VerifyAlignment ) {
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> cpu_buffer( cpu_device_id_, 100 );
        EXPECT_TRUE( cpu_buffer.isAligned() );
        EXPECT_EQ( (cpu_buffer.alignedSize() % TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource>::alignment), 0 );

        if (has_cuda_) {
            TensorBuffer<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> cuda_buffer( cuda_device_id_, 100 );
            EXPECT_TRUE( cuda_buffer.isAligned() );
            EXPECT_EQ( (cuda_buffer.alignedSize() % TensorBuffer<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>::alignment), 0 );
        }
    }

    TEST_F( TensorBufferTests, VerifyAlignedSize ) {
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> cpu_buffer( cpu_device_id_, 100 );
        size_t expected_aligned_size = (100 * sizeof( float ) + TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource>::alignment - 1)
            & ~(TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource>::alignment - 1);
        EXPECT_EQ( cpu_buffer.alignedSize(), expected_aligned_size );
    }

    TEST_F( TensorBufferTests, StorageBytes ) {
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> buffer( cpu_device_id_, 100 );
        EXPECT_EQ( buffer.storageBytes(), 100 * sizeof( float ) );

        if (has_cuda_) {
            TensorBuffer<TensorDataType::FP16, Compute::CudaDeviceMemoryResource> half_buffer( cuda_device_id_, 50 );
            EXPECT_EQ( half_buffer.storageBytes(), 50 * sizeof( half ) );
        }

        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> empty_buffer( cpu_device_id_, 0 );
        EXPECT_EQ( empty_buffer.storageBytes(), 0 );
    }

    TEST_F( TensorBufferTests, ConstRawDataAccess ) {
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> buffer( cpu_device_id_, 10 );
        const auto& const_buffer = buffer;
        const void* const_ptr = const_buffer.data();
        EXPECT_NE( const_ptr, nullptr );
        EXPECT_EQ( const_ptr, buffer.data() );
    }

    // ============================================================================
    // Performance Tests
    // ============================================================================

    TEST_F( TensorBufferTests, LargeBufferPerformance ) {
        constexpr size_t large_size = 10000000;
        auto start = std::chrono::high_resolution_clock::now();
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> buffer( cpu_device_id_, large_size );
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        EXPECT_LT( duration.count(), 1000 );
    }

    // ============================================================================
    // Zero Initialization Validation
    // ============================================================================

    TEST_F( TensorBufferTests, ZeroInitializationValidation ) {
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> float_buffer( cpu_device_id_, 100 );
        auto float_data = static_cast<float*>(float_buffer.data());
        for (size_t i = 0; i < 100; ++i) {
            EXPECT_FLOAT_EQ( float_data[i], 0.0f );
        }

        TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource> int_buffer( cpu_device_id_, 50 );
        auto int_data = static_cast<int32_t*>( int_buffer.data() );
        for (size_t i = 0; i < 50; ++i) {
            EXPECT_EQ( int_data[i], 0 );
        }

        if (has_cuda_) {
            TensorBuffer<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> cuda_buffer( cuda_device_id_, 10 );
            std::vector<float> host_data( 10 );
            cudaMemcpy( host_data.data(), cuda_buffer.data(), 10 * sizeof( float ), cudaMemcpyDeviceToHost );
            for (float val : host_data) {
                EXPECT_FLOAT_EQ( val, 0.0f );
            }
        }
    }

    // ============================================================================
    // Buffer Data Preservation Tests
    // ============================================================================

    TEST_F( TensorBufferTests, ResizeDataPreservation ) {
        TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource> buffer( cpu_device_id_, 10 );

        // Initialize with test pattern
        auto data = static_cast<int32_t*>(buffer.data());
        for (size_t i = 0; i < 10; ++i) {
            data[i] = static_cast<int32_t>( i * 10 );
        }

        // Resize larger
        buffer.resize( 20 );
        auto new_data = static_cast<int32_t*>( buffer.data() );
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ( new_data[i], static_cast<int32_t>( i * 10 ) );
        }

        // Verify new elements are zero-initialized
        for (size_t i = 10; i < 20; ++i) {
            EXPECT_EQ( new_data[i], 0 );
        }

        // Resize smaller
        buffer.resize( 5 );
        auto smaller_data = static_cast<int32_t*>( buffer.data() );
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ( smaller_data[i], static_cast<int32_t>( i * 10 ) );
        }
    }

    // ============================================================================
    // Architecture Compliance Tests
    // ============================================================================

    TEST_F( TensorBufferTests, MemoryResourceIntegration ) {
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> cpu_buffer( cpu_device_id_, 100 );
        EXPECT_NE( cpu_buffer.getMemoryResource(), nullptr );

        if (has_cuda_) {
            TensorBuffer<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> cuda_buffer( cuda_device_id_, 100 );
            EXPECT_NE( cuda_buffer.getMemoryResource(), nullptr );
        }
    }

    TEST_F( TensorBufferTests, ArchitecturalSeparationOfConcerns ) {
        // TensorBuffer focuses purely on memory management
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> buffer( cpu_device_id_, 100 );

        // Memory management primitives
        EXPECT_NE( buffer.data(), nullptr );
        EXPECT_EQ( buffer.size(), 100 );
        EXPECT_GT( buffer.storageBytes(), 0 );
        EXPECT_GT( buffer.alignedSize(), 0 );

        // Resize for owned memory
        EXPECT_NO_THROW( buffer.resize( 200 ) );
        EXPECT_EQ( buffer.size(), 200 );
    }
}