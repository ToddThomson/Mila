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
import Compute.CpuDeviceContext;
import Compute.CudaDeviceContext;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorBufferTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Initialize device registrar to register all available devices
            DeviceRegistrar::instance();

            // Create CPU device context (using new parameterless constructor)
            cpu_context_ = std::make_shared<CpuDeviceContext>();

            // Check if CUDA devices are available before creating CUDA context
            int device_count;
            cudaError_t error = cudaGetDeviceCount( &device_count );

            has_cuda_ = (error == cudaSuccess && device_count > 0);

            if (has_cuda_) {
                cuda_context_ = std::make_shared<CudaDeviceContext>( "CUDA:0" );
            }
        }

        void TearDown() override {}

        std::shared_ptr<DeviceContext> cpu_context_;
        std::shared_ptr<DeviceContext> cuda_context_;
        bool has_cuda_ = false;
    };

    // ============================================================================
    // Basic Construction Tests
    // ============================================================================

    TEST_F( TensorBufferTests, CpuBufferConstruction ) {
        // Default construction with size only
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> buffer( cpu_context_, 100 );
        EXPECT_EQ( buffer.size(), 100 );
        EXPECT_NE( buffer.rawData(), nullptr );
        EXPECT_TRUE( buffer.isAligned() );
        EXPECT_FALSE( buffer.empty() );
        EXPECT_EQ( buffer.getDeviceContext(), cpu_context_ );

        // Construction with zero size
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> zero_buffer( cpu_context_, 0 );
        EXPECT_EQ( zero_buffer.size(), 0 );
        EXPECT_EQ( zero_buffer.rawData(), nullptr );
        EXPECT_TRUE( zero_buffer.empty() );
        EXPECT_EQ( zero_buffer.getDeviceContext(), cpu_context_ );
    }

    TEST_F( TensorBufferTests, CudaBufferConstruction ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        TensorBuffer<TensorDataType::FP32, Compute::CudaMemoryResource> buffer( cuda_context_, 100 );
        EXPECT_EQ( buffer.size(), 100 );
        EXPECT_NE( buffer.rawData(), nullptr );
        EXPECT_EQ( buffer.getDeviceContext(), cuda_context_ );

        // Construction with zero size
        TensorBuffer<TensorDataType::FP32, Compute::CudaMemoryResource> zero_buffer( cuda_context_, 0 );
        EXPECT_EQ( zero_buffer.size(), 0 );
        EXPECT_EQ( zero_buffer.rawData(), nullptr );
        EXPECT_TRUE( zero_buffer.empty() );
        EXPECT_EQ( zero_buffer.getDeviceContext(), cuda_context_ );
    }

    TEST_F( TensorBufferTests, DeviceContextValidation ) {
        // Test null device context throws
        std::shared_ptr<DeviceContext> null_context;
        EXPECT_THROW(
            (TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource>( null_context, 100 )),
            std::invalid_argument
        );
    }

    // ============================================================================
    // External Memory Tests
    // ============================================================================

    TEST_F( TensorBufferTests, CpuBufferExternalMemory ) {
        std::vector<float> external_data( 100, 3.14f );
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> buffer(
            cpu_context_,
            external_data.size(),
            reinterpret_cast<std::byte*>(external_data.data()),
            external_data.size() * sizeof( float ) );

        EXPECT_EQ( buffer.size(), external_data.size() );
        EXPECT_EQ( buffer.rawData(), external_data.data() );
        EXPECT_EQ( buffer.getDeviceContext(), cpu_context_ );

        // Verify resize throws for external buffer (no memory resource available)
        EXPECT_THROW( buffer.resize( 50 ), std::runtime_error );
    }

    TEST_F( TensorBufferTests, ExternalMemoryValidation ) {
        // Null device context
        std::shared_ptr<DeviceContext> null_context;
        std::vector<int> valid_buffer( 100 );
        EXPECT_THROW( (TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource>(
            null_context, 100, reinterpret_cast<std::byte*>(valid_buffer.data()), 100 * sizeof( int ) )),
            std::invalid_argument );

        // Null external pointer
        std::byte* null_ptr = nullptr;
        EXPECT_THROW( (TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource>(
            cpu_context_, 100, null_ptr, 1000 )),
            std::invalid_argument );

        // Insufficient external memory size
        std::vector<int> small_buffer( 10 );
        EXPECT_THROW( (TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource>(
            cpu_context_, 100, reinterpret_cast<std::byte*>(small_buffer.data()), 10 * sizeof( int ) )),
            std::invalid_argument );

        // Exact size requirement (should work)
        std::vector<int> exact_buffer( 50 );
        EXPECT_NO_THROW( (TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource>(
            cpu_context_, 50, reinterpret_cast<std::byte*>(exact_buffer.data()), 50 * sizeof( int ) )) );
    }

    // ============================================================================
    // Move Semantics Tests
    // ============================================================================

    TEST_F( TensorBufferTests, MoveConstructor ) {
        // Create original buffer
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> original( cpu_context_, 100 );
        auto original_ptr = original.rawData();
        auto original_size = original.size();
        auto original_context = original.getDeviceContext();

        // Move construct
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> moved( std::move( original ) );

        // Verify moved buffer has original's data
        EXPECT_EQ( moved.size(), original_size );
        EXPECT_EQ( moved.rawData(), original_ptr );
        EXPECT_EQ( moved.getDeviceContext(), original_context );

        // Verify original is in valid but empty state
        EXPECT_EQ( original.size(), 0 );
        EXPECT_EQ( original.rawData(), nullptr );
        EXPECT_TRUE( original.empty() );
    }

    TEST_F( TensorBufferTests, MoveAssignment ) {
        // Create buffers
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> source( cpu_context_, 100 );
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> target( cpu_context_, 50 );

        auto source_ptr = source.rawData();
        auto source_size = source.size();
        auto source_context = source.getDeviceContext();

        // Move assign
        target = std::move( source );

        // Verify target has source's data
        EXPECT_EQ( target.size(), source_size );
        EXPECT_EQ( target.rawData(), source_ptr );
        EXPECT_EQ( target.getDeviceContext(), source_context );

        // Verify source is in valid but empty state
        EXPECT_EQ( source.size(), 0 );
        EXPECT_EQ( source.rawData(), nullptr );
    }

    TEST_F( TensorBufferTests, MoveAssignmentSelfAssignment ) {
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> buffer( cpu_context_, 100 );
        auto original_ptr = buffer.rawData();
        auto original_size = buffer.size();
        auto original_context = buffer.getDeviceContext();

        // Self-assignment should be safe
        buffer = std::move( buffer );

        EXPECT_EQ( buffer.size(), original_size );
        EXPECT_EQ( buffer.rawData(), original_ptr );
        EXPECT_EQ( buffer.getDeviceContext(), original_context );
    }

    // ============================================================================
    // Resize Operation Tests
    // ============================================================================

    TEST_F( TensorBufferTests, CpuBufferResize ) {
        TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource> buffer( cpu_context_, 50 );

        // Resize larger - verify basic functionality
        buffer.resize( 100 );
        EXPECT_EQ( buffer.size(), 100 );
        EXPECT_NE( buffer.rawData(), nullptr );

        // Resize smaller
        buffer.resize( 25 );
        EXPECT_EQ( buffer.size(), 25 );
        EXPECT_NE( buffer.rawData(), nullptr );

        // Resize to zero
        buffer.resize( 0 );
        EXPECT_EQ( buffer.size(), 0 );
        EXPECT_EQ( buffer.rawData(), nullptr );
        EXPECT_TRUE( buffer.empty() );

        // Resize from zero
        buffer.resize( 10 );
        EXPECT_EQ( buffer.size(), 10 );
        EXPECT_NE( buffer.rawData(), nullptr );

        // Resize to same size (should be no-op)
        auto original_ptr = buffer.rawData();
        buffer.resize( 10 );
        EXPECT_EQ( buffer.rawData(), original_ptr );
    }

    TEST_F( TensorBufferTests, CudaBufferResize ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        TensorBuffer<TensorDataType::INT32, Compute::CudaMemoryResource> buffer( cuda_context_, 50 );
        auto original_ptr = buffer.rawData();

        // Resize larger
        buffer.resize( 100 );
        EXPECT_EQ( buffer.size(), 100 );
        EXPECT_NE( buffer.rawData(), original_ptr );

        // Resize smaller
        buffer.resize( 25 );
        EXPECT_EQ( buffer.size(), 25 );

        // Resize to zero
        buffer.resize( 0 );
        EXPECT_EQ( buffer.size(), 0 );
        EXPECT_EQ( buffer.rawData(), nullptr );
    }

    TEST_F( TensorBufferTests, ExternalMemoryResizeRestriction ) {
        std::vector<float> external_data( 100, 1.5f );
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> external_buffer(
            cpu_context_,
            external_data.size(),
            reinterpret_cast<std::byte*>(external_data.data()),
            external_data.size() * sizeof( float ) );

        // Verify resize throws for external buffer (no memory resource management)
        EXPECT_THROW( external_buffer.resize( 50 ), std::runtime_error );
    }

    // ============================================================================
    // Error Handling and Edge Cases
    // ============================================================================

    TEST_F( TensorBufferTests, OverflowExceptions ) {
        // Very large allocation (should throw overflow_error)
        EXPECT_THROW( (TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource>(
            cpu_context_, std::numeric_limits<size_t>::max() )),
            std::overflow_error );

        // Overflow in aligned_size calculation in resize
        TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource> buffer( cpu_context_, 10 );
        EXPECT_THROW( buffer.resize( std::numeric_limits<size_t>::max() ),
            std::overflow_error );
    }

    // ============================================================================
    // Memory Tracking Tests
    // ============================================================================

    TEST_F( TensorBufferTests, MemoryTracking ) {
        // Test with memory tracking enabled
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource, true> tracked_buffer( cpu_context_, 100 );
        EXPECT_EQ( tracked_buffer.size(), 100 );
        EXPECT_NE( tracked_buffer.rawData(), nullptr );
        EXPECT_EQ( tracked_buffer.getDeviceContext(), cpu_context_ );

        // The memory tracking should work without errors
        // (actual output verification would require capturing stdout)
        tracked_buffer.resize( 200 );
        EXPECT_EQ( tracked_buffer.size(), 200 );
    }

    TEST_F( TensorBufferTests, MemoryTrackingWithAllTypes ) {
        // Test memory tracking with various types

        // Integer types with tracking
        TensorBuffer<TensorDataType::INT16, Compute::CpuMemoryResource, true> tracked_int16_buffer( cpu_context_, 10 );
        TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource, true> tracked_int32_buffer( cpu_context_, 10 );

        // Float type with tracking
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource, true> tracked_float_buffer( cpu_context_, 10 );

        if (has_cuda_) {
            // CUDA types with tracking
            TensorBuffer<TensorDataType::FP32, Compute::CudaMemoryResource, true> tracked_cuda_float_buffer( cuda_context_, 10 );
            TensorBuffer<TensorDataType::FP16, Compute::CudaMemoryResource, true> tracked_cuda_half_buffer( cuda_context_, 10 );

            // Test basic operations with tracked buffers
            tracked_cuda_float_buffer.resize( 20 );
            tracked_cuda_half_buffer.resize( 20 );

            EXPECT_EQ( tracked_cuda_float_buffer.size(), 20 );
            EXPECT_EQ( tracked_cuda_half_buffer.size(), 20 );
        }

        // Test basic operations with tracked buffers
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
        // Test with CPU memory resource for host-compatible types

        // Integer types
        TensorBuffer<TensorDataType::INT16, Compute::CpuMemoryResource> int16_buffer( cpu_context_, 10 );
        TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource> int32_buffer( cpu_context_, 10 );
        TensorBuffer<TensorDataType::UINT16, Compute::CpuMemoryResource> uint16_buffer( cpu_context_, 10 );
        TensorBuffer<TensorDataType::UINT32, Compute::CpuMemoryResource> uint32_buffer( cpu_context_, 10 );

        // Floating point types for CPU
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> float_buffer( cpu_context_, 10 );

        EXPECT_EQ( int16_buffer.size(), 10 );
        EXPECT_EQ( int32_buffer.size(), 10 );
        EXPECT_EQ( uint16_buffer.size(), 10 );
        EXPECT_EQ( uint32_buffer.size(), 10 );
        EXPECT_EQ( float_buffer.size(), 10 );

        if (has_cuda_) {
            // Test with CUDA memory resource for all types (including device-only types)

            // Integer types
            TensorBuffer<TensorDataType::INT16, Compute::CudaMemoryResource> cuda_int16_buffer( cuda_context_, 10 );
            TensorBuffer<TensorDataType::INT32, Compute::CudaMemoryResource> cuda_int32_buffer( cuda_context_, 10 );
            TensorBuffer<TensorDataType::UINT16, Compute::CudaMemoryResource> cuda_uint16_buffer( cuda_context_, 10 );
            TensorBuffer<TensorDataType::UINT32, Compute::CudaMemoryResource> cuda_uint32_buffer( cuda_context_, 10 );

            // Floating point types for CUDA
            TensorBuffer<TensorDataType::FP32, Compute::CudaMemoryResource> cuda_float_buffer( cuda_context_, 10 );
            TensorBuffer<TensorDataType::FP16, Compute::CudaMemoryResource> cuda_half_buffer( cuda_context_, 10 );
            TensorBuffer<TensorDataType::BF16, Compute::CudaMemoryResource> cuda_bf16_buffer( cuda_context_, 10 );
            TensorBuffer<TensorDataType::FP8_E4M3, Compute::CudaMemoryResource> cuda_fp8_e4m3_buffer( cuda_context_, 10 );
            TensorBuffer<TensorDataType::FP8_E5M2, Compute::CudaMemoryResource> cuda_fp8_e5m2_buffer( cuda_context_, 10 );

            EXPECT_EQ( cuda_int16_buffer.size(), 10 );
            EXPECT_EQ( cuda_int32_buffer.size(), 10 );
            EXPECT_EQ( cuda_uint16_buffer.size(), 10 );
            EXPECT_EQ( cuda_uint32_buffer.size(), 10 );
            EXPECT_EQ( cuda_float_buffer.size(), 10 );
            EXPECT_EQ( cuda_half_buffer.size(), 10 );
            EXPECT_EQ( cuda_bf16_buffer.size(), 10 );
            EXPECT_EQ( cuda_fp8_e4m3_buffer.size(), 10 );
            EXPECT_EQ( cuda_fp8_e5m2_buffer.size(), 10 );

            // Check alignment for CUDA types
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

        // Check alignment for CPU types
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
        // Test compile-time constants for different data types
        using FP32Buffer = TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource>;
        using INT32Buffer = TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource>;

        // Check data type constants
        EXPECT_EQ( FP32Buffer::data_type, TensorDataType::FP32 );
        EXPECT_EQ( INT32Buffer::data_type, TensorDataType::INT32 );

        // Check element sizes
        EXPECT_EQ( FP32Buffer::element_size, sizeof( float ) );
        EXPECT_EQ( INT32Buffer::element_size, sizeof( int32_t ) );

        // Check type classifications
        EXPECT_TRUE( FP32Buffer::is_float_type );
        EXPECT_FALSE( FP32Buffer::is_integer_type );
        EXPECT_FALSE( INT32Buffer::is_float_type );
        EXPECT_TRUE( INT32Buffer::is_integer_type );

        if (has_cuda_) {
            using FP16Buffer = TensorBuffer<TensorDataType::FP16, Compute::CudaMemoryResource>;

            EXPECT_EQ( FP16Buffer::data_type, TensorDataType::FP16 );
            EXPECT_EQ( FP16Buffer::element_size, sizeof( half ) );
            EXPECT_TRUE( FP16Buffer::is_float_type );
            EXPECT_FALSE( FP16Buffer::is_integer_type );
        }
    }

    TEST_F( TensorBufferTests, VerifyAlignment ) {
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> cpu_buffer( cpu_context_, 100 );

        EXPECT_TRUE( cpu_buffer.isAligned() );
        EXPECT_EQ( (cpu_buffer.alignedSize() % TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource>::alignment), 0 );

        if (has_cuda_) {
            TensorBuffer<TensorDataType::FP32, Compute::CudaMemoryResource> cuda_buffer( cuda_context_, 100 );
            EXPECT_TRUE( cuda_buffer.isAligned() );
            EXPECT_EQ( (cuda_buffer.alignedSize() % TensorBuffer<TensorDataType::FP32, Compute::CudaMemoryResource>::alignment), 0 );
        }
    }

    TEST_F( TensorBufferTests, VerifyAlignedSize ) {
        // Test that alignedSize works correctly
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> cpu_buffer( cpu_context_, 100 );
        size_t expected_aligned_size = (100 * sizeof( float ) + TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource>::alignment - 1)
            & ~(TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource>::alignment - 1);
        EXPECT_EQ( cpu_buffer.alignedSize(), expected_aligned_size );
    }

    TEST_F( TensorBufferTests, StorageBytes ) {
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> buffer( cpu_context_, 100 );
        EXPECT_EQ( buffer.storageBytes(), 100 * sizeof( float ) );

        if (has_cuda_) {
            TensorBuffer<TensorDataType::FP16, Compute::CudaMemoryResource> half_buffer( cuda_context_, 50 );
            EXPECT_EQ( half_buffer.storageBytes(), 50 * sizeof( half ) );
        }

        // Test zero-size buffer
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> empty_buffer( cpu_context_, 0 );
        EXPECT_EQ( empty_buffer.storageBytes(), 0 );
    }

    TEST_F( TensorBufferTests, ConstRawDataAccess ) {
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> buffer( cpu_context_, 10 );

        // Test const rawData() method
        const auto& const_buffer = buffer;
        const void* const_ptr = const_buffer.rawData();
        EXPECT_NE( const_ptr, nullptr );
        EXPECT_EQ( const_ptr, buffer.rawData() );
    }

    // ============================================================================
    // Performance Tests
    // ============================================================================

    TEST_F( TensorBufferTests, LargeBufferPerformance ) {
        constexpr size_t large_size = 10000000;

        auto start = std::chrono::high_resolution_clock::now();
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> buffer( cpu_context_, large_size );
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        EXPECT_LT( duration.count(), 1000 ); // Should allocate within 1 second
    }

    // ============================================================================
    // Zero Initialization Validation
    // ============================================================================

    TEST_F( TensorBufferTests, ZeroInitializationValidation ) {
        // Test that buffers are zero-initialized by default
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> float_buffer( cpu_context_, 100 );
        auto float_data = static_cast<float*>(float_buffer.rawData());
        for (size_t i = 0; i < 100; ++i) {
            EXPECT_FLOAT_EQ( float_data[i], 0.0f );
        }

        TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource> int_buffer( cpu_context_, 50 );
        auto int_data = static_cast<int32_t*>( int_buffer.rawData() );
        for (size_t i = 0; i < 50; ++i) {
            EXPECT_EQ( int_data[i], 0 );
        }

        if (has_cuda_) {
            // Test CUDA buffer zero initialization
            TensorBuffer<TensorDataType::FP32, Compute::CudaMemoryResource> cuda_buffer( cuda_context_, 10 );
            std::vector<float> host_data( 10 );
            cudaMemcpy( host_data.data(), cuda_buffer.rawData(), 10 * sizeof( float ), cudaMemcpyDeviceToHost );
            for (float val : host_data) {
                EXPECT_FLOAT_EQ( val, 0.0f );
            }
        }
    }

    // ============================================================================
    // Buffer Data Preservation Tests 
    // ============================================================================

    TEST_F( TensorBufferTests, ResizeDataPreservation ) {
        TensorBuffer<TensorDataType::INT32, Compute::CpuMemoryResource> buffer( cpu_context_, 10 );

        // Initialize with test pattern
        auto data = static_cast<int32_t*>(buffer.rawData());
        for (size_t i = 0; i < 10; ++i) {
            data[i] = static_cast<int32_t>( i * 10 );
        }

        // Resize larger - verify existing data preserved
        buffer.resize( 20 );
        auto new_data = static_cast<int32_t*>( buffer.rawData() );
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ( new_data[i], static_cast<int32_t>( i * 10 ) );
        }

        // Verify new elements are zero-initialized
        for (size_t i = 10; i < 20; ++i) {
            EXPECT_EQ( new_data[i], 0 );
        }

        // Resize smaller - verify remaining data preserved
        buffer.resize( 5 );
        auto smaller_data = static_cast<int32_t*>( buffer.rawData() );
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ( smaller_data[i], static_cast<int32_t>( i * 10 ) );
        }
    }

    // ============================================================================
    // Architecture Compliance Tests
    // ============================================================================

    TEST_F( TensorBufferTests, MemoryResourceIntegration ) {
        // Verify TensorBuffer properly integrates with memory resources
        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> cpu_buffer( cpu_context_, 100 );

        // Buffer should have valid device context
        EXPECT_NE( cpu_buffer.getDeviceContext(), nullptr );
        EXPECT_EQ( cpu_buffer.getDeviceContext()->getDeviceType(), DeviceType::Cpu );

        if (has_cuda_) {
            TensorBuffer<TensorDataType::FP32, Compute::CudaMemoryResource> cuda_buffer( cuda_context_, 100 );

            EXPECT_NE( cuda_buffer.getDeviceContext(), nullptr );
            EXPECT_EQ( cuda_buffer.getDeviceContext()->getDeviceType(), DeviceType::Cuda );
        }
    }

    TEST_F( TensorBufferTests, ArchitecturalSeparationOfConcerns ) {
        // Verify TensorBuffer focuses purely on memory management
        // Transfer operations should NOT be present (removed copyFrom)

        TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> buffer( cpu_context_, 100 );

        // Buffer provides memory management primitives
        EXPECT_NE( buffer.rawData(), nullptr );
        EXPECT_EQ( buffer.size(), 100 );
        EXPECT_GT( buffer.storageBytes(), 0 );
        EXPECT_GT( buffer.alignedSize(), 0 );

        // Buffer supports resize for owned memory
        EXPECT_NO_THROW( buffer.resize( 200 ) );
        EXPECT_EQ( buffer.size(), 200 );

        // Transfer operations are intentionally NOT available at this level
        // They belong at the Tensor level where device contexts are meaningful
    }
}