#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <sstream>
#include <iostream>
#include <atomic>

import Mila;

namespace Tensors::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TrackedMemoryResourceTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Reset global statistics before each test
            MemoryStats::reset();
        }

        void TearDown() override {
            // Reset statistics after each test
            MemoryStats::reset();
        }
    };

    // ============================================================================
    // Basic Memory Statistics Tests
    // ============================================================================

    TEST_F( TrackedMemoryResourceTests, MemoryStatsInitialization ) {
        // Verify initial state of memory statistics
        EXPECT_EQ( MemoryStats::totalAllocated.load(), 0 );
        EXPECT_EQ( MemoryStats::totalDeallocated.load(), 0 );
        EXPECT_EQ( MemoryStats::currentUsage.load(), 0 );
        EXPECT_EQ( MemoryStats::peakUsage.load(), 0 );
        EXPECT_EQ( MemoryStats::allocationCount.load(), 0 );
        EXPECT_EQ( MemoryStats::deallocationCount.load(), 0 );
        EXPECT_EQ( MemoryStats::fillOperationCount.load(), 0 );
    }

    TEST_F( TrackedMemoryResourceTests, MemoryStatsReset ) {
        // Modify statistics
        MemoryStats::totalAllocated = 1000;
        MemoryStats::totalDeallocated = 500;
        MemoryStats::currentUsage = 500;
        MemoryStats::peakUsage = 800;
        MemoryStats::allocationCount = 10;
        MemoryStats::deallocationCount = 5;
        MemoryStats::fillOperationCount = 3;

        // Reset should zero everything
        MemoryStats::reset();

        EXPECT_EQ( MemoryStats::totalAllocated.load(), 0 );
        EXPECT_EQ( MemoryStats::totalDeallocated.load(), 0 );
        EXPECT_EQ( MemoryStats::currentUsage.load(), 0 );
        EXPECT_EQ( MemoryStats::peakUsage.load(), 0 );
        EXPECT_EQ( MemoryStats::allocationCount.load(), 0 );
        EXPECT_EQ( MemoryStats::deallocationCount.load(), 0 );
        EXPECT_EQ( MemoryStats::fillOperationCount.load(), 0 );
    }

    TEST_F( TrackedMemoryResourceTests, MemoryStatsPrint ) {
        // Set some statistics
        MemoryStats::totalAllocated = 2048;
        MemoryStats::totalDeallocated = 1024;
        MemoryStats::currentUsage = 1024;
        MemoryStats::peakUsage = 1536;
        MemoryStats::allocationCount = 15;
        MemoryStats::deallocationCount = 8;
        MemoryStats::fillOperationCount = 5;

        // Capture output
        std::ostringstream output;
        MemoryStats::print( output );

        std::string result = output.str();

        // Verify output contains expected values
        EXPECT_NE( result.find( "Total Allocated:    2048" ), std::string::npos );
        EXPECT_NE( result.find( "Total Deallocated:  1024" ), std::string::npos );
        EXPECT_NE( result.find( "Current Usage:      1024" ), std::string::npos );
        EXPECT_NE( result.find( "Peak Usage:         1536" ), std::string::npos );
        EXPECT_NE( result.find( "Allocation Count:   15" ), std::string::npos );
        EXPECT_NE( result.find( "Deallocation Count: 8" ), std::string::npos );
        EXPECT_NE( result.find( "Fill Operations:    5" ), std::string::npos );
    }

    // ============================================================================
    // TrackedMemoryResource Construction Tests
    // ============================================================================

    TEST_F( TrackedMemoryResourceTests, TrackedMemoryResourceConstruction ) {
        auto underlying = std::make_unique<CpuMemoryResource>();
        auto underlying_ptr = underlying.get();

        TrackedMemoryResource tracked( underlying.release(), "test_resource" );

        EXPECT_EQ( tracked.name(), "test_resource" );

        // Test with empty name
        auto underlying2 = std::make_unique<CpuMemoryResource>();
        TrackedMemoryResource tracked2( underlying2.release() );

        EXPECT_EQ( tracked2.name(), "" );
    }

    // ============================================================================
    // Allocation and Deallocation Tracking Tests
    // ============================================================================

    TEST_F( TrackedMemoryResourceTests, BasicAllocationTracking ) {
        auto underlying = std::make_unique<CpuMemoryResource>();
        TrackedMemoryResource tracked( underlying.release(), "cpu_tracker" );

        // Allocate memory
        void* ptr = tracked.allocate( 1024, 64 );
        EXPECT_NE( ptr, nullptr );

        // Verify statistics
        EXPECT_EQ( MemoryStats::totalAllocated.load(), 1024 );
        EXPECT_EQ( MemoryStats::currentUsage.load(), 1024 );
        EXPECT_EQ( MemoryStats::peakUsage.load(), 1024 );
        EXPECT_EQ( MemoryStats::allocationCount.load(), 1 );
        EXPECT_EQ( MemoryStats::deallocationCount.load(), 0 );

        // Deallocate memory
        tracked.deallocate( ptr, 1024, 64 );

        // Verify deallocation statistics
        EXPECT_EQ( MemoryStats::totalAllocated.load(), 1024 );
        EXPECT_EQ( MemoryStats::totalDeallocated.load(), 1024 );
        EXPECT_EQ( MemoryStats::currentUsage.load(), 0 );
        EXPECT_EQ( MemoryStats::peakUsage.load(), 1024 );
        EXPECT_EQ( MemoryStats::allocationCount.load(), 1 );
        EXPECT_EQ( MemoryStats::deallocationCount.load(), 1 );
    }

    TEST_F( TrackedMemoryResourceTests, MultipleAllocationTracking ) {
        auto underlying = std::make_unique<CpuMemoryResource>();
        TrackedMemoryResource tracked( underlying.release(), "multi_tracker" );

        std::vector<void*> pointers;
        std::vector<size_t> sizes = { 512, 1024, 2048, 256 };

        // Allocate multiple blocks
        for ( size_t size : sizes ) {
            void* ptr = tracked.allocate( size, 32 );
            EXPECT_NE( ptr, nullptr );
            pointers.push_back( ptr );
        }

        // Verify cumulative statistics
        size_t expected_total = 512 + 1024 + 2048 + 256; // 3840
        EXPECT_EQ( MemoryStats::totalAllocated.load(), expected_total );
        EXPECT_EQ( MemoryStats::currentUsage.load(), expected_total );
        EXPECT_EQ( MemoryStats::peakUsage.load(), expected_total );
        EXPECT_EQ( MemoryStats::allocationCount.load(), 4 );

        // Deallocate in reverse order
        for ( int i = static_cast<int>(pointers.size()) - 1; i >= 0; --i ) {
            tracked.deallocate( pointers[ i ], sizes[ i ], 32 );
        }

        // Verify final statistics
        EXPECT_EQ( MemoryStats::totalAllocated.load(), expected_total );
        EXPECT_EQ( MemoryStats::totalDeallocated.load(), expected_total );
        EXPECT_EQ( MemoryStats::currentUsage.load(), 0 );
        EXPECT_EQ( MemoryStats::peakUsage.load(), expected_total );
        EXPECT_EQ( MemoryStats::deallocationCount.load(), 4 );
    }

    TEST_F( TrackedMemoryResourceTests, PeakUsageTracking ) {
        auto underlying = std::make_unique<CpuMemoryResource>();
        TrackedMemoryResource tracked( underlying.release(), "peak_tracker" );

        // Allocate and track peak usage
        void* ptr1 = tracked.allocate( 1000, 32 );
        EXPECT_EQ( MemoryStats::peakUsage.load(), 1000 );

        void* ptr2 = tracked.allocate( 500, 32 );
        EXPECT_EQ( MemoryStats::peakUsage.load(), 1500 );

        void* ptr3 = tracked.allocate( 2000, 32 );
        EXPECT_EQ( MemoryStats::peakUsage.load(), 3500 );

        // Deallocate middle allocation
        tracked.deallocate( ptr2, 500, 32 );
        EXPECT_EQ( MemoryStats::currentUsage.load(), 3000 );
        EXPECT_EQ( MemoryStats::peakUsage.load(), 3500 ); // Peak should remain

        // Allocate smaller amount - peak shouldn't change
        void* ptr4 = tracked.allocate( 200, 32 );
        EXPECT_EQ( MemoryStats::currentUsage.load(), 3200 );
        EXPECT_EQ( MemoryStats::peakUsage.load(), 3500 ); // Peak unchanged

        // Cleanup
        tracked.deallocate( ptr1, 1000, 32 );
        tracked.deallocate( ptr3, 2000, 32 );
        tracked.deallocate( ptr4, 200, 32 );
    }

    // ============================================================================
    // Fill Operation Tracking Tests
    // ============================================================================

    TEST_F( TrackedMemoryResourceTests, FillOperationTracking ) {
        auto underlying = std::make_unique<CpuMemoryResource>();
        TrackedMemoryResource tracked( underlying.release(), "fill_tracker" );

        void* ptr = tracked.allocate( 1024, 32 );

        // Test original fill method
        int32_t value = 42;
        tracked.fill( ptr, 256, &value, sizeof( value ) );
        EXPECT_EQ( MemoryStats::fillOperationCount.load(), 1 );

        // Test int32_t array fill
        std::vector<int32_t> int_values = { 1, 2, 3, 4, 5 };
        tracked.fill( ptr, 5, int_values.data(), TensorDataType::INT32 );
        EXPECT_EQ( MemoryStats::fillOperationCount.load(), 2 );

        // Test float array fill
        std::vector<float> float_values = { 1.1f, 2.2f, 3.3f };
        tracked.fill( ptr, 3, float_values.data(), TensorDataType::FP32 );
        EXPECT_EQ( MemoryStats::fillOperationCount.load(), 3 );

        // Test int32_t constant fill
        tracked.fill( ptr, 100, 123, TensorDataType::INT32 );
        EXPECT_EQ( MemoryStats::fillOperationCount.load(), 4 );

        // Test float constant fill
        tracked.fill( ptr, 50, 3.14159f, TensorDataType::FP32 );
        EXPECT_EQ( MemoryStats::fillOperationCount.load(), 5 );

        tracked.deallocate( ptr, 1024, 32 );
    }

    TEST_F( TrackedMemoryResourceTests, MemcpyOperationTracking ) {
        auto underlying = std::make_unique<CpuMemoryResource>();
        TrackedMemoryResource tracked( underlying.release(), "memcpy_tracker" );

        void* src = tracked.allocate( 512, 32 );
        void* dst = tracked.allocate( 512, 32 );

        // Test memcpy - should not affect fill operation count
        tracked.memcpy( dst, src, 256 );
        EXPECT_EQ( MemoryStats::fillOperationCount.load(), 0 );

        // Verify allocation tracking still works
        EXPECT_EQ( MemoryStats::allocationCount.load(), 2 );

        tracked.deallocate( src, 512, 32 );
        tracked.deallocate( dst, 512, 32 );
    }

    // ============================================================================
    // Memory Resource Equality Tests
    // ============================================================================

    TEST_F( TrackedMemoryResourceTests, MemoryResourceEquality ) {
        auto underlying1 = std::make_unique<CpuMemoryResource>();
        auto underlying2 = std::make_unique<CpuMemoryResource>();
        auto underlying1_ptr = underlying1.get();
        auto underlying2_ptr = underlying2.get();

        TrackedMemoryResource tracked1( underlying1.release(), "tracker1" );
        TrackedMemoryResource tracked2( underlying2.release(), "tracker2" );

        // TrackedMemoryResource equality is based on underlying resource equality
        // Since different CPU memory resources are not equal, these should not be equal
        EXPECT_FALSE( tracked1.is_equal( tracked2 ) );

        // Test with same underlying resource (theoretical case)
        auto shared_underlying = std::make_shared<CpuMemoryResource>();
        TrackedMemoryResource tracked3( shared_underlying.get(), "tracker3" );
        TrackedMemoryResource tracked4( shared_underlying.get(), "tracker4" );

        EXPECT_TRUE( tracked3.is_equal( tracked4 ) );

        // Test comparison with non-TrackedMemoryResource
        CpuMemoryResource direct_resource;
        EXPECT_FALSE( tracked1.is_equal( direct_resource ) );
    }

    // ============================================================================
    // Thread Safety Tests
    // ============================================================================

    TEST_F( TrackedMemoryResourceTests, ThreadSafetyBasic ) {
        auto underlying = std::make_unique<CpuMemoryResource>();
        TrackedMemoryResource tracked( underlying.release(), "thread_tracker" );

        constexpr int num_threads = 4;
        constexpr int allocations_per_thread = 100;
        constexpr size_t allocation_size = 256;

        std::vector<std::thread> threads;
        std::vector<std::vector<void*>> thread_pointers( num_threads );

        // Launch threads that allocate memory
        for ( int t = 0; t < num_threads; ++t ) {
            threads.emplace_back( [&tracked, &thread_pointers, t, allocations_per_thread, allocation_size]() {
                for ( int i = 0; i < allocations_per_thread; ++i ) {
                    void* ptr = tracked.allocate( allocation_size, 32 );
                    thread_pointers[ t ].push_back( ptr );
                }
                } );
        }

        // Wait for all threads to complete
        for ( auto& thread : threads ) {
            thread.join();
        }

        // Verify statistics
        size_t expected_total = num_threads * allocations_per_thread * allocation_size;
        EXPECT_EQ( MemoryStats::totalAllocated.load(), expected_total );
        EXPECT_EQ( MemoryStats::currentUsage.load(), expected_total );
        EXPECT_EQ( MemoryStats::allocationCount.load(), num_threads * allocations_per_thread );

        // Cleanup
        threads.clear();
        for ( int t = 0; t < num_threads; ++t ) {
            threads.emplace_back( [&tracked, &thread_pointers, t, allocation_size]() {
                for ( void* ptr : thread_pointers[ t ] ) {
                    tracked.deallocate( ptr, allocation_size, 32 );
                }
                } );
        }

        for ( auto& thread : threads ) {
            thread.join();
        }

        // Verify final cleanup
        EXPECT_EQ( MemoryStats::currentUsage.load(), 0 );
        EXPECT_EQ( MemoryStats::deallocationCount.load(), num_threads * allocations_per_thread );
    }

    TEST_F( TrackedMemoryResourceTests, ConcurrentPeakUsageTracking ) {
        auto underlying = std::make_unique<CpuMemoryResource>();
        TrackedMemoryResource tracked( underlying.release(), "concurrent_peak_tracker" );

        constexpr int num_threads = 8;
        constexpr size_t large_allocation = 1024 * 1024; // 1MB per thread

        std::vector<std::thread> threads;
        std::atomic<int> ready_count{ 0 };
        std::atomic<bool> start_flag{ false };

        // Launch threads that allocate simultaneously to test peak tracking
        for ( int t = 0; t < num_threads; ++t ) {
            threads.emplace_back( [&tracked, &ready_count, &start_flag, large_allocation]() {
                ready_count++;
                while ( !start_flag.load() ) {
                    std::this_thread::yield();
                }

                void* ptr = tracked.allocate( large_allocation, 64 );
                std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
                tracked.deallocate( ptr, large_allocation, 64 );
                } );
        }

        // Wait for all threads to be ready, then start simultaneously
        while ( ready_count.load() < num_threads ) {
            std::this_thread::yield();
        }
        start_flag = true;

        for ( auto& thread : threads ) {
            thread.join();
        }

        // Peak usage should reflect the maximum concurrent allocation
        size_t expected_peak = num_threads * large_allocation;
        EXPECT_EQ( MemoryStats::peakUsage.load(), expected_peak );
        EXPECT_EQ( MemoryStats::currentUsage.load(), 0 );
    }

    // ============================================================================
    // Integration Tests with TensorBuffer
    // ============================================================================

    TEST_F( TrackedMemoryResourceTests, TensorBufferIntegration ) {
        // Create tensor buffer with tracked memory resource
        TensorBuffer<TensorDataType::FP32, CpuMemoryResource, true> tracked_buffer( 1000 );

        // Verify that memory tracking occurred during buffer creation
        EXPECT_GT( MemoryStats::totalAllocated.load(), 0 );
        EXPECT_GT( MemoryStats::allocationCount.load(), 0 );
        EXPECT_EQ( MemoryStats::currentUsage.load(), MemoryStats::totalAllocated.load() );

        auto initial_allocated = MemoryStats::totalAllocated.load();
        auto initial_count = MemoryStats::allocationCount.load();

        // Test fill operations through tracked buffer
        std::vector<float> test_values( 100, 3.14159f );
        tracked_buffer.fill( test_values.data(), 100 );

        // Verify fill operation was tracked
        EXPECT_GT( MemoryStats::fillOperationCount.load(), 0 );

        // Test resize operation
        tracked_buffer.resize( 2000 );

        // Verify additional allocation was tracked
        EXPECT_GT( MemoryStats::totalAllocated.load(), initial_allocated );
        EXPECT_GT( MemoryStats::allocationCount.load(), initial_count );
    }

    // ============================================================================
    // Edge Cases and Error Conditions
    // ============================================================================

    TEST_F( TrackedMemoryResourceTests, ZeroSizeAllocation ) {
        auto underlying = std::make_unique<CpuMemoryResource>();
        TrackedMemoryResource tracked( underlying.release(), "zero_tracker" );

        // Allocate zero bytes
        void* ptr = tracked.allocate( 0, 32 );

        // allocationCount should ALWAYS be incremented for any allocate() call
        EXPECT_EQ( MemoryStats::allocationCount.load(), 1 );

        // totalAllocated should be 0 for zero-byte allocation
        EXPECT_EQ( MemoryStats::totalAllocated.load(), 0 );

        // Deallocate if pointer was returned
        if ( ptr != nullptr ) {
            tracked.deallocate( ptr, 0, 32 );
            EXPECT_EQ( MemoryStats::deallocationCount.load(), 1 );
        }
    }

    TEST_F( TrackedMemoryResourceTests, FillOperationEdgeCases ) {
        auto underlying = std::make_unique<CpuMemoryResource>();
        TrackedMemoryResource tracked( underlying.release(), "edge_tracker" );

        void* ptr = tracked.allocate( 100, 32 );

        // Test that fill operations are counted even with zero count
        float dummy_val = 0.0f;
        tracked.fill( ptr, 0, dummy_val, TensorDataType::FP32 );
        EXPECT_EQ( MemoryStats::fillOperationCount.load(), 1 );

        // Test fill with various data types
        float float_val = 1.0f;
        tracked.fill( ptr, 10, float_val, TensorDataType::FP32 );
        EXPECT_EQ( MemoryStats::fillOperationCount.load(), 2 );

        int32_t int_val = 42;
        tracked.fill( ptr, 10, int_val, TensorDataType::INT32 );
        EXPECT_EQ( MemoryStats::fillOperationCount.load(), 3 );

        tracked.deallocate( ptr, 100, 32 );
    }

    // ============================================================================
    // Performance Impact Tests
    // ============================================================================

    TEST_F( TrackedMemoryResourceTests, PerformanceOverhead ) {
        constexpr int num_operations = 10000;
        constexpr size_t allocation_size = 1024;

        // Test untracked performance
        CpuMemoryResource direct_resource;
        auto start_direct = std::chrono::high_resolution_clock::now();

        std::vector<void*> direct_pointers;
        for ( int i = 0; i < num_operations; ++i ) {
            void* ptr = direct_resource.allocate( allocation_size, 32 );
            direct_pointers.push_back( ptr );
        }
        for ( int i = 0; i < num_operations; ++i ) {
            direct_resource.deallocate( direct_pointers[ i ], allocation_size, 32 );
        }

        auto end_direct = std::chrono::high_resolution_clock::now();
        auto direct_duration = std::chrono::duration_cast<std::chrono::microseconds>( end_direct - start_direct );

        // Test tracked performance
        auto underlying = std::make_unique<CpuMemoryResource>();
        TrackedMemoryResource tracked( underlying.release(), "perf_tracker" );

        auto start_tracked = std::chrono::high_resolution_clock::now();

        std::vector<void*> tracked_pointers;
        for ( int i = 0; i < num_operations; ++i ) {
            void* ptr = tracked.allocate( allocation_size, 32 );
            tracked_pointers.push_back( ptr );
        }
        for ( int i = 0; i < num_operations; ++i ) {
            tracked.deallocate( tracked_pointers[ i ], allocation_size, 32 );
        }

        auto end_tracked = std::chrono::high_resolution_clock::now();
        auto tracked_duration = std::chrono::duration_cast<std::chrono::microseconds>( end_tracked - start_tracked );

        // Verify tracking overhead is reasonable (should be less than 50% overhead)
        double overhead_ratio = static_cast<double>( tracked_duration.count() ) / direct_duration.count();
        EXPECT_LT( overhead_ratio, 1.5 ); // Less than 50% overhead

        // Verify statistics accuracy
        EXPECT_EQ( MemoryStats::allocationCount.load(), num_operations );
        EXPECT_EQ( MemoryStats::deallocationCount.load(), num_operations );
        EXPECT_EQ( MemoryStats::currentUsage.load(), 0 );
    }
}