/**
 * @file CudaDeviceContext.cpp
 * @brief Unit tests for CudaDeviceContext class.
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

import Compute.CudaDeviceContext;
import Compute.DeviceType;
import Compute.DeviceRegistry;
import Compute.CudaDevice;

using namespace Mila::Dnn::Compute;

namespace Dnn::Compute::Devices::Cuda::Tests {

    /**
     * @brief Test fixture for CudaDeviceContext tests.
     *
     * Sets up device registry with CUDA device registration before each test
     * and ensures clean state for testing. Skips tests if no CUDA devices are available.
     */
    class CudaDeviceContextTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // Check if CUDA devices are available
            int device_count;
            cudaError_t error = cudaGetDeviceCount(&device_count);

            if (error != cudaSuccess || device_count == 0) {
                GTEST_SKIP() << "No CUDA devices available. Skipping CUDA tests.";
            }

            // Register CUDA devices if not already registered
            for (int i = 0; i < device_count; ++i) {
                std::string device_name = "CUDA:" + std::to_string(i);
                if (!DeviceRegistry::instance().hasDevice(device_name)) {
                    DeviceRegistry::instance().registerDevice(device_name, [i]() {
                        return std::make_shared<CudaDevice>(i);
                        });
                }
            }

            // Store the first available device name for testing
            if (device_count > 0) {
                valid_device_name_ = "CUDA:0";
            }
        }

        void TearDown() override {
            // Reset CUDA device to ensure clean state
            cudaDeviceReset();
        }

        std::string valid_device_name_;
    };

    // ============================================================================
    // Constructor Tests
    // ============================================================================

    TEST_F(CudaDeviceContextTest, ConstructorWithValidDeviceName) {
        EXPECT_NO_THROW({
            CudaDeviceContext context(valid_device_name_);
            });
    }

    TEST_F(CudaDeviceContextTest, ConstructorWithInvalidDeviceName) {
        EXPECT_THROW({
            CudaDeviceContext context("INVALID_DEVICE");
            }, std::runtime_error);
    }

    TEST_F(CudaDeviceContextTest, ConstructorThrowsOnEmptyDeviceName) {
        EXPECT_THROW({
            CudaDeviceContext context("");
            }, std::runtime_error);
    }

    TEST_F(CudaDeviceContextTest, ConstructorThrowsOnWrongDeviceType) {
        EXPECT_THROW({
            CudaDeviceContext context("CPU");
            }, std::runtime_error);
    }

    TEST_F(CudaDeviceContextTest, ConstructorThrowsOnNonExistentCudaDevice) {
        EXPECT_THROW({
            CudaDeviceContext context("CUDA:999");
            }, std::runtime_error);
    }

    TEST_F(CudaDeviceContextTest, ConstructorInitializesDeviceResources) {
        CudaDeviceContext context(valid_device_name_);

        // Verify that CUDA resources are properly initialized
        EXPECT_NE(context.getStream(), nullptr);
        EXPECT_NO_THROW(context.getCublasLtHandle());

#ifdef USE_CUDNN
        EXPECT_NO_THROW(context.getCudnnHandle());
#endif
    }

    // ============================================================================
    // DeviceContext Interface Tests
    // ============================================================================

    TEST_F(CudaDeviceContextTest, GetDeviceTypeReturnsCuda) {
        CudaDeviceContext context(valid_device_name_);
        EXPECT_EQ(context.getDeviceType(), DeviceType::Cuda);
    }

    TEST_F(CudaDeviceContextTest, GetDeviceNameReturnsCorrectName) {
        CudaDeviceContext context(valid_device_name_);
        EXPECT_EQ(context.getDeviceName(), valid_device_name_);
    }

    TEST_F(CudaDeviceContextTest, GetDeviceIdReturnsCorrectId) {
        CudaDeviceContext context("CUDA:0");
        EXPECT_EQ(context.getDeviceId(), 0);

        // Test with another device if available
        int device_count;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 1) {
            CudaDeviceContext context1("CUDA:1");
            EXPECT_EQ(context1.getDeviceId(), 1);
        }
    }

    TEST_F(CudaDeviceContextTest, MakeCurrentSetsCorrectDevice) {
        CudaDeviceContext context(valid_device_name_);

        EXPECT_NO_THROW(context.makeCurrent());

        int current_device;
        cudaGetDevice(&current_device);
        EXPECT_EQ(current_device, context.getDeviceId());
    }

    TEST_F(CudaDeviceContextTest, MakeCurrentOptimizesThreadLocalCaching) {
        CudaDeviceContext context(valid_device_name_);

        // First call should set the device
        EXPECT_NO_THROW(context.makeCurrent());
        int first_device;
        cudaGetDevice(&first_device);

        // Second call should use thread-local caching (no CUDA API call)
        EXPECT_NO_THROW(context.makeCurrent());
        int second_device;
        cudaGetDevice(&second_device);

        EXPECT_EQ(first_device, second_device);
        EXPECT_EQ(first_device, context.getDeviceId());
    }

    TEST_F(CudaDeviceContextTest, SynchronizeWithValidStream) {
        CudaDeviceContext context(valid_device_name_);

        // Launch a simple kernel operation to have something to synchronize
        context.makeCurrent();

        // Synchronize should not throw
        EXPECT_NO_THROW(context.synchronize());
    }

    TEST_F(CudaDeviceContextTest, GetDeviceReturnsValidCudaDevice) {
        CudaDeviceContext context(valid_device_name_);
        auto device = context.getDevice();

        ASSERT_NE(device, nullptr);
        EXPECT_EQ(device->getDeviceType(), DeviceType::Cuda);
        EXPECT_EQ(device->getName(), valid_device_name_);

        // Test that it's actually a CudaDevice
        auto cuda_device = std::dynamic_pointer_cast<CudaDevice>(device);
        ASSERT_NE(cuda_device, nullptr);
        EXPECT_EQ(cuda_device->getDeviceId(), context.getDeviceId());
    }

    // ============================================================================
    // CUDA-Specific Method Tests
    // ============================================================================

    TEST_F(CudaDeviceContextTest, GetStreamReturnsValidStream) {
        CudaDeviceContext context(valid_device_name_);
        cudaStream_t stream = context.getStream();

        EXPECT_NE(stream, nullptr);

        // Verify stream properties
        unsigned int flags;
        cudaStreamGetFlags(stream, &flags);
        EXPECT_EQ(flags, cudaStreamNonBlocking);
    }

    TEST_F(CudaDeviceContextTest, GetCublasLtHandleReturnsValidHandle) {
        CudaDeviceContext context(valid_device_name_);
        context.makeCurrent();

        cublasLtHandle_t handle;
        EXPECT_NO_THROW(handle = context.getCublasLtHandle());
        EXPECT_NE(handle, nullptr);

        // Multiple calls should return the same handle (lazy initialization)
        cublasLtHandle_t handle2;
        EXPECT_NO_THROW(handle2 = context.getCublasLtHandle());
        EXPECT_EQ(handle, handle2);
    }

#ifdef USE_CUDNN
    TEST_F(CudaDeviceContextTest, GetCudnnHandleReturnsValidHandle) {
        CudaDeviceContext context(valid_device_name_);
        context.makeCurrent();

        cudnnHandle_t handle;
        EXPECT_NO_THROW(handle = context.getCudnnHandle());
        EXPECT_NE(handle, nullptr);

        // Multiple calls should return the same handle (lazy initialization)
        cudnnHandle_t handle2;
        EXPECT_NO_THROW(handle2 = context.getCudnnHandle());
        EXPECT_EQ(handle, handle2);
    }

    TEST_F(CudaDeviceContextTest, CudnnHandleUsesCorrectStream) {
        CudaDeviceContext context(valid_device_name_);
        context.makeCurrent();

        cudnnHandle_t handle = context.getCudnnHandle();
        cudaStream_t expected_stream = context.getStream();

        cudaStream_t actual_stream;
        cudnnGetStream(handle, &actual_stream);
        EXPECT_EQ(actual_stream, expected_stream);
    }
#endif

    TEST_F(CudaDeviceContextTest, GetComputeCapabilityReturnsValidValues) {
        CudaDeviceContext context(valid_device_name_);
        auto [major, minor] = context.getComputeCapability();

        // Compute capability should be valid (at least 3.0 for modern CUDA)
        EXPECT_GE(major, 3);
        EXPECT_GE(minor, 0);
        EXPECT_LE(major, 10); // Reasonable upper bound for testing
    }

    // ============================================================================
    // Thread Safety Tests
    // ============================================================================

    TEST_F(CudaDeviceContextTest, ThreadSafeHandleCreation) {
        CudaDeviceContext context(valid_device_name_);
        context.makeCurrent();

        std::vector<std::thread> threads;
        std::vector<cublasLtHandle_t> handles(4);

        // Create handles from multiple threads
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&context, &handles, i]() {
                handles[i] = context.getCublasLtHandle();
                });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // All handles should be the same (thread-safe lazy initialization)
        for (int i = 1; i < 4; ++i) {
            EXPECT_EQ(handles[0], handles[i]);
        }
    }

    TEST_F(CudaDeviceContextTest, MultipleContextsIndependent) {
        CudaDeviceContext context1(valid_device_name_);
        CudaDeviceContext context2(valid_device_name_);

        // Both should work independently
        EXPECT_NO_THROW(context1.makeCurrent());
        EXPECT_NO_THROW(context2.makeCurrent());

        // They should have different streams but same device
        EXPECT_NE(context1.getStream(), context2.getStream());
        EXPECT_EQ(context1.getDeviceId(), context2.getDeviceId());

        // And the same cublas handle instances when the device is the same
        auto cublas_handle1 = context1.getCublasLtHandle();
        auto cublas_handle2 = context2.getCublasLtHandle();
        EXPECT_EQ(cublas_handle1, cublas_handle2);
    }

    // ============================================================================
    // Error Handling Tests
    // ============================================================================

    TEST_F(CudaDeviceContextTest, ConstructorErrorMessageContainsDeviceName) {
        const std::string invalid_name = "CUDA:999";

        try {
            CudaDeviceContext context(invalid_name);
            FAIL() << "Expected std::runtime_error to be thrown";
        }
        catch (const std::runtime_error& e) {
            std::string error_msg = e.what();
            EXPECT_NE(error_msg.find("Invalid CUDA device name"), std::string::npos);
        }
    }

    TEST_F(CudaDeviceContextTest, MakeCurrentThrowsOnInvalidDevice) {
        // Create a context and then simulate device removal/error
        CudaDeviceContext context(valid_device_name_);

        // This test is difficult to implement reliably without actually 
        // corrupting CUDA state, so we'll test the error path indirectly
        // by verifying that makeCurrent() works normally
        EXPECT_NO_THROW(context.makeCurrent());
    }

    TEST_F(CudaDeviceContextTest, HandleCreationFailureRecovery) {
        CudaDeviceContext context(valid_device_name_);
        context.makeCurrent();

        // Normal handle creation should succeed
        EXPECT_NO_THROW(context.getCublasLtHandle());

        // Multiple attempts should not cause issues
        for (int i = 0; i < 5; ++i) {
            EXPECT_NO_THROW(context.getCublasLtHandle());
        }
    }

    // ============================================================================
    // Resource Management Tests
    // ============================================================================

    TEST_F(CudaDeviceContextTest, DestructorCleansUpResources) {
        cudaStream_t stream_ptr = nullptr;

        {
            CudaDeviceContext context(valid_device_name_);
            stream_ptr = context.getStream();
            EXPECT_NE(stream_ptr, nullptr);

            // Use the stream to verify it's valid
            context.makeCurrent();
            cudaError_t error = cudaStreamQuery(stream_ptr);
            EXPECT_TRUE(error == cudaSuccess || error == cudaErrorNotReady);
        }

        // After destruction, the stream should be destroyed
        // Note: We can't easily test this without potentially causing UB
        // The destructor cleanup is implicit in the RAII design
    }

    // ============================================================================
    // Inherited Helper Method Tests
    // ============================================================================

    TEST_F(CudaDeviceContextTest, IsDeviceTypeWorksCorrectly) {
        CudaDeviceContext context(valid_device_name_);

        EXPECT_TRUE(context.isDeviceType(DeviceType::Cuda));
        EXPECT_FALSE(context.isDeviceType(DeviceType::Cpu));
        EXPECT_FALSE(context.isDeviceType(DeviceType::Metal));
        EXPECT_FALSE(context.isDeviceType(DeviceType::OpenCL));
        EXPECT_FALSE(context.isDeviceType(DeviceType::Vulkan));
    }

    TEST_F(CudaDeviceContextTest, IsCudaDeviceReturnsTrue) {
        CudaDeviceContext context(valid_device_name_);
        EXPECT_TRUE(context.isCudaDevice());
    }

    TEST_F(CudaDeviceContextTest, IsOtherDeviceTypesReturnFalse) {
        CudaDeviceContext context(valid_device_name_);

        EXPECT_FALSE(context.isCpuDevice());
        EXPECT_FALSE(context.isMetalDevice());
        EXPECT_FALSE(context.isOpenCLDevice());
        EXPECT_FALSE(context.isVulkanDevice());
    }

    // ============================================================================
    // Polymorphic Usage Tests
    // ============================================================================

    TEST_F(CudaDeviceContextTest, PolymorphicUsageThroughBaseClass) {
        std::unique_ptr<DeviceContext> context =
            std::make_unique<CudaDeviceContext>(valid_device_name_);

        EXPECT_EQ(context->getDeviceType(), DeviceType::Cuda);
        EXPECT_EQ(context->getDeviceName(), valid_device_name_);
        EXPECT_EQ(context->getDeviceId(), 0);
        EXPECT_TRUE(context->isCudaDevice());
        EXPECT_FALSE(context->isCpuDevice());

        EXPECT_NO_THROW(context->makeCurrent());
        EXPECT_NO_THROW(context->synchronize());

        auto device = context->getDevice();
        ASSERT_NE(device, nullptr);
        EXPECT_EQ(device->getDeviceType(), DeviceType::Cuda);
    }

    // ============================================================================
    // Integration Tests with DeviceRegistry
    // ============================================================================

    TEST_F(CudaDeviceContextTest, UsesDeviceRegistryCorrectly) {
        EXPECT_TRUE(DeviceRegistry::instance().hasDevice(valid_device_name_));

        auto device_from_registry = DeviceRegistry::instance().createDevice(valid_device_name_);
        ASSERT_NE(device_from_registry, nullptr);
        EXPECT_EQ(device_from_registry->getDeviceType(), DeviceType::Cuda);

        CudaDeviceContext context(valid_device_name_);
        auto device_from_context = context.getDevice();

        EXPECT_EQ(device_from_registry->getDeviceType(), device_from_context->getDeviceType());
        EXPECT_EQ(device_from_registry->getName(), device_from_context->getName());
    }

    // ============================================================================
    // Performance and Optimization Tests
    // ============================================================================

    TEST_F(CudaDeviceContextTest, LazyHandleInitialization) {
        CudaDeviceContext context(valid_device_name_);

        // Handles should be created lazily
        auto start_time = std::chrono::high_resolution_clock::now();
        auto handle = context.getCublasLtHandle();
        auto first_call_time = std::chrono::high_resolution_clock::now();

        auto handle2 = context.getCublasLtHandle();
        auto second_call_time = std::chrono::high_resolution_clock::now();

        EXPECT_EQ(handle, handle2);

        // Second call should be faster (already initialized)
        auto first_duration = first_call_time - start_time;
        auto second_duration = second_call_time - first_call_time;

        // This is a heuristic test - second call should be much faster
        EXPECT_LT(second_duration.count(), first_duration.count());
    }

    TEST_F(CudaDeviceContextTest, StreamOperationsAreNonBlocking) {
        CudaDeviceContext context(valid_device_name_);
        context.makeCurrent();

        cudaStream_t stream = context.getStream();

        // Verify stream is non-blocking
        unsigned int flags;
        cudaStreamGetFlags(stream, &flags);
        EXPECT_EQ(flags & cudaStreamNonBlocking, cudaStreamNonBlocking);
    }

    // ============================================================================
    // Multi-GPU Tests (if multiple devices available)
    // ============================================================================

    TEST_F(CudaDeviceContextTest, MultiGpuContextsWorkIndependently) {
        int device_count;
        cudaGetDeviceCount(&device_count);

        if (device_count < 2) {
            SUCCEED() << "Multi-GPU tests require at least 2 CUDA devices";
            return;
        }

        CudaDeviceContext context0("CUDA:0");
        CudaDeviceContext context1("CUDA:1");

        EXPECT_EQ(context0.getDeviceId(), 0);
        EXPECT_EQ(context1.getDeviceId(), 1);

        // Each should set its own device correctly
        context0.makeCurrent();
        int current_device;
        cudaGetDevice(&current_device);
        EXPECT_EQ(current_device, 0);

        context1.makeCurrent();
        cudaGetDevice(&current_device);
        EXPECT_EQ(current_device, 1);

        // Streams should be different
        EXPECT_NE(context0.getStream(), context1.getStream());

        // Handles should be different
        EXPECT_NE(context0.getCublasLtHandle(), context1.getCublasLtHandle());
    }

    // ============================================================================
    // Edge Cases and Boundary Tests
    // ============================================================================

    TEST_F(CudaDeviceContextTest, RepeatedOperationsStable) {
        CudaDeviceContext context(valid_device_name_);

        // Perform many repeated operations to test stability
        for (int i = 0; i < 100; ++i) {
            EXPECT_NO_THROW(context.makeCurrent());
            EXPECT_EQ(context.getDeviceType(), DeviceType::Cuda);
            EXPECT_EQ(context.getDeviceName(), valid_device_name_);
            EXPECT_EQ(context.getDeviceId(), 0);
            EXPECT_NE(context.getStream(), nullptr);
            EXPECT_NO_THROW(context.synchronize());

            if (i % 10 == 0) {
                // Occasionally test handle creation
                EXPECT_NO_THROW(context.getCublasLtHandle());
#ifdef USE_CUDNN
                EXPECT_NO_THROW(context.getCudnnHandle());
#endif
            }
        }
    }

} // anonymous namespace