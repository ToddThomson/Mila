#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorConstructionTest : public testing::Test {
    protected:
        TensorConstructionTest() {}

        void SetUp() override {
            // Initialize device registrar to register all available devices
            //DeviceRegistrar::instance();

            // Create CPU device context
            cpu_context_ = std::make_shared<CpuDeviceContext>();

            // Check if CUDA devices are available before creating CUDA context
            int device_count;
            cudaError_t error = cudaGetDeviceCount( &device_count );

            has_cuda_ = (error == cudaSuccess && device_count > 0);

            if (has_cuda_) {
                cuda_context_ = std::make_shared<CudaDeviceContext>( "CUDA:0" );
            }
        }

        std::shared_ptr<DeviceContext> cpu_context_;
        std::shared_ptr<DeviceContext> cuda_context_;
        bool has_cuda_ = false;
    };

    // ====================================================================
    // Device Context Constructor Tests
    // ====================================================================

    TEST_F(TensorConstructionTest, ConstructorWithDeviceContext) {
        std::vector<size_t> shape = { 2, 3 };

        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_tensor(cpu_context_, shape);

        EXPECT_FALSE(cpu_tensor.empty());
        EXPECT_EQ(cpu_tensor.size(), 6);
        EXPECT_EQ(cpu_tensor.rank(), 2);
        EXPECT_EQ(cpu_tensor.shape(), shape);
        EXPECT_EQ(cpu_tensor.getDeviceContext(), cpu_context_);

        if (has_cuda_) {
            Tensor<TensorDataType::FP32, CudaMemoryResource> cuda_tensor(cuda_context_, shape);

            EXPECT_FALSE(cuda_tensor.empty());
            EXPECT_EQ(cuda_tensor.size(), 6);
            EXPECT_EQ(cuda_tensor.rank(), 2);
            EXPECT_EQ(cuda_tensor.shape(), shape);
            EXPECT_EQ(cuda_tensor.getDeviceContext(), cuda_context_);
        }
        else {
            GTEST_SKIP() << "CUDA device not available for CUDA tensor test";
        }
    }

    TEST_F(TensorConstructionTest, ConstructorWithDeviceName) {
        std::vector<size_t> shape = { 2, 3 };

        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_tensor("CPU", shape);

        EXPECT_FALSE(cpu_tensor.empty());
        EXPECT_EQ(cpu_tensor.size(), 6);
        EXPECT_EQ(cpu_tensor.rank(), 2);
        EXPECT_EQ(cpu_tensor.shape(), shape);

        if (has_cuda_) {
            Tensor<TensorDataType::FP32, CudaMemoryResource> cuda_tensor("CUDA:0", shape);

            EXPECT_FALSE(cuda_tensor.empty());
            EXPECT_EQ(cuda_tensor.size(), 6);
            EXPECT_EQ(cuda_tensor.rank(), 2);
            EXPECT_EQ(cuda_tensor.shape(), shape);
        }
        else {
            GTEST_SKIP() << "CUDA device not available for CUDA tensor test";
        }
    }

    TEST_F(TensorConstructionTest, ConstructorWithNullDeviceContext) {
        std::vector<size_t> shape = { 2, 3 };
        std::shared_ptr<DeviceContext> null_context;

        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CpuMemoryResource>(null_context, shape)),
            std::invalid_argument
        );
    }

    TEST_F(TensorConstructionTest, ConstructorWithInvalidDeviceName) {
        std::vector<size_t> shape = { 2, 3 };

        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CpuMemoryResource>("INVALID_DEVICE", shape)),
            std::runtime_error
        );
    }

    // ====================================================================
    // Shape Constructor Tests
    // ====================================================================

    TEST_F(TensorConstructionTest, ConstructorWithShape) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, CudaMemoryResource> tensor(cuda_context_, shape);

        EXPECT_FALSE(tensor.empty());
        EXPECT_EQ(tensor.size(), 6);
        EXPECT_EQ(tensor.rank(), 2);
        EXPECT_EQ(tensor.shape(), shape);
    }

    TEST_F(TensorConstructionTest, ConstructorWithEmptyShape) {
        std::vector<size_t> shape = {};
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor(cpu_context_, shape);

        EXPECT_TRUE(tensor.empty());
        EXPECT_EQ(tensor.size(), 0);
        EXPECT_EQ(tensor.rank(), 0);
        EXPECT_EQ(tensor.strides().size(), 0);
        EXPECT_EQ(tensor.shape(), shape);
    }

    // ====================================================================
    // External Data Constructor Tests
    // ====================================================================

    TEST_F(TensorConstructionTest, ConstructWithExternalData) {
        std::vector<size_t> shape = { 2, 3 };
        auto data_ptr = std::make_shared<float[]>(6);
        
        for (int i = 0; i < 6; i++) 
            data_ptr[i] = static_cast<float>(i);

        std::shared_ptr<void> data_ptr_void(data_ptr, data_ptr.get());
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor(cpu_context_, shape, data_ptr_void);

        EXPECT_EQ(tensor.shape(), shape);
        EXPECT_EQ(tensor.size(), 6);
        EXPECT_FALSE(tensor.empty());
        EXPECT_EQ(tensor.getDeviceContext(), cpu_context_);
    }

    TEST_F(TensorConstructionTest, ConstructWithNullExternalData) {
        std::vector<size_t> shape = { 2, 3 };
        std::shared_ptr<void> null_ptr;

        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CpuMemoryResource>(cpu_context_, shape, null_ptr)),
            std::invalid_argument
        );
    }

    TEST_F(TensorConstructionTest, ConstructWithExternalDataNullContext) {
        std::vector<size_t> shape = { 2, 3 };
        auto data_ptr = std::make_shared<float[]>(6);
        std::shared_ptr<void> data_ptr_void(data_ptr, data_ptr.get());
        std::shared_ptr<DeviceContext> null_context;

        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CpuMemoryResource>(null_context, shape, data_ptr_void)),
            std::invalid_argument
        );
    }

    // ====================================================================
    // Move Constructor Tests
    // ====================================================================

    TEST_F(TensorConstructionTest, MoveConstructor) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> original(cpu_context_, shape);
        std::string original_uid = original.getUId();

        Tensor<TensorDataType::FP32, CpuMemoryResource> moved(std::move(original));

        EXPECT_EQ(moved.shape(), shape);
        EXPECT_EQ(moved.size(), 6);
        EXPECT_EQ(moved.getUId(), original_uid);
        EXPECT_EQ(moved.getDeviceContext(), cpu_context_);

        // The original should be in a moved-from state
        EXPECT_TRUE(original.empty());
        EXPECT_EQ(original.size(), 0);
    }

    TEST_F(TensorConstructionTest, MoveConstructor_PreservesData) {
        std::vector<size_t> shape = { 2, 2 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> original(cpu_context_, shape);

        Tensor<TensorDataType::FP32, CpuMemoryResource> moved(std::move(original));

        // Verify tensor structure is preserved in moved tensor
        EXPECT_EQ(moved.shape(), shape);
        EXPECT_EQ(moved.size(), 4);
        EXPECT_FALSE(moved.empty());
        EXPECT_EQ(moved.getDeviceContext(), cpu_context_);
    }

    // ====================================================================
    // Move Assignment Tests
    // ====================================================================

    TEST_F(TensorConstructionTest, MoveAssignment) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> original(cpu_context_, shape);
        std::string original_uid = original.getUId();

        Tensor<TensorDataType::FP32, CpuMemoryResource> moved(cpu_context_, {});
        moved = std::move(original);

        EXPECT_EQ(moved.shape(), shape);
        EXPECT_EQ(moved.size(), 6);
        EXPECT_EQ(moved.getUId(), original_uid);
        EXPECT_EQ(moved.getDeviceContext(), cpu_context_);

        // The original should be in a moved-from state
        EXPECT_TRUE(original.empty());
        EXPECT_EQ(original.size(), 0);
    }

    TEST_F(TensorConstructionTest, MoveAssignment_SelfMove) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor(cpu_context_, shape);
        std::string original_uid = tensor.getUId();

        tensor = std::move(tensor);

        // Self-move should leave tensor in valid state
        EXPECT_EQ(tensor.getUId(), original_uid);
        EXPECT_EQ(tensor.shape(), shape);
        EXPECT_EQ(tensor.size(), 6);
        EXPECT_EQ(tensor.getDeviceContext(), cpu_context_);
    }

    // ====================================================================
    // Deleted Operations Tests
    // ====================================================================

    TEST_F(TensorConstructionTest, CopyOperationsAreDeleted) {
        // These should not compile - testing that copy operations are deleted
        // Uncomment these lines to verify compilation errors:

        // Tensor<TensorDataType::FP32, CpuMemoryResource> tensor1(cpu_context_, {2, 3});
        // Tensor<TensorDataType::FP32, CpuMemoryResource> tensor2(tensor1);  // Should not compile
        // Tensor<TensorDataType::FP32, CpuMemoryResource> tensor3(cpu_context_, {});
        // tensor3 = tensor1;  // Should not compile

        SUCCEED(); // This test passes if the above lines don't compile
    }

    // ====================================================================
    // Unique ID Generation Tests
    // ====================================================================

    TEST_F(TensorConstructionTest, UniqueIdGeneration) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor1(cpu_context_, {});
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor2(cpu_context_, {});

        // Each tensor should have a unique ID
        EXPECT_NE(tensor1.getUId(), tensor2.getUId());
    }

    TEST_F(TensorConstructionTest, UniqueIdGenerationWithShape) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor1(cpu_context_, shape);
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor2(cpu_context_, shape);

        // Each tensor should have a unique ID even with same shape
        EXPECT_NE(tensor1.getUId(), tensor2.getUId());
    }

    TEST_F(TensorConstructionTest, UniqueIdGenerationAfterMove) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> original(cpu_context_, shape);
        std::string original_uid = original.getUId();

        Tensor<TensorDataType::FP32, CpuMemoryResource> moved(std::move(original));

        // UID should transfer with move
        EXPECT_EQ(moved.getUId(), original_uid);

        // Creating new tensor should get different UID
        Tensor<TensorDataType::FP32, CpuMemoryResource> new_tensor(cpu_context_, shape);
        EXPECT_NE(new_tensor.getUId(), original_uid);
    }

    // ====================================================================
    // Construction with Different Memory Resources
    // ====================================================================

    TEST_F(TensorConstructionTest, ConstructWithDifferentMemoryResources) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        std::vector<size_t> shape = { 2, 3 };

        // Test construction with various memory resource types
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_tensor(cpu_context_, shape);
        Tensor<TensorDataType::FP32, CudaMemoryResource> cuda_tensor(cuda_context_, shape);
        Tensor<TensorDataType::FP32, CudaPinnedMemoryResource> pinned_tensor(cuda_context_, shape);
        Tensor<TensorDataType::FP32, CudaManagedMemoryResource> managed_tensor(cuda_context_, shape);

        // Verify all tensors have correct properties
        EXPECT_EQ(host_tensor.shape(), shape);
        EXPECT_EQ(cuda_tensor.shape(), shape);
        EXPECT_EQ(pinned_tensor.shape(), shape);
        EXPECT_EQ(managed_tensor.shape(), shape);

        EXPECT_EQ(host_tensor.size(), 6);
        EXPECT_EQ(cuda_tensor.size(), 6);
        EXPECT_EQ(pinned_tensor.size(), 6);
        EXPECT_EQ(managed_tensor.size(), 6);

        // Verify device contexts
        EXPECT_EQ(host_tensor.getDeviceContext(), cpu_context_);
        EXPECT_EQ(cuda_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(pinned_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(managed_tensor.getDeviceContext(), cuda_context_);

        // Verify memory accessibility properties
        EXPECT_TRUE(host_tensor.is_host_accessible());
        EXPECT_FALSE(host_tensor.is_device_accessible());

        EXPECT_FALSE(cuda_tensor.is_host_accessible());
        EXPECT_TRUE(cuda_tensor.is_device_accessible());

        EXPECT_TRUE(pinned_tensor.is_host_accessible());
        EXPECT_TRUE(pinned_tensor.is_device_accessible());

        EXPECT_TRUE(managed_tensor.is_host_accessible());
        EXPECT_TRUE(managed_tensor.is_device_accessible());

        // Verify all have unique UIDs
        EXPECT_NE(host_tensor.getUId(), cuda_tensor.getUId());
        EXPECT_NE(host_tensor.getUId(), pinned_tensor.getUId());
        EXPECT_NE(host_tensor.getUId(), managed_tensor.getUId());
        EXPECT_NE(cuda_tensor.getUId(), pinned_tensor.getUId());
    }

    // ====================================================================
    // Device Context Validation Tests
    // ====================================================================

    TEST_F(TensorConstructionTest, CudaMemoryResourceRequiresCudaContext) {
        std::vector<size_t> shape = { 2, 3 };

        // CPU context with CUDA memory resource should fail
        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CudaMemoryResource>(cpu_context_, shape)),
			std::runtime_error
        );

        // CPU context with CUDA pinned memory resource should fail
        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CudaPinnedMemoryResource>(cpu_context_, shape)),
			std::runtime_error
        );

        // CPU context with CUDA managed memory resource should fail
        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CudaManagedMemoryResource>(cpu_context_, shape)),
			std::runtime_error
        );
    }

    TEST_F(TensorConstructionTest, CpuMemoryResourceRequiresCpuContext) {
        std::vector<size_t> shape = { 2, 3 };

        if (has_cuda_) {
            EXPECT_THROW(
                (Tensor<TensorDataType::FP32, CpuMemoryResource>( cuda_context_, shape )),
				std::runtime_error
            );
        }
    }

    // ====================================================================
    // Constructor Edge Cases
    // ====================================================================

    TEST_F(TensorConstructionTest, ConstructorWithLargeShape) {
        std::vector<size_t> large_shape = { 100, 200, 50 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor(cpu_context_, large_shape);

        EXPECT_EQ(tensor.shape(), large_shape);
        EXPECT_EQ(tensor.size(), 1000000);  // 100 * 200 * 50
        EXPECT_EQ(tensor.rank(), 3);
        EXPECT_FALSE(tensor.empty());
        EXPECT_EQ(tensor.getDeviceContext(), cpu_context_);
    }

    TEST_F(TensorConstructionTest, ConstructorWithSingleDimension) {
        std::vector<size_t> single_dim = { 42 };
        Tensor<TensorDataType::INT32, CpuMemoryResource> tensor(cpu_context_, single_dim);

        EXPECT_EQ(tensor.shape(), single_dim);
        EXPECT_EQ(tensor.size(), 42);
        EXPECT_EQ(tensor.rank(), 1);
        EXPECT_FALSE(tensor.empty());
        EXPECT_EQ(tensor.getDeviceContext(), cpu_context_);
    }

    TEST_F(TensorConstructionTest, ConstructorWithZeroInitialization) {
        std::vector<size_t> shape = { 3, 4 };

        // Test tensor initialization (should allocate memory)
        Tensor<TensorDataType::FP32, CpuMemoryResource> float_tensor(cpu_context_, shape);
        Tensor<TensorDataType::INT32, CpuMemoryResource> int_tensor(cpu_context_, shape);

        EXPECT_EQ(float_tensor.shape(), shape);
        EXPECT_EQ(int_tensor.shape(), shape);
        EXPECT_EQ(float_tensor.size(), 12);
        EXPECT_EQ(int_tensor.size(), 12);
        EXPECT_FALSE(float_tensor.empty());
        EXPECT_FALSE(int_tensor.empty());
        EXPECT_EQ(float_tensor.getDeviceContext(), cpu_context_);
        EXPECT_EQ(int_tensor.getDeviceContext(), cpu_context_);
    }

    // ====================================================================
    // Constructor Tests with All CPU-Supported Data Types
    // ====================================================================

    TEST_F(TensorConstructionTest, ConstructorWithAllCpuSupportedDataTypes) {
        std::vector<size_t> shape = { 2, 3 };

        // Test all CPU-supported data types (host-compatible only)
        Tensor<TensorDataType::FP32, CpuMemoryResource> fp32_tensor(cpu_context_, shape);
        Tensor<TensorDataType::INT8, CpuMemoryResource> int8_tensor(cpu_context_, shape);
        Tensor<TensorDataType::INT16, CpuMemoryResource> int16_tensor(cpu_context_, shape);
        Tensor<TensorDataType::INT32, CpuMemoryResource> int32_tensor(cpu_context_, shape);
        Tensor<TensorDataType::UINT8, CpuMemoryResource> uint8_tensor(cpu_context_, shape);
        Tensor<TensorDataType::UINT16, CpuMemoryResource> uint16_tensor(cpu_context_, shape);
        Tensor<TensorDataType::UINT32, CpuMemoryResource> uint32_tensor(cpu_context_, shape);

        // Verify all tensors have correct data types
        EXPECT_EQ(fp32_tensor.getDataType(), TensorDataType::FP32);
        EXPECT_EQ(int8_tensor.getDataType(), TensorDataType::INT8);
        EXPECT_EQ(int16_tensor.getDataType(), TensorDataType::INT16);
        EXPECT_EQ(int32_tensor.getDataType(), TensorDataType::INT32);
        EXPECT_EQ(uint8_tensor.getDataType(), TensorDataType::UINT8);
        EXPECT_EQ(uint16_tensor.getDataType(), TensorDataType::UINT16);
        EXPECT_EQ(uint32_tensor.getDataType(), TensorDataType::UINT32);

        // Verify type names
        EXPECT_EQ(fp32_tensor.getDataTypeName(), "FP32");
        EXPECT_EQ(int8_tensor.getDataTypeName(), "INT8");
        EXPECT_EQ(int16_tensor.getDataTypeName(), "INT16");
        EXPECT_EQ(int32_tensor.getDataTypeName(), "INT32");
        EXPECT_EQ(uint8_tensor.getDataTypeName(), "UINT8");
        EXPECT_EQ(uint16_tensor.getDataTypeName(), "UINT16");
        EXPECT_EQ(uint32_tensor.getDataTypeName(), "UINT32");

        // All should have same shape and size
        EXPECT_EQ(fp32_tensor.size(), 6);
        EXPECT_EQ(int8_tensor.size(), 6);
        EXPECT_EQ(int16_tensor.size(), 6);
        EXPECT_EQ(int32_tensor.size(), 6);
        EXPECT_EQ(uint8_tensor.size(), 6);
        EXPECT_EQ(uint16_tensor.size(), 6);
        EXPECT_EQ(uint32_tensor.size(), 6);

        // Verify element sizes
        EXPECT_EQ(fp32_tensor.getElementSizeInBytes(), 4);
        EXPECT_EQ(int8_tensor.getElementSizeInBytes(), 1);
        EXPECT_EQ(int16_tensor.getElementSizeInBytes(), 2);
        EXPECT_EQ(int32_tensor.getElementSizeInBytes(), 4);
        EXPECT_EQ(uint8_tensor.getElementSizeInBytes(), 1);
        EXPECT_EQ(uint16_tensor.getElementSizeInBytes(), 2);
        EXPECT_EQ(uint32_tensor.getElementSizeInBytes(), 4);

        // Verify memory accessibility (all CPU tensors should be host-accessible but not device-accessible)
        EXPECT_TRUE(fp32_tensor.is_host_accessible());
        EXPECT_FALSE(fp32_tensor.is_device_accessible());
        EXPECT_TRUE(int32_tensor.is_host_accessible());
        EXPECT_FALSE(int32_tensor.is_device_accessible());

        // Verify device contexts
        EXPECT_EQ(fp32_tensor.getDeviceContext(), cpu_context_);
        EXPECT_EQ(int32_tensor.getDeviceContext(), cpu_context_);
    }

    // ====================================================================
    // Constructor Tests with All CUDA-Supported Data Types
    // ====================================================================

    TEST_F(TensorConstructionTest, ConstructorWithAllCudaSupportedDataTypes) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        std::vector<size_t> shape = { 2, 3 };

        // Test all CUDA-supported data types
        Tensor<TensorDataType::FP32, CudaMemoryResource> fp32_tensor(cuda_context_, shape);
        Tensor<TensorDataType::FP16, CudaMemoryResource> fp16_tensor(cuda_context_, shape);
        Tensor<TensorDataType::BF16, CudaMemoryResource> bf16_tensor(cuda_context_, shape);
        Tensor<TensorDataType::FP8_E4M3, CudaMemoryResource> fp8_e4m3_tensor(cuda_context_, shape);
        Tensor<TensorDataType::FP8_E5M2, CudaMemoryResource> fp8_e5m2_tensor(cuda_context_, shape);
        Tensor<TensorDataType::INT8, CudaMemoryResource> int8_tensor(cuda_context_, shape);
        Tensor<TensorDataType::INT16, CudaMemoryResource> int16_tensor(cuda_context_, shape);
        Tensor<TensorDataType::INT32, CudaMemoryResource> int32_tensor(cuda_context_, shape);
        Tensor<TensorDataType::UINT8, CudaMemoryResource> uint8_tensor(cuda_context_, shape);
        Tensor<TensorDataType::UINT16, CudaMemoryResource> uint16_tensor(cuda_context_, shape);
        Tensor<TensorDataType::UINT32, CudaMemoryResource> uint32_tensor(cuda_context_, shape);

        // Verify all tensors have correct data types
        EXPECT_EQ(fp32_tensor.getDataType(), TensorDataType::FP32);
        EXPECT_EQ(fp16_tensor.getDataType(), TensorDataType::FP16);
        EXPECT_EQ(bf16_tensor.getDataType(), TensorDataType::BF16);
        EXPECT_EQ(fp8_e4m3_tensor.getDataType(), TensorDataType::FP8_E4M3);
        EXPECT_EQ(fp8_e5m2_tensor.getDataType(), TensorDataType::FP8_E5M2);
        EXPECT_EQ(int8_tensor.getDataType(), TensorDataType::INT8);
        EXPECT_EQ(int16_tensor.getDataType(), TensorDataType::INT16);
        EXPECT_EQ(int32_tensor.getDataType(), TensorDataType::INT32);
        EXPECT_EQ(uint8_tensor.getDataType(), TensorDataType::UINT8);
        EXPECT_EQ(uint16_tensor.getDataType(), TensorDataType::UINT16);
        EXPECT_EQ(uint32_tensor.getDataType(), TensorDataType::UINT32);

        // Verify type names
        EXPECT_EQ(fp32_tensor.getDataTypeName(), "FP32");
        EXPECT_EQ(fp16_tensor.getDataTypeName(), "FP16");
        EXPECT_EQ(bf16_tensor.getDataTypeName(), "BF16");
        EXPECT_EQ(fp8_e4m3_tensor.getDataTypeName(), "FP8_E4M3");
        EXPECT_EQ(fp8_e5m2_tensor.getDataTypeName(), "FP8_E5M2");
        EXPECT_EQ(int8_tensor.getDataTypeName(), "INT8");
        EXPECT_EQ(int16_tensor.getDataTypeName(), "INT16");
        EXPECT_EQ(int32_tensor.getDataTypeName(), "INT32");
        EXPECT_EQ(uint8_tensor.getDataTypeName(), "UINT8");
        EXPECT_EQ(uint16_tensor.getDataTypeName(), "UINT16");
        EXPECT_EQ(uint32_tensor.getDataTypeName(), "UINT32");

        // All should have same shape and size
        EXPECT_EQ(fp32_tensor.size(), 6);
        EXPECT_EQ(fp16_tensor.size(), 6);
        EXPECT_EQ(bf16_tensor.size(), 6);
        EXPECT_EQ(fp8_e4m3_tensor.size(), 6);
        EXPECT_EQ(fp8_e5m2_tensor.size(), 6);
        EXPECT_EQ(int8_tensor.size(), 6);
        EXPECT_EQ(int16_tensor.size(), 6);
        EXPECT_EQ(int32_tensor.size(), 6);
        EXPECT_EQ(uint8_tensor.size(), 6);
        EXPECT_EQ(uint16_tensor.size(), 6);
        EXPECT_EQ(uint32_tensor.size(), 6);

        // Verify element sizes
        EXPECT_EQ(fp32_tensor.getElementSizeInBytes(), 4);
        EXPECT_EQ(fp16_tensor.getElementSizeInBytes(), 2);
        EXPECT_EQ(bf16_tensor.getElementSizeInBytes(), 2);
        EXPECT_EQ(fp8_e4m3_tensor.getElementSizeInBytes(), 1);
        EXPECT_EQ(fp8_e5m2_tensor.getElementSizeInBytes(), 1);
        EXPECT_EQ(int8_tensor.getElementSizeInBytes(), 1);
        EXPECT_EQ(int16_tensor.getElementSizeInBytes(), 2);
        EXPECT_EQ(int32_tensor.getElementSizeInBytes(), 4);
        EXPECT_EQ(uint8_tensor.getElementSizeInBytes(), 1);
        EXPECT_EQ(uint16_tensor.getElementSizeInBytes(), 2);
        EXPECT_EQ(uint32_tensor.getElementSizeInBytes(), 4);

        // Verify device accessibility (all CUDA tensors should be device-accessible but not host-accessible)
        EXPECT_FALSE(fp32_tensor.is_host_accessible());
        EXPECT_TRUE(fp32_tensor.is_device_accessible());
        EXPECT_FALSE(fp16_tensor.is_host_accessible());
        EXPECT_TRUE(fp16_tensor.is_device_accessible());

        // Verify device contexts
        EXPECT_EQ(fp32_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(fp16_tensor.getDeviceContext(), cuda_context_);
    }

    // ====================================================================
    // Constructor Tests with Managed Memory
    // ====================================================================

    TEST_F(TensorConstructionTest, ConstructorWithManagedMemoryAllDataTypes) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        std::vector<size_t> shape = { 2, 3 };

        // Test selection of CUDA data types with managed memory (both host and device accessible)
        Tensor<TensorDataType::FP32, CudaManagedMemoryResource> fp32_tensor(cuda_context_, shape);
        Tensor<TensorDataType::FP16, CudaManagedMemoryResource> fp16_tensor(cuda_context_, shape);
        Tensor<TensorDataType::BF16, CudaManagedMemoryResource> bf16_tensor(cuda_context_, shape);
        Tensor<TensorDataType::INT8, CudaManagedMemoryResource> int8_tensor(cuda_context_, shape);
        Tensor<TensorDataType::INT32, CudaManagedMemoryResource> int32_tensor(cuda_context_, shape);
        Tensor<TensorDataType::UINT32, CudaManagedMemoryResource> uint32_tensor(cuda_context_, shape);

        // Verify managed memory accessibility (should be both host and device accessible)
        EXPECT_TRUE(fp32_tensor.is_host_accessible());
        EXPECT_TRUE(fp32_tensor.is_device_accessible());
        EXPECT_TRUE(fp16_tensor.is_host_accessible());
        EXPECT_TRUE(fp16_tensor.is_device_accessible());
        EXPECT_TRUE(bf16_tensor.is_host_accessible());
        EXPECT_TRUE(bf16_tensor.is_device_accessible());
        EXPECT_TRUE(int8_tensor.is_host_accessible());
        EXPECT_TRUE(int8_tensor.is_device_accessible());
        EXPECT_TRUE(int32_tensor.is_host_accessible());
        EXPECT_TRUE(int32_tensor.is_device_accessible());
        EXPECT_TRUE(uint32_tensor.is_host_accessible());
        EXPECT_TRUE(uint32_tensor.is_device_accessible());

        // Verify data types
        EXPECT_EQ(fp32_tensor.getDataType(), TensorDataType::FP32);
        EXPECT_EQ(fp16_tensor.getDataType(), TensorDataType::FP16);
        EXPECT_EQ(bf16_tensor.getDataType(), TensorDataType::BF16);
        EXPECT_EQ(int8_tensor.getDataType(), TensorDataType::INT8);
        EXPECT_EQ(int32_tensor.getDataType(), TensorDataType::INT32);
        EXPECT_EQ(uint32_tensor.getDataType(), TensorDataType::UINT32);

        // Verify device contexts
        EXPECT_EQ(fp32_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(fp16_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(bf16_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(int8_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(int32_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(uint32_tensor.getDeviceContext(), cuda_context_);
    }

    // ====================================================================
    // Constructor Tests with Pinned Memory
    // ====================================================================

    TEST_F(TensorConstructionTest, ConstructorWithPinnedMemoryAllDataTypes) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        std::vector<size_t> shape = { 2, 3 };

        // Test selection of CUDA data types with pinned memory (both host and device accessible)
        Tensor<TensorDataType::FP32, CudaPinnedMemoryResource> fp32_tensor(cuda_context_, shape);
        Tensor<TensorDataType::FP16, CudaPinnedMemoryResource> fp16_tensor(cuda_context_, shape);
        Tensor<TensorDataType::BF16, CudaPinnedMemoryResource> bf16_tensor(cuda_context_, shape);
        Tensor<TensorDataType::FP8_E4M3, CudaPinnedMemoryResource> fp8_e4m3_tensor(cuda_context_, shape);
        Tensor<TensorDataType::INT8, CudaPinnedMemoryResource> int8_tensor(cuda_context_, shape);
        Tensor<TensorDataType::INT32, CudaPinnedMemoryResource> int32_tensor(cuda_context_, shape);
        Tensor<TensorDataType::UINT16, CudaPinnedMemoryResource> uint16_tensor(cuda_context_, shape);

        // Verify pinned memory accessibility (should be both host and device accessible)
        EXPECT_TRUE(fp32_tensor.is_host_accessible());
        EXPECT_TRUE(fp32_tensor.is_device_accessible());
        EXPECT_TRUE(fp16_tensor.is_host_accessible());
        EXPECT_TRUE(fp16_tensor.is_device_accessible());
        EXPECT_TRUE(bf16_tensor.is_host_accessible());
        EXPECT_TRUE(bf16_tensor.is_device_accessible());
        EXPECT_TRUE(fp8_e4m3_tensor.is_host_accessible());
        EXPECT_TRUE(fp8_e4m3_tensor.is_device_accessible());
        EXPECT_TRUE(int8_tensor.is_host_accessible());
        EXPECT_TRUE(int8_tensor.is_device_accessible());
        EXPECT_TRUE(int32_tensor.is_host_accessible());
        EXPECT_TRUE(int32_tensor.is_device_accessible());
        EXPECT_TRUE(uint16_tensor.is_host_accessible());
        EXPECT_TRUE(uint16_tensor.is_device_accessible());

        // Verify data types
        EXPECT_EQ(fp32_tensor.getDataType(), TensorDataType::FP32);
        EXPECT_EQ(fp16_tensor.getDataType(), TensorDataType::FP16);
        EXPECT_EQ(bf16_tensor.getDataType(), TensorDataType::BF16);
        EXPECT_EQ(fp8_e4m3_tensor.getDataType(), TensorDataType::FP8_E4M3);
        EXPECT_EQ(int8_tensor.getDataType(), TensorDataType::INT8);
        EXPECT_EQ(int32_tensor.getDataType(), TensorDataType::INT32);
        EXPECT_EQ(uint16_tensor.getDataType(), TensorDataType::UINT16);

        // Verify device contexts
        EXPECT_EQ(fp32_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(fp16_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(bf16_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(fp8_e4m3_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(int8_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(int32_tensor.getDeviceContext(), cuda_context_);
        EXPECT_EQ(uint16_tensor.getDeviceContext(), cuda_context_);
    }

    // ====================================================================
    // Device Context Integration Tests
    // ====================================================================

    TEST_F(TensorConstructionTest, DeviceContextIntegration) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        std::vector<size_t> shape = { 2, 3 };

        Tensor<TensorDataType::FP32, CudaMemoryResource> tensor(cuda_context_, shape);

        // Verify device context is accessible
        auto retrieved_context = tensor.getDeviceContext();
        EXPECT_EQ(retrieved_context, cuda_context_);
        EXPECT_TRUE(retrieved_context->isCudaDevice());
        EXPECT_EQ(retrieved_context->getDeviceId(), 0);  // cuda:0

        // Verify memory resource is accessible
        // FIXME:
        /*auto memory_resource = tensor.getMemoryResource();
        EXPECT_NE(memory_resource, nullptr);*/
    }

    TEST_F(TensorConstructionTest, DeviceContextStringRepresentation) {
        std::vector<size_t> shape = { 2, 3 };

        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_tensor(cpu_context_, shape);
        std::string cpu_str = cpu_tensor.toString();

        // Should include device information in string representation
        EXPECT_NE(cpu_str.find("CPU"), std::string::npos);

        if (has_cuda_) {
            Tensor<TensorDataType::FP32, CudaMemoryResource> cuda_tensor(cuda_context_, shape);
            std::string cuda_str = cuda_tensor.toString();
            EXPECT_NE(cuda_str.find("CUDA:0"), std::string::npos);
        }
    }

    // Note: The remaining tests (Type Constraint Validation Tests) remain unchanged
    // as they are compile-time tests and don't require runtime CUDA availability checks

    // ====================================================================
    // Type Constraint Validation Tests
    // ====================================================================

    TEST_F(TensorConstructionTest, TypeConstraintValidation_ManagedMemoryCompatibility) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        // Test that managed memory works with both host and device data types

        // Host-compatible types with managed memory
        {
            Tensor<TensorDataType::FP32, CudaManagedMemoryResource> tensor(cuda_context_, { 2, 3 });
            EXPECT_TRUE(tensor.is_host_accessible());
            EXPECT_TRUE(tensor.is_device_accessible());
            EXPECT_EQ(tensor.getDataType(), TensorDataType::FP32);
            EXPECT_EQ(tensor.getDeviceContext(), cuda_context_);
        }

        // Device-specific types with managed memory
        {
            Tensor<TensorDataType::FP16, CudaManagedMemoryResource> tensor(cuda_context_, { 2, 3 });
            EXPECT_TRUE(tensor.is_host_accessible());
            EXPECT_TRUE(tensor.is_device_accessible());
            EXPECT_EQ(tensor.getDataType(), TensorDataType::FP16);
            EXPECT_EQ(tensor.getDeviceContext(), cuda_context_);
        }

        // Device-only types with managed memory
        {
            Tensor<TensorDataType::FP8_E4M3, CudaManagedMemoryResource> tensor(cuda_context_, { 2, 3 });
            EXPECT_TRUE(tensor.is_host_accessible());
            EXPECT_TRUE(tensor.is_device_accessible());
            EXPECT_EQ(tensor.getDataType(), TensorDataType::FP8_E4M3);
            EXPECT_EQ(tensor.getDeviceContext(), cuda_context_);
        }
    }

    TEST_F(TensorConstructionTest, TypeConstraintValidation_PinnedMemoryCompatibility) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        // Test that pinned memory works correctly with various data types

        // Standard floating-point types
        {
            Tensor<TensorDataType::FP32, CudaPinnedMemoryResource> tensor(cuda_context_, { 2, 3 });
            EXPECT_TRUE(tensor.is_host_accessible());
            EXPECT_TRUE(tensor.is_device_accessible());
            EXPECT_EQ(tensor.getDataType(), TensorDataType::FP32);
            EXPECT_EQ(tensor.getDeviceContext(), cuda_context_);
        }

        // Half precision types
        {
            Tensor<TensorDataType::FP16, CudaPinnedMemoryResource> tensor(cuda_context_, { 2, 3 });
            EXPECT_TRUE(tensor.is_host_accessible());
            EXPECT_TRUE(tensor.is_device_accessible());
            EXPECT_EQ(tensor.getDataType(), TensorDataType::FP16);
            EXPECT_EQ(tensor.getDeviceContext(), cuda_context_);
        }

        // Integer types
        {
            Tensor<TensorDataType::INT32, CudaPinnedMemoryResource> tensor(cuda_context_, { 2, 3 });
            EXPECT_TRUE(tensor.is_host_accessible());
            EXPECT_TRUE(tensor.is_device_accessible());
            EXPECT_EQ(tensor.getDataType(), TensorDataType::INT32);
            EXPECT_EQ(tensor.getDeviceContext(), cuda_context_);
        }
    }

    TEST_F(TensorConstructionTest, TypeConstraintValidation_DataTypeTraitsConsistency) {
        // Test that data type traits are consistent across different tensor configurations

        // Verify element sizes are consistent
        {
            Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_tensor(cpu_context_, { 2, 3 });

            if (has_cuda_) {
                Tensor<TensorDataType::FP32, CudaMemoryResource> cuda_tensor(cuda_context_, { 2, 3 });
                EXPECT_EQ(cpu_tensor.getElementSizeInBytes(), cuda_tensor.getElementSizeInBytes());
            }

            EXPECT_EQ(cpu_tensor.getElementSizeInBytes(), 4);  // FP32 = 4 bytes
        }

        // Verify data type names are consistent
        if (has_cuda_) {
            Tensor<TensorDataType::FP16, CudaMemoryResource> cuda_tensor(cuda_context_, { 2, 3 });
            Tensor<TensorDataType::FP16, CudaManagedMemoryResource> managed_tensor(cuda_context_, { 2, 3 });

            EXPECT_EQ(cuda_tensor.getDataTypeName(), managed_tensor.getDataTypeName());
            EXPECT_EQ(cuda_tensor.getDataTypeName(), "FP16");
        }

        // Verify device-only type characteristics
        if (has_cuda_) {
            Tensor<TensorDataType::FP8_E4M3, CudaMemoryResource> tensor(cuda_context_, { 2, 3 });
            EXPECT_EQ(tensor.getElementSizeInBytes(), 1);  // FP8 = 1 byte
            EXPECT_EQ(tensor.getDataTypeName(), "FP8_E4M3");
        }
    }

    TEST_F(TensorConstructionTest, TypeConstraintValidation_CompileTimeProperties) {
        // Test that compile-time properties are correctly defined

        // Test static constexpr members
        {
            using TensorType = Tensor<TensorDataType::FP32, CpuMemoryResource>;

            static_assert(TensorType::data_type == TensorDataType::FP32);
            static_assert(TensorType::element_size == 4);
            static_assert(TensorType::is_float_type == true);
            static_assert(TensorType::is_integer_type == false);
            static_assert(TensorType::is_device_only == false);
        }

        // Test device-only type properties
        {
            using DeviceOnlyTensorType = Tensor<TensorDataType::FP8_E4M3, CudaMemoryResource>;

            static_assert(DeviceOnlyTensorType::data_type == TensorDataType::FP8_E4M3);
            static_assert(DeviceOnlyTensorType::element_size == 1);
            static_assert(DeviceOnlyTensorType::is_float_type == true);
            static_assert(DeviceOnlyTensorType::is_integer_type == false);
            static_assert(DeviceOnlyTensorType::is_device_only == true);
        }

        // Test integer type properties
        {
            using IntTensorType = Tensor<TensorDataType::INT32, CpuMemoryResource>;

            static_assert(IntTensorType::data_type == TensorDataType::INT32);
            static_assert(IntTensorType::element_size == 4);
            static_assert(IntTensorType::is_float_type == false);
            static_assert(IntTensorType::is_integer_type == true);
            static_assert(IntTensorType::is_device_only == false);
        }
    }

    TEST_F(TensorConstructionTest, TypeConstraintValidation_MemoryResourceInheritance) {
        // Test that memory resources properly inherit from base classes

        // Verify all memory resources inherit from base
        static_assert(std::is_base_of_v<MemoryResource, CpuMemoryResource>);
        static_assert(std::is_base_of_v<MemoryResource, CudaMemoryResource>);
        static_assert(std::is_base_of_v<MemoryResource, CudaManagedMemoryResource>);
        static_assert(std::is_base_of_v<MemoryResource, CudaPinnedMemoryResource>);

        // Verify memory accessibility properties are correctly defined
        static_assert(CpuMemoryResource::is_host_accessible == true);
        static_assert(CpuMemoryResource::is_device_accessible == false);

        static_assert(CudaMemoryResource::is_host_accessible == false);
        static_assert(CudaMemoryResource::is_device_accessible == true);

        static_assert(CudaManagedMemoryResource::is_host_accessible == true);
        static_assert(CudaManagedMemoryResource::is_device_accessible == true);

        static_assert(CudaPinnedMemoryResource::is_host_accessible == true);
        static_assert(CudaPinnedMemoryResource::is_device_accessible == true);
    }

    TEST_F(TensorConstructionTest, TypeConstraintValidation_ConceptValidation) {
        // Test that the isValidTensor concept works correctly

        // Valid combinations should satisfy the concept
        static_assert(isValidTensor<TensorDataType::FP32, CpuMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP16, CudaMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP8_E4M3, CudaMemoryResource>);
        static_assert(isValidTensor<TensorDataType::INT32, CpuMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP32, CudaManagedMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP16, CudaPinnedMemoryResource>);

        // Device-only types should require device-accessible memory
        static_assert(isValidTensor<TensorDataType::FP8_E4M3, CudaManagedMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP8_E5M2, CudaPinnedMemoryResource>);

        // Test that concept validation catches invalid combinations at compile time
        // These should fail to compile:
        // static_assert( !isValidTensor<TensorDataType::FP8_E4M3, CpuMemoryResource> );
    }
}