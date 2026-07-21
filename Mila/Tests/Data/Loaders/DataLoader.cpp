/**
 * @file DataLoader.cpp
 * @brief Base-contract tests for DataLoader<TInput, TTarget, TMemoryResource> (CPU).
 *
 * DataLoader is the base every concrete loader on the training path derives from
 * (MnistDataLoader, TokenSequenceLoader). This proves the device-agnostic iteration
 * contract once -- batch counting, hasNext/reset, the tensor-access surface, the
 * data-type/mixed-precision traits, and getDatasetInfo/validateCurrentBatch -- via
 * a mock loader, so concrete loaders need only assert their own data semantics.
 *
 * DataLoader is abstract, so the mock IS the surface: MockDataLoader implements the
 * pure-virtual remainder (numBatches / nextBatch / inputs / targets) over fixed
 * input {batch,10} and target {batch,5} tensors. The mock does not fill values --
 * allocated tensors are already non-empty, which is all the contract tests need.
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate. The pinned-memory
 * (CudaPinnedMemoryResource / PinnedDataLoader) cases are CUDA-only in the Src and
 * live in the DataLoader.Cuda.cpp companion.
 */

#include <gtest/gtest.h>
#include <memory>
#include <cstdint>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Data
{
    using namespace Mila::Data;
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        template<TensorDataType TInput, TensorDataType TTarget = TInput, typename TMemoryResource = CpuMemoryResource>
        class MockDataLoader : public DataLoader<TInput, TTarget, TMemoryResource>
        {
        public:
            using Base = DataLoader<TInput, TTarget, TMemoryResource>;
            using InputTensor = typename Base::InputTensor;
            using TargetTensor = typename Base::TargetTensor;

            MockDataLoader( int64_t batch_size, int64_t num_batches, DeviceId device_id = Device::Cpu() )
                : Base( batch_size ),
                num_batches_( num_batches ),
                input_tensor_( device_id, shape_t{ batch_size, 10 } ),
                target_tensor_( device_id, shape_t{ batch_size, 5 } )
            {}

            int64_t numBatches() const override
            {
                return num_batches_;
            }

            void nextBatch() override
            {
                if ( this->hasNext() )
                {
                    this->incrementBatch();
                }
            }

            InputTensor& inputs() override
            {
                return input_tensor_;
            }

            const InputTensor& inputs() const override
            {
                return input_tensor_;
            }

            TargetTensor& targets() override
            {
                return target_tensor_;
            }

            const TargetTensor& targets() const override
            {
                return target_tensor_;
            }

        private:
            int64_t num_batches_;
            InputTensor input_tensor_;
            TargetTensor target_tensor_;
        };
    }

    class DataLoaderTests : public ::testing::Test
    {
    protected:
        static constexpr int64_t kBatchSize = 32;
        static constexpr int64_t kNumBatches = 10;
    };

    // ====================================================================
    // Construction & properties
    // ====================================================================

    TEST_F( DataLoaderTests, Construct_ValidBatchSize )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        EXPECT_EQ( loader.batchSize(), kBatchSize );
        EXPECT_EQ( loader.numBatches(), kNumBatches );
        EXPECT_EQ( loader.currentBatch(), 0 );
    }

    TEST_F( DataLoaderTests, Construct_ThrowsOnZeroBatchSize )
    {
        EXPECT_THROW( ( MockDataLoader<TensorDataType::FP32>( 0, kNumBatches ) ), std::invalid_argument );
    }

    TEST_F( DataLoaderTests, GetDatasetInfo_DescribesLoader )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        const std::string info = loader.getDatasetInfo();

        EXPECT_NE( info.find( "DataLoader" ), std::string::npos );
        EXPECT_NE( info.find( "batches" ), std::string::npos );
        EXPECT_NE( info.find( "samples per batch" ), std::string::npos );
    }

    // ====================================================================
    // Batch iteration
    // ====================================================================

    TEST_F( DataLoaderTests, Iteration_AdvancesSequentially )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        EXPECT_EQ( loader.currentBatch(), 0 );
        EXPECT_TRUE( loader.hasNext() );

        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 1 );

        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 2 );
        EXPECT_TRUE( loader.hasNext() );
    }

    TEST_F( DataLoaderTests, Iteration_HasNextTracksRemaining )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        EXPECT_TRUE( loader.hasNext() );

        for ( int64_t i = 0; i < kNumBatches - 1; ++i )
        {
            loader.nextBatch();
            EXPECT_TRUE( loader.hasNext() );
        }

        loader.nextBatch();
        EXPECT_FALSE( loader.hasNext() );
        EXPECT_EQ( loader.currentBatch(), kNumBatches );
    }

    TEST_F( DataLoaderTests, Iteration_DoesNotAdvancePastExhaustion )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        for ( int64_t i = 0; i < kNumBatches; ++i )
        {
            EXPECT_TRUE( loader.hasNext() );
            loader.nextBatch();
        }

        EXPECT_EQ( loader.currentBatch(), kNumBatches );
        EXPECT_FALSE( loader.hasNext() );

        // Past exhaustion, nextBatch() is a guarded no-op.
        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), kNumBatches );
        EXPECT_FALSE( loader.hasNext() );
    }

    TEST_F( DataLoaderTests, Reset_ReturnsToBeginning )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        loader.nextBatch();
        loader.nextBatch();
        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 3 );

        loader.reset();
        EXPECT_EQ( loader.currentBatch(), 0 );
        EXPECT_TRUE( loader.hasNext() );
    }

    TEST_F( DataLoaderTests, Reset_RestartsAfterFullIteration )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        for ( int64_t i = 0; i < kNumBatches; ++i )
        {
            loader.nextBatch();
        }
        EXPECT_FALSE( loader.hasNext() );

        loader.reset();
        EXPECT_EQ( loader.currentBatch(), 0 );
        EXPECT_TRUE( loader.hasNext() );
    }

    // ====================================================================
    // Tensor access
    // ====================================================================

    TEST_F( DataLoaderTests, Inputs_HaveBatchShape )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        const auto& inputs = loader.inputs();

        EXPECT_EQ( inputs.shape()[ 0 ], kBatchSize );
        EXPECT_EQ( inputs.shape()[ 1 ], 10 );
    }

    TEST_F( DataLoaderTests, Targets_HaveBatchShape )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        const auto& targets = loader.targets();

        EXPECT_EQ( targets.shape()[ 0 ], kBatchSize );
        EXPECT_EQ( targets.shape()[ 1 ], 5 );
    }

    TEST_F( DataLoaderTests, Tensors_AreNonEmpty )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        EXPECT_FALSE( loader.inputs().empty() );
        EXPECT_FALSE( loader.targets().empty() );
        EXPECT_GT( loader.inputs().size(), 0u );
        EXPECT_GT( loader.targets().size(), 0u );
    }

    TEST_F( DataLoaderTests, Tensors_ConstAccess )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );
        const auto& const_loader = loader;

        EXPECT_EQ( const_loader.inputs().shape()[ 0 ], kBatchSize );
        EXPECT_EQ( const_loader.targets().shape()[ 0 ], kBatchSize );
    }

    TEST_F( DataLoaderTests, ValidateCurrentBatch_PassesForAllocatedTensors )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        EXPECT_TRUE( loader.validateCurrentBatch() );

        loader.nextBatch();
        EXPECT_TRUE( loader.validateCurrentBatch() );
    }

    // ====================================================================
    // Data-type axis (single + mixed precision)
    // ====================================================================

    TEST_F( DataLoaderTests, DataType_FP32 )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        EXPECT_EQ( loader.input_data_type, TensorDataType::FP32 );
        EXPECT_EQ( loader.target_data_type, TensorDataType::FP32 );
        EXPECT_FALSE( loader.supportsMixedPrecision() );
    }

    TEST_F( DataLoaderTests, DataType_INT32 )
    {
        MockDataLoader<TensorDataType::INT32> loader( kBatchSize, kNumBatches );

        EXPECT_EQ( loader.input_data_type, TensorDataType::INT32 );
        EXPECT_EQ( loader.target_data_type, TensorDataType::INT32 );
    }

    TEST_F( DataLoaderTests, DataType_MixedPrecisionFP32InputInt32Target )
    {
        MockDataLoader<TensorDataType::FP32, TensorDataType::INT32> loader( kBatchSize, kNumBatches );

        EXPECT_EQ( loader.input_data_type, TensorDataType::FP32 );
        EXPECT_EQ( loader.target_data_type, TensorDataType::INT32 );
        EXPECT_TRUE( loader.supportsMixedPrecision() );

        loader.nextBatch();
        EXPECT_EQ( loader.inputs().shape()[ 0 ], kBatchSize );
        EXPECT_EQ( loader.targets().shape()[ 0 ], kBatchSize );
    }

    // ====================================================================
    // CPU memory resource + alias
    // ====================================================================

    TEST_F( DataLoaderTests, CpuMemoryResource_NotPinned )
    {
        MockDataLoader<TensorDataType::FP32, TensorDataType::FP32, CpuMemoryResource> loader( kBatchSize, kNumBatches );

        EXPECT_EQ( loader.batchSize(), kBatchSize );
        EXPECT_FALSE( loader.usesPinnedMemory() );

        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 1 );
    }

    TEST_F( DataLoaderTests, CpuDataLoaderAlias_IsCpuMemoryResource )
    {
        // The CpuDataLoader<> alias resolves to DataLoader<..., CpuMemoryResource>;
        // a concrete loader over it must be CPU (non-pinned).
        class TestCpuLoader : public CpuDataLoader<TensorDataType::FP32>
        {
        public:
            TestCpuLoader( int64_t batch_size, int64_t num_batches )
                : CpuDataLoader<TensorDataType::FP32>( batch_size ),
                num_batches_( num_batches ),
                input_( Device::Cpu(), shape_t{ batch_size, 10 } ),
                target_( Device::Cpu(), shape_t{ batch_size, 5 } )
            {}

            int64_t numBatches() const override { return num_batches_; }
            void nextBatch() override { this->incrementBatch(); }
            InputTensor& inputs() override { return input_; }
            const InputTensor& inputs() const override { return input_; }
            TargetTensor& targets() override { return target_; }
            const TargetTensor& targets() const override { return target_; }

        private:
            int64_t num_batches_;
            InputTensor input_;
            TargetTensor target_;
        };

        TestCpuLoader loader( kBatchSize, kNumBatches );

        EXPECT_EQ( loader.batchSize(), kBatchSize );
        EXPECT_FALSE( loader.usesPinnedMemory() );
    }

    // ====================================================================
    // Complete iteration workflow (two epochs)
    // ====================================================================

    TEST_F( DataLoaderTests, CompleteIteration_Workflow )
    {
        MockDataLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        int64_t iterations = 0;

        while ( loader.hasNext() )
        {
            loader.nextBatch();

            EXPECT_TRUE( loader.validateCurrentBatch() );
            EXPECT_FALSE( loader.inputs().empty() );
            EXPECT_FALSE( loader.targets().empty() );

            ++iterations;
        }

        EXPECT_EQ( iterations, kNumBatches );
        EXPECT_EQ( loader.currentBatch(), kNumBatches );

        loader.reset();
        EXPECT_EQ( loader.currentBatch(), 0 );
        EXPECT_TRUE( loader.hasNext() );
    }
}
