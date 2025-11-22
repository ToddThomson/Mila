#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include <cstdint>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Dnn::Data::Tests
{
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TInput, TensorDataType TTarget = TInput, typename TMemoryResource = CpuMemoryResource>
    class MockDataLoader : public DatasetLoader<TInput, TTarget, TMemoryResource>
    {
    public:
        MockDataLoader( int64_t batch_size, int64_t num_batches, const std::string& device_name = "CPU" )
            : DatasetLoader<TInput, TTarget, TMemoryResource>( batch_size ),
            num_batches_( num_batches ),
            device_name_( device_name ),
            input_tensor_( device_name, { static_cast<int64_t>(batch_size), 10 } ),
            target_tensor_( device_name, { static_cast<int64_t>(batch_size), 5 } )
        {
            // Initialize tensors using Mila tensor initialization helpers
            initializeTensors();
        }

        int64_t numBatches() const override
        {
            return num_batches_;
        }

        void nextBatch() override
        {
            if (this->hasNext())
            {
                // Simulate loading a new batch by reinitializing with different values
                // based on current batch index
                updateTensorsForBatch( this->currentBatch() );

                this->incrementBatch();
            }
        }

        typename DatasetLoader<TInput, TTarget, TMemoryResource>::InputTensor& inputs() override
        {
            return input_tensor_;
        }

        const typename DatasetLoader<TInput, TTarget, TMemoryResource>::InputTensor& inputs() const override
        {
            return input_tensor_;
        }

        typename DatasetLoader<TInput, TTarget, TMemoryResource>::TargetTensor& targets() override
        {
            return target_tensor_;
        }

        const typename DatasetLoader<TInput, TTarget, TMemoryResource>::TargetTensor& targets() const override
        {
            return target_tensor_;
        }

    private:
        void initializeTensors()
        {
            // Use Mila tensor initialization helpers
            if constexpr (TensorDataTypeTraits<TInput>::is_integer_type)
            {
                random( input_tensor_, 0, 100 );
            }
            else
            {
                random( input_tensor_, 0.0f, 1.0f );
            }

            if constexpr (TensorDataTypeTraits<TTarget>::is_integer_type)
            {
                random( target_tensor_, 0, 10 );
            }
            else
            {
                random( target_tensor_, -1.0f, 1.0f );
            }
        }

        void updateTensorsForBatch( size_t batch_idx )
        {
            // Simulate different data per batch using different random ranges
            float offset = static_cast<float>(batch_idx) * 0.1f;

            if constexpr (TensorDataTypeTraits<TInput>::is_integer_type)
            {
                int32_t int_offset = static_cast<int32_t>(batch_idx);
                random( input_tensor_, int_offset, int_offset + 100 );
            }
            else
            {
                random( input_tensor_, offset, offset + 1.0f );
            }

            if constexpr (TensorDataTypeTraits<TTarget>::is_integer_type)
            {
                int32_t int_offset = static_cast<int32_t>(batch_idx);
                random( target_tensor_, int_offset, int_offset + 10 );
            }
            else
            {
                random( target_tensor_, -1.0f + offset, 1.0f + offset );
            }
        }

        int64_t num_batches_;
        std::string device_name_;
        typename DatasetLoader<TInput, TTarget, TMemoryResource>::InputTensor input_tensor_;
        typename DatasetLoader<TInput, TTarget, TMemoryResource>::TargetTensor target_tensor_;
    };

    class DataLoaderTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            batch_size_ = 32;
            num_batches_ = 10;
        }

        int64_t batch_size_;
        int64_t num_batches_;
    };

    // ====================================================================
    // Constructor and Basic Properties Tests
    // ====================================================================

    TEST_F( DataLoaderTest, Constructor_ValidBatchSize )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        EXPECT_EQ( loader.batchSize(), batch_size_);
        EXPECT_EQ( loader.numBatches(), num_batches_ );
        EXPECT_EQ( loader.currentBatch(), 0);
    }

    TEST_F( DataLoaderTest, Constructor_ZeroBatchSize_ThrowsException )
    {
        EXPECT_THROW(
            (MockDataLoader<TensorDataType::FP32>( 0, num_batches_ )),
            std::invalid_argument
        );
    }

    TEST_F( DataLoaderTest, GetDatasetInfo )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        std::string info = loader.getDatasetInfo();

        EXPECT_NE( info.find( "DataLoader" ), std::string::npos );
        EXPECT_NE( info.find( "batches" ), std::string::npos );
        EXPECT_NE( info.find( "samples per batch" ), std::string::npos );
    }

    // ====================================================================
    // Batch Iteration Tests
    // ====================================================================

    TEST_F( DataLoaderTest, BatchIteration_Sequential )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        EXPECT_EQ( loader.currentBatch(), 0 );
        EXPECT_TRUE( loader.hasNext() );

        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 1 );
        EXPECT_TRUE( loader.hasNext() );

        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 2 );
        EXPECT_TRUE( loader.hasNext() );
    }

    TEST_F( DataLoaderTest, BatchIteration_HasNext )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        // Should have next before loading any batches
        EXPECT_TRUE( loader.hasNext() );

        // Load all but last batch
        for (size_t i = 0; i < num_batches_ - 1; ++i)
        {
            loader.nextBatch();
            EXPECT_TRUE( loader.hasNext() );
        }

        // Load last batch
        loader.nextBatch();
        EXPECT_FALSE( loader.hasNext() );
        EXPECT_EQ( loader.currentBatch(), num_batches_ );
    }

    TEST_F( DataLoaderTest, BatchIteration_Exhaustion )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        // Process all batches
        for (size_t i = 0; i < num_batches_; ++i)
        {
            EXPECT_TRUE( loader.hasNext() );
            loader.nextBatch();
        }

        EXPECT_EQ( loader.currentBatch(), num_batches_ );
        EXPECT_FALSE( loader.hasNext() );

        // Try to move beyond available batches - should not change state
        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), num_batches_ );
        EXPECT_FALSE( loader.hasNext() );
    }

    TEST_F( DataLoaderTest, Reset_ResetsToBeginning )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        // Advance several batches
        loader.nextBatch();
        loader.nextBatch();
        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 3 );

        // Reset should return to batch 0
        loader.reset();
        EXPECT_EQ( loader.currentBatch(), 0 );
        EXPECT_TRUE( loader.hasNext() );
    }

    TEST_F( DataLoaderTest, Reset_AfterFullIteration )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        // Exhaust all batches
        for (size_t i = 0; i < num_batches_; ++i)
        {
            loader.nextBatch();
        }

        EXPECT_FALSE( loader.hasNext() );

        // Reset and verify we can iterate again
        loader.reset();
        EXPECT_EQ( loader.currentBatch(), 0 );
        EXPECT_TRUE( loader.hasNext() );
    }

    // ====================================================================
    // Tensor Access Tests
    // ====================================================================

    TEST_F( DataLoaderTest, TensorAccess_InputShape )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        const auto& inputs = loader.inputs();

        EXPECT_EQ( inputs.shape()[0], static_cast<int64_t>(batch_size_) );
        EXPECT_EQ( inputs.shape()[1], 10 );
    }

    TEST_F( DataLoaderTest, TensorAccess_TargetShape )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        const auto& targets = loader.targets();

        EXPECT_EQ( targets.shape()[0], static_cast<int64_t>(batch_size_) );
        EXPECT_EQ( targets.shape()[1], 5 );
    }

    TEST_F( DataLoaderTest, TensorAccess_NonEmptyAfterInitialization )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        EXPECT_FALSE( loader.inputs().empty() );
        EXPECT_FALSE( loader.targets().empty() );
        EXPECT_GT( loader.inputs().size(), 0 );
        EXPECT_GT( loader.targets().size(), 0 );
    }

    TEST_F( DataLoaderTest, TensorAccess_ConstAccess )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        const auto& const_loader = loader;

        // Test const access to tensors
        const auto& const_inputs = const_loader.inputs();
        const auto& const_targets = const_loader.targets();

        EXPECT_EQ( const_inputs.shape()[0], static_cast<int64_t>(batch_size_) );
        EXPECT_EQ( const_targets.shape()[0], static_cast<int64_t>(batch_size_) );
    }

    TEST_F( DataLoaderTest, ValidateCurrentBatch_ValidData )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        EXPECT_TRUE( loader.validateCurrentBatch() );

        loader.nextBatch();
        EXPECT_TRUE( loader.validateCurrentBatch() );
    }

    // ====================================================================
    // Different Data Type Tests
    // ====================================================================

    TEST_F( DataLoaderTest, FloatPrecision_FP32 )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        EXPECT_EQ( loader.batchSize(), batch_size_ );
        EXPECT_EQ( loader.input_data_type, TensorDataType::FP32 );
        EXPECT_EQ( loader.target_data_type, TensorDataType::FP32 );
        EXPECT_FALSE( loader.supportsMixedPrecision() );
    }

    TEST_F( DataLoaderTest, IntegerType_INT32 )
    {
        MockDataLoader<TensorDataType::INT32> loader( batch_size_, num_batches_ );

        EXPECT_EQ( loader.batchSize(), batch_size_ );
        EXPECT_EQ( loader.input_data_type, TensorDataType::INT32 );
        EXPECT_EQ( loader.target_data_type, TensorDataType::INT32 );
    }

    TEST_F( DataLoaderTest, MixedPrecision_FP32_INT32 )
    {
        MockDataLoader<TensorDataType::FP32, TensorDataType::INT32> loader( batch_size_, num_batches_ );

        EXPECT_EQ( loader.input_data_type, TensorDataType::FP32 );
        EXPECT_EQ( loader.target_data_type, TensorDataType::INT32 );
        EXPECT_TRUE( loader.supportsMixedPrecision() );

        loader.nextBatch();
        EXPECT_EQ( loader.inputs().shape()[0], static_cast<int64_t>(batch_size_) );
        EXPECT_EQ( loader.targets().shape()[0], static_cast<int64_t>(batch_size_) );
    }

    // ====================================================================
    // Memory Resource Tests
    // ====================================================================

    TEST_F( DataLoaderTest, CpuMemoryResource )
    {
        MockDataLoader<TensorDataType::FP32, TensorDataType::FP32, CpuMemoryResource> loader( batch_size_, num_batches_ );

        EXPECT_EQ( loader.batchSize(), batch_size_ );
        EXPECT_FALSE( loader.usesPinnedMemory() );

        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 1 );
    }

    TEST_F( DataLoaderTest, CudaPinnedMemoryResource )
    {
        MockDataLoader<TensorDataType::FP32, TensorDataType::FP32, CudaPinnedMemoryResource> loader(
            batch_size_, num_batches_, "CUDA:0" );

        EXPECT_EQ( loader.batchSize(), batch_size_ );
        EXPECT_TRUE( loader.usesPinnedMemory() );

        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 1 );
    }

    // ====================================================================
    // Type Alias Tests
    // ====================================================================

    TEST_F( DataLoaderTest, CpuDataLoader_Alias )
    {
        class TestCpuLoader : public CpuDataLoader<TensorDataType::FP32>
        {
        public:
            TestCpuLoader( size_t batch_size, size_t num_batches )
                : CpuDataLoader<TensorDataType::FP32>( batch_size ),
                num_batches_( num_batches ),
                input_( "CPU", { static_cast<int64_t>(batch_size), 10 } ),
                target_( "CPU", { static_cast<int64_t>(batch_size), 5 } )
            {
                zeros( input_ );
                zeros( target_ );
            }

            int64_t numBatches() const override
            {
                return num_batches_;
            }
            void nextBatch() override
            {
                this->incrementBatch();
            }

            InputTensor& inputs() override
            {
                return input_;
            }
            const InputTensor& inputs() const override
            {
                return input_;
            }
            TargetTensor& targets() override
            {
                return target_;
            }
            const TargetTensor& targets() const override
            {
                return target_;
            }

        private:
            size_t num_batches_;
            InputTensor input_;
            TargetTensor target_;
        };

        TestCpuLoader loader( batch_size_, num_batches_ );

        EXPECT_EQ( loader.batchSize(), batch_size_ );
        EXPECT_FALSE( loader.usesPinnedMemory() );
    }

    TEST_F( DataLoaderTest, PinnedDataLoader_Alias )
    {
        class TestPinnedLoader : public PinnedDataLoader<TensorDataType::FP32>
        {
        public:
            TestPinnedLoader( size_t batch_size, size_t num_batches )
                : PinnedDataLoader<TensorDataType::FP32>( batch_size ),
                num_batches_( num_batches ),
                input_( "CUDA:0", { static_cast<int64_t>(batch_size), 10 } ),
                target_( "CUDA:0", { static_cast<int64_t>(batch_size), 5 } )
            {
                zeros( input_ );
                zeros( target_ );
            }

            int64_t numBatches() const override
            {
                return num_batches_;
            }
            void nextBatch() override
            {
                this->incrementBatch();
            }

            InputTensor& inputs() override
            {
                return input_;
            }
            const InputTensor& inputs() const override
            {
                return input_;
            }
            TargetTensor& targets() override
            {
                return target_;
            }
            const TargetTensor& targets() const override
            {
                return target_;
            }

        private:
            size_t num_batches_;
            InputTensor input_;
            TargetTensor target_;
        };

        TestPinnedLoader loader( batch_size_, num_batches_ );

        EXPECT_EQ( loader.batchSize(), batch_size_ );
        EXPECT_TRUE( loader.usesPinnedMemory() );
    }

    // ====================================================================
    // Complete Iteration Workflow Test
    // ====================================================================

    TEST_F( DataLoaderTest, CompleteIteration_Workflow )
    {
        MockDataLoader<TensorDataType::FP32> loader( batch_size_, num_batches_ );

        size_t iteration_count = 0;

        // Simulate training loop
        while (loader.hasNext())
        {
            loader.nextBatch();

            EXPECT_TRUE( loader.validateCurrentBatch() );
            EXPECT_FALSE( loader.inputs().empty() );
            EXPECT_FALSE( loader.targets().empty() );

            ++iteration_count;
        }

        EXPECT_EQ( iteration_count, num_batches_ );
        EXPECT_EQ( loader.currentBatch(), num_batches_ );

        // Reset and verify second epoch
        loader.reset();
        EXPECT_EQ( loader.currentBatch(), 0 );
        EXPECT_TRUE( loader.hasNext() );
    }
}