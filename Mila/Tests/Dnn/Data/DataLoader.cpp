#include <gtest/gtest.h>
#include <memory>

import Data.DataLoader;
import Dnn.Tensor;
import Compute.DeviceType;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaPinnedMemoryResource;

namespace Mila::Dnn::Data::Tests
{
    template<typename TInput, typename TTarget = TInput, typename TMemoryResource = Compute::CpuMemoryResource>
        requires ValidTensorTypes<TInput, TTarget> &&
            (std::is_same_v<TMemoryResource, Compute::CudaPinnedMemoryResource> ||
        std::is_same_v<TMemoryResource, Compute::CpuMemoryResource>)
        class MockDataLoader : public DataLoader<TInput, TTarget, TMemoryResource> {
        public:
            MockDataLoader( size_t batch_size, size_t num_batches )
                : DataLoader<TInput, TTarget, TMemoryResource>( batch_size ),
                num_batches_( num_batches ),
                input_tensor_( { batch_size, 10 } ),  // Mock dimensions for testing
                target_tensor_( { batch_size, 5 } )   // Mock dimensions for testing
            {
                // Initialize tensors with some values for testing
                for ( size_t i = 0; i < input_tensor_.size(); ++i ) {
                    input_tensor_.data()[ i ] = static_cast<TInput>( i );
                }

                for ( size_t i = 0; i < target_tensor_.size(); ++i ) {
                    target_tensor_.data()[ i ] = static_cast<TTarget>( i * 2 );
                }
            }

            size_t numBatches() const override {
                return num_batches_;
            }

            void nextBatch() override {
                if ( this->current_batch_ < num_batches_ ) {
                    // Simulate loading a batch by modifying tensor values
                    for ( size_t i = 0; i < input_tensor_.size(); ++i ) {
                        input_tensor_.data()[ i ] = static_cast<TInput>( i + this->current_batch_ );
                    }

                    for ( size_t i = 0; i < target_tensor_.size(); ++i ) {
                        target_tensor_.data()[ i ] = static_cast<TTarget>( (i + this->current_batch_) * 2 );
                    }

                    this->current_batch_++;
                }
            }

            Tensor<TInput, TMemoryResource>& inputs() override {
                return input_tensor_;
            }

            const Tensor<TInput, TMemoryResource>& inputs() const override {
                return input_tensor_;
            }

            Tensor<TTarget, TMemoryResource>& targets() override {
                return target_tensor_;
            }

            const Tensor<TTarget, TMemoryResource>& targets() const override {
                return target_tensor_;
            }

        private:
            size_t num_batches_;
            Tensor<TInput, TMemoryResource> input_tensor_;
            Tensor<TTarget, TMemoryResource> target_tensor_;
    };

    class DataLoaderTest : public testing::Test {
    protected:
        // Set up common test parameters
        const size_t kBatchSize = 32;
        const size_t kNumBatches = 10;
    };

    // Test constructors and base functionality
    TEST_F( DataLoaderTest, ConstructorAndBasicMethods ) {
        MockDataLoader<float> loader( kBatchSize, kNumBatches );

        // Test constructor results
        EXPECT_EQ( loader.batchSize(), kBatchSize );
        EXPECT_EQ( loader.numBatches(), kNumBatches );
        EXPECT_EQ( loader.currentBatch(), 0 );
    }

    // Test batch iteration
    TEST_F( DataLoaderTest, BatchIteration ) {
        MockDataLoader<float> loader( kBatchSize, kNumBatches );

        // Initial state
        EXPECT_EQ( loader.currentBatch(), 0 );

        // Move to next batch
        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 1 );

        // Move to another batch
        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 2 );
    }

    // Test reset functionality
    TEST_F( DataLoaderTest, Reset ) {
        MockDataLoader<float> loader( kBatchSize, kNumBatches );

        // Advance a few batches
        loader.nextBatch();
        loader.nextBatch();
        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 3 );

        // Reset should return to batch 0
        loader.reset();
        EXPECT_EQ( loader.currentBatch(), 0 );
    }

    // Test tensor access
    TEST_F( DataLoaderTest, TensorAccess ) {
        MockDataLoader<float> loader( kBatchSize, kNumBatches );

        // Check initial tensor dimensions
        EXPECT_EQ( loader.inputs().shape()[ 0 ], kBatchSize );
        EXPECT_EQ( loader.inputs().shape()[ 1 ], 10 );
        EXPECT_EQ( loader.targets().shape()[ 0 ], kBatchSize );
        EXPECT_EQ( loader.targets().shape()[ 1 ], 5 );

        // Check tensor data after moving to next batch
        loader.nextBatch();

        // Check that the values are updated correctly based on our mock implementation
        for ( size_t i = 0; i < loader.inputs().size(); ++i ) {
            EXPECT_FLOAT_EQ( loader.inputs().data()[ i ], static_cast<float>( i ) );
        }

        for ( size_t i = 0; i < loader.targets().size(); ++i ) {
            EXPECT_FLOAT_EQ( loader.targets().data()[ i ], static_cast<float>( i * 2 ) );
        }
    }

    // Test batch exhaustion
    TEST_F( DataLoaderTest, BatchExhaustion ) {
        MockDataLoader<float> loader( kBatchSize, kNumBatches );

        // Process all batches
        for ( size_t i = 0; i < kNumBatches; ++i ) {
            loader.nextBatch();
        }
        EXPECT_EQ( loader.currentBatch(), kNumBatches );

        // Try to move beyond available batches
        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), kNumBatches ); // Should not change
    }

    // Test different template parameters
    TEST_F( DataLoaderTest, DifferentTemplateParameters ) {
        // Test with float precision
        MockDataLoader<float> loader1( kBatchSize, kNumBatches );
        EXPECT_EQ( loader1.batchSize(), kBatchSize );

        // Test with different input and target types
        MockDataLoader<float, int> loader2( kBatchSize, kNumBatches );
        loader2.nextBatch();
        EXPECT_EQ( loader2.inputs().shape()[ 0 ], kBatchSize );
        EXPECT_EQ( loader2.targets().shape()[ 0 ], kBatchSize );
    }

    // Test with CudaPinnedMemoryResource
    TEST_F( DataLoaderTest, CudaPinnedMemory ) {
        MockDataLoader<float, float, Compute::CudaPinnedMemoryResource> loader( kBatchSize, kNumBatches );

        EXPECT_EQ( loader.batchSize(), kBatchSize );
        EXPECT_EQ( loader.numBatches(), kNumBatches );

        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 1 );
    }

    // Test const access methods
    TEST_F( DataLoaderTest, ConstAccessMethods ) {
        MockDataLoader<float> loader( kBatchSize, kNumBatches );

        const auto& const_loader = loader;

        // Test const access to tensors
        EXPECT_EQ( const_loader.inputs().shape()[ 0 ], kBatchSize );
        EXPECT_EQ( const_loader.targets().shape()[ 0 ], kBatchSize );

        // Verify const tensor access returns correct values
        loader.nextBatch();

        // Check that the values match our mock implementation
        for ( size_t i = 0; i < const_loader.inputs().size(); ++i ) {
            EXPECT_FLOAT_EQ( const_loader.inputs().data()[ i ], static_cast<float>( i ) );
        }
    }
}
