/**
 * @file DataLoader.Cuda.cpp
 * @brief Pinned-memory DataLoader cases (CUDA companion to DataLoader.cpp).
 *
 * CudaPinnedMemoryResource, the PinnedDataLoader<> alias, and usesPinnedMemory()==true
 * are CUDA-only in the Src (#ifdef MILA_HAS_CUDA), so they cannot compile under the
 * MILA_ENABLE_CUDA=OFF gate and live here instead of in DataLoader.cpp. The
 * device-agnostic iteration contract is covered once in the CPU file; this asserts
 * only the pinned-memory delta (the resource the MnistDataLoader uses on the GPU
 * training path).
 */

#include <gtest/gtest.h>
#include <cstdint>

import Mila;

namespace Mila::Tests::Dnn::Data
{
    using namespace Mila::Data;
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        template<TensorDataType TInput, TensorDataType TTarget = TInput, typename TMemoryResource = CudaPinnedMemoryResource>
        class PinnedMockLoader : public DataLoader<TInput, TTarget, TMemoryResource>
        {
        public:
            using Base = DataLoader<TInput, TTarget, TMemoryResource>;
            using InputTensor = typename Base::InputTensor;
            using TargetTensor = typename Base::TargetTensor;

            PinnedMockLoader( int64_t batch_size, int64_t num_batches )
                : Base( batch_size ),
                num_batches_( num_batches ),
                input_tensor_( Device::Cuda( 0 ), shape_t{ batch_size, 10 } ),
                target_tensor_( Device::Cuda( 0 ), shape_t{ batch_size, 5 } )
            {}

            int64_t numBatches() const override { return num_batches_; }
            void nextBatch() override { if ( this->hasNext() ) this->incrementBatch(); }
            InputTensor& inputs() override { return input_tensor_; }
            const InputTensor& inputs() const override { return input_tensor_; }
            TargetTensor& targets() override { return target_tensor_; }
            const TargetTensor& targets() const override { return target_tensor_; }

        private:
            int64_t num_batches_;
            InputTensor input_tensor_;
            TargetTensor target_tensor_;
        };
    }

    class DataLoaderCudaTests : public ::testing::Test
    {
    protected:
        static constexpr int64_t kBatchSize = 32;
        static constexpr int64_t kNumBatches = 10;
    };

    TEST_F( DataLoaderCudaTests, PinnedMemoryResource_IsPinnedAndIterates )
    {
        PinnedMockLoader<TensorDataType::FP32> loader( kBatchSize, kNumBatches );

        EXPECT_EQ( loader.batchSize(), kBatchSize );
        EXPECT_TRUE( loader.usesPinnedMemory() );

        loader.nextBatch();
        EXPECT_EQ( loader.currentBatch(), 1 );
    }

    TEST_F( DataLoaderCudaTests, PinnedDataLoaderAlias_IsPinned )
    {
        // The PinnedDataLoader<> alias resolves to DataLoader<..., CudaPinnedMemoryResource>.
        class TestPinnedLoader : public PinnedDataLoader<TensorDataType::FP32>
        {
        public:
            TestPinnedLoader( int64_t batch_size, int64_t num_batches )
                : PinnedDataLoader<TensorDataType::FP32>( batch_size ),
                num_batches_( num_batches ),
                input_( Device::Cuda( 0 ), shape_t{ batch_size, 10 } ),
                target_( Device::Cuda( 0 ), shape_t{ batch_size, 5 } )
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

        TestPinnedLoader loader( kBatchSize, kNumBatches );

        EXPECT_EQ( loader.batchSize(), kBatchSize );
        EXPECT_TRUE( loader.usesPinnedMemory() );
    }
}
