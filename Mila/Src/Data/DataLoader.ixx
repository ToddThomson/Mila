module;
#include <type_traits>

export module Data.DataLoader;

import Dnn.Tensor;
import Compute.DeviceType;
import Compute.MemoryResource;
import Compute.CudaMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.CpuMemoryResource;

namespace Mila::Data
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
    * @brief Template for data loaders used in training and evaluation.
    *
    * @tparam TInput The precision type for computations.
    * @tparam TDeviceType The device type (CPU or CUDA).
    */
    export template<typename TInput, typename TMemoryResource>
        requires ValidFloatTensorType<TInput> &&
            (std::is_same_v<TMemoryResource, CudaPinnedMemoryResource> || std::is_same_v<TMemoryResource, CpuMemoryResource>)
    class DataLoader {
    public:

        DataLoader( size_t batch_size ) : batch_size_( batch_size ), current_batch_( 0 ) {}

        virtual ~DataLoader() = default;

        /**
        * @brief Get the number of batches in the dataset.
        * @return The number of batches.
        */
        virtual size_t numBatches() const = 0;

        /**
        * @brief Get the size of each batch.
        * @return The batch size.
        */
        size_t batchSize() const { return batch_size_; }

        /**
        * @brief Reset the loader to the beginning of the dataset.
        */
        virtual void reset() { current_batch_ = 0; }

        /**
        * @brief Get the next batch of data.
        * @param inputs The tensor to store input data.
        * @param targets The tensor to store target data.
        * @return True if there was data to load, false if end of dataset.
        */
        virtual void nextBatch() = 0;

        /**
        * @brief Get the current batch index.
        * @return The current batch index.
        */
        size_t currentBatch() const { return current_batch_; }

        /**
        * @brief Get the input tensor.
        * @return Reference to the input tensor.
        */
        virtual Tensor<TInput, TMemoryResource>& inputs() = 0;

        /**
        * @brief Get the input tensor (const version).
        * @return Const reference to the input tensor.
        */
        virtual const Tensor<TInput, TMemoryResource>& inputs() const = 0;

        /**
        * @brief Get the target tensor.
        * @return Reference to the target tensor.
        */
        virtual Tensor<TInput, TMemoryResource>& targets() = 0;

        /**
        * @brief Get the target tensor (const version).
        * @return Const reference to the target tensor.
        */
        virtual const Tensor<TInput, TMemoryResource>& targets() const = 0;

    protected:
        size_t batch_size_;     // Size of each batch
        size_t current_batch_;  // Current batch index
    };
}
