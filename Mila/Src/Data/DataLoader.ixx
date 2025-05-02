module;
#include <type_traits>

export module Data.DataLoader;

import Dnn.Tensor;
import Compute.DeviceType;
import Compute.MemoryResource;
import Compute.CudaMemoryResource;
import Compute.CpuMemoryResource;

namespace Mila::Data
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
    * @brief Template for data loaders used in training and evaluation.
    *
    * @tparam TPrecision The precision type for computations.
    * @tparam TDeviceType The device type (CPU or CUDA).
    */
    export template<typename TPrecision, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
    class DataLoader {
    public:
        using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda,
            CudaMemoryResource,
            HostMemoryResource>;

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
        virtual bool nextBatch( Tensor<TPrecision, MR>& inputs, Tensor<TPrecision, MR>& targets ) = 0;

        /**
        * @brief Get the current batch index.
        * @return The current batch index.
        */
        size_t currentBatch() const { return current_batch_; }

    protected:
        size_t batch_size_;     // Size of each batch
        size_t current_batch_;  // Current batch index
    };
}