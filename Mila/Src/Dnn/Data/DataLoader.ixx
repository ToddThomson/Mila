/**
 * @file DataLoader.ixx
 * @brief Abstract base class for data loaders used in training and evaluation.
 *
 * This module provides a generic data loader interface for efficiently feeding
 * data into neural network models during training and evaluation processes.
 */

module;
#include <type_traits>

export module Data.DataLoader;

import Dnn.Tensor;
import Compute.DeviceType;
import Compute.MemoryResource;
import Compute.CudaMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Data
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Abstract base class for data loaders used in training and evaluation.
     *
     * The DataLoader class provides a generic interface for loading batches of data
     * from various sources (files, databases, etc.) into tensors that can be used
     * for model training and evaluation. It supports both CPU and CUDA pinned memory
     * resources for efficient data transfer to GPU devices.
     *
     * @tparam TInput The data type for input and target tensors (must be a valid floating point type).
     * @tparam TMemoryResource The memory resource type to use (either CudaPinnedMemoryResource or CpuMemoryResource).
     */
    export template<typename TInput, typename TTarget = TInput, typename TMemoryResource = CpuMemoryResource>
        requires ValidTensorTypes<TInput, TTarget> &&
            (std::is_same_v<TMemoryResource, CudaPinnedMemoryResource> || std::is_same_v<TMemoryResource, CpuMemoryResource>)
    class DataLoader {
    public:
		/**
		* @brief Constructs a new DataLoader with the specified batch size.
		*
		* @param batch_size The number of samples to include in each batch.
		*/
		DataLoader( size_t batch_size ) : batch_size_( batch_size ), current_batch_( 0 ) {}

        /**
            * @brief Virtual destructor to ensure proper cleanup in derived classes.
            */
        virtual ~DataLoader() = default;

        /**
        * @brief Gets the total number of batches in the dataset.
        *
        * This method must be implemented by derived classes to report
        * the total number of batches available in the dataset.
        *
        * @return size_t The number of batches.
        */
        virtual size_t numBatches() const = 0;

        /**
        * @brief Gets the size of each batch.
        *
        * Returns the number of samples in each batch as specified during
        * the DataLoader construction.
        *
        * @return size_t The batch size.
        */
        size_t batchSize() const { return batch_size_; }

        /**
        * @brief Resets the loader to the beginning of the dataset.
        *
        * Calling this method resets the internal state of the data loader
        * to start from the first batch again. Derived classes may override
        * this method to implement additional reset functionality.
        */
        virtual void reset() { current_batch_ = 0; }

        /**
        * @brief Loads the next batch of data from the dataset.
        *
        * This method must be implemented by derived classes to load the next
        * batch of data into the input and target tensors. The implementation
        * should update the current_batch_ counter after successfully loading
        * a new batch.
        */
        virtual void nextBatch() = 0;

        /**
        * @brief Gets the current batch index.
        *
        * Returns the index of the batch that was most recently loaded.
        *
        * @return size_t The current batch index (0-based).
        */
        size_t currentBatch() const { return current_batch_; }

        /**
        * @brief Gets the input tensor containing the current batch of input data.
        *
        * This method must be implemented by derived classes to provide access
        * to the tensor containing input data for the current batch.
        *
        * @return Tensor<TInput, TMemoryResource>& Reference to the input tensor.
        */
        virtual Tensor<TInput, TMemoryResource>& inputs() = 0;

        /**
        * @brief Gets the input tensor containing the current batch of input data (const version).
        *
        * This method must be implemented by derived classes to provide read-only
        * access to the tensor containing input data for the current batch.
        *
        * @return const Tensor<TInput, TMemoryResource>& Const reference to the input tensor.
        */
        virtual const Tensor<TInput, TMemoryResource>& inputs() const = 0;

        /**
        * @brief Gets the target tensor containing the current batch of target data.
        *
        * This method must be implemented by derived classes to provide access
        * to the tensor containing target/label data for the current batch.
        *
        * @return Tensor<TTarget, TMemoryResource>& Reference to the target tensor.
        */
        virtual Tensor<TTarget, TMemoryResource>& targets() = 0;

        /**
        * @brief Gets the target tensor containing the current batch of target data (const version).
        *
        * This method must be implemented by derived classes to provide read-only
        * access to the tensor containing target/label data for the current batch.
        *
        * @return const Tensor<TTarget, TMemoryResource>& Const reference to the target tensor.
        */
        virtual const Tensor<TTarget, TMemoryResource>& targets() const = 0;

    protected:
        size_t batch_size_;     ///< Number of samples in each batch
        size_t current_batch_;  ///< Index of the current batch (0-based)
    };
}
