/**
 * @file DataLoader.ixx
 * @brief Device-agnostic data loader interface using abstract tensor data types
 *
 * This module provides a sophisticated data loading framework for efficiently feeding
 * heterogeneous data into neural network models during training and evaluation processes.
 * Uses abstract TensorDataType enumeration to enable seamless operation across different
 * compute devices (CPU, CUDA, Metal, OpenCL, Vulkan) without exposing device-specific
 * concrete types to host compilation.
 *
 * Key architectural features:
 * - Abstract data type system prevents device-specific compilation dependencies
 * - Support for mixed-precision training with different input/target data types
 * - Optimized memory resource selection for efficient host-device data transfers
 * - Type-safe batch operations with compile-time validation
 * - Extensible design for various data sources and formats
 */

module;
#include <type_traits>
#include <cstddef>
#include <memory>
#include <stdexcept>

export module Data.DataLoader;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorTraits;
import Compute.DeviceType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.CpuMemoryResource;
//import Compute.TensorDataTypeCompatibility;

namespace Mila::Dnn::Data
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Device-agnostic data loader interface using abstract tensor data types
     *
     * Advanced data loading framework providing efficient batch processing for neural
     * network training and evaluation across heterogeneous compute environments. Uses
     * abstract TensorDataType enumeration to enable seamless operation on different
     * devices without exposing device-specific concrete types to host compilation.
     *
     * Core architectural principles:
     * - Abstract data types prevent device-specific compilation issues
     * - Support for mixed-precision workflows with different input/target types
     * - Optimized memory resource selection for efficient data pipeline performance
     * - Type-safe operations with compile-time compatibility validation
     * - Extensible design supporting various data sources and preprocessing pipelines
     *
     * The loader supports both CPU and pinned memory resources for optimal performance
     * in GPU training scenarios, enabling efficient overlapped data transfers while
     * maintaining device independence through the abstract type system.
     *
     * @tparam TInputDataType Abstract data type for input tensors from TensorDataType enumeration
     * @tparam TTargetDataType Abstract data type for target tensors from TensorDataType enumeration
     * @tparam TMemoryResource Memory resource type determining allocation strategy and device targeting
     *
     * @note Memory resource must be either CudaPinnedMemoryResource or CpuMemoryResource for host accessibility
     * @note Input and target data types must be compatible with the specified memory resource
     * @note Derived classes must implement pure virtual methods for specific data source integration
     *
     * @see TensorDataType for supported abstract data type enumeration
     * @see TensorDataTypeTraits for compile-time data type characteristics
     * @see MemoryResource for device memory abstraction layer
     *
     * Example usage:
     * @code
     * // Mixed-precision data loader for CPU preprocessing
     * class ImageDataLoader : public DataLoader<TensorDataType::FP32, TensorDataType::INT32, CpuMemoryResource> {
     *     // Implementation for image data loading
     * };
     *
     * // High-performance data loader with pinned memory for GPU training
     * class PinnedDataLoader : public DataLoader<TensorDataType::FP16, TensorDataType::FP16, CudaPinnedMemoryResource> {
     *     // Implementation optimized for GPU transfer
     * };
     * @endcode
     */
    export template<TensorDataType TInputDataType = TensorDataType::FP32,
        TensorDataType TTargetDataType = TInputDataType,
        typename TMemoryResource = CpuMemoryResource>
        requires isValidTensorConfiguration<TInputDataType, TMemoryResource>&&
    isValidTensorConfiguration<TTargetDataType, TMemoryResource> &&
        (std::is_same_v<TMemoryResource, CudaPinnedMemoryResource> ||
            std::is_same_v<TMemoryResource, CpuMemoryResource>)
        class DataLoader {
        public:
            // ====================================================================
            // Type Aliases and Compile-Time Properties
            // ====================================================================

            using InputDataType = TensorDataType;                                      ///< Input tensor abstract data type
            using TargetDataType = TensorDataType;                                     ///< Target tensor abstract data type
            using MemoryResource = TMemoryResource;                                    ///< Memory resource type for tensor allocation
            using InputTensor = Tensor<TInputDataType, TMemoryResource>;               ///< Input tensor type alias
            using TargetTensor = Tensor<TTargetDataType, TMemoryResource>;             ///< Target tensor type alias

            static constexpr TensorDataType input_data_type = TInputDataType;          ///< Compile-time input data type constant
            static constexpr TensorDataType target_data_type = TTargetDataType;        ///< Compile-time target data type constant
            static constexpr bool is_mixed_precision = (TInputDataType != TTargetDataType); ///< Mixed-precision workflow detection
            static constexpr bool uses_pinned_memory = std::is_same_v<TMemoryResource, CudaPinnedMemoryResource>; ///< Pinned memory optimization

            // ====================================================================
            // Construction and Destruction
            // ====================================================================

            /**
             * @brief Constructs data loader with specified batch configuration
             *
             * Initializes the data loader with the specified batch size and prepares
             * the internal state for efficient batch processing. The loader is ready
             * to begin data iteration after construction.
             *
             * @param batch_size Number of samples to include in each batch
             *
             * @throws std::invalid_argument If batch_size is zero
             *
             * @note Batch size affects memory allocation and processing efficiency
             * @note Larger batches generally improve throughput but require more memory
             * @note Consider GPU memory constraints when selecting batch size for device training
             */
            explicit DataLoader( size_t batch_size )
                : batch_size_( batch_size ), current_batch_( 0 ) {
                if ( batch_size == 0 ) {
                    throw std::invalid_argument( "Batch size must be greater than zero" );
                }
            }

            /**
             * @brief Virtual destructor ensuring proper cleanup in derived classes
             *
             * Provides proper resource cleanup for polymorphic destruction,
             * enabling safe use of base class pointers to derived instances.
             */
            virtual ~DataLoader() = default;

            /**
             * @brief Copy operations explicitly deleted for performance safety
             *
             * Prevents accidental expensive copy operations involving large datasets
             * and complex internal state management.
             */
            DataLoader( const DataLoader& ) = delete;
            DataLoader& operator=( const DataLoader& ) = delete;

            /**
             * @brief Move operations for efficient ownership transfer
             *
             * Enables efficient transfer of data loader instances without
             * copying internal state or dataset references.
             */
            DataLoader( DataLoader&& ) = default;
            DataLoader& operator=( DataLoader&& ) = default;

            // ====================================================================
            // Dataset Properties and Introspection
            // ====================================================================

            /**
             * @brief Returns the total number of batches in the dataset
             *
             * Derived classes must implement this method to report the total number
             * of batches available in their specific dataset. This information is
             * essential for training loop progress tracking and epoch management.
             *
             * @return Total number of batches available in the dataset
             *
             * @note Implementation should account for partial batches at dataset end
             * @note Value may change if dataset is modified or resampled
             * @note Used for training progress reporting and epoch boundary detection
             */
            virtual size_t numBatches() const = 0;

            /**
             * @brief Returns the configured batch size
             *
             * Provides the number of samples included in each batch as specified
             * during data loader construction. This value remains constant throughout
             * the loader's lifetime.
             *
             * @return Number of samples in each batch
             *
             * @note Final batch may contain fewer samples if dataset size is not divisible by batch size
             * @note Batch size affects memory requirements and processing efficiency
             */
            size_t batchSize() const noexcept {
                return batch_size_;
            }

            /**
             * @brief Returns the current batch index
             *
             * Provides the zero-based index of the batch that was most recently
             * loaded through nextBatch(). Useful for progress tracking and
             * debugging data loading workflows.
             *
             * @return Zero-based index of current batch
             *
             * @note Returns 0 before first call to nextBatch()
             * @note Index increments with each successful nextBatch() call
             * @note Reset to 0 when reset() method is called
             */
            size_t currentBatch() const noexcept {
                return current_batch_;
            }

            /**
             * @brief Checks if more batches are available
             *
             * Determines whether additional batches can be loaded from the dataset,
             * enabling efficient iteration control in training and evaluation loops.
             *
             * @return true if more batches are available, false if dataset is exhausted
             *
             * @note Implementation should consider current position and total dataset size
             * @note Used to determine when to reset or stop iteration
             */
            virtual bool hasNext() const {
                return current_batch_ < numBatches();
            }

            /**
             * @brief Checks if data loader supports mixed-precision workflows
             *
             * Compile-time detection of whether the loader uses different data types
             * for inputs and targets, enabling mixed-precision training optimizations.
             *
             * @return true if input and target use different data types, false otherwise
             */
            static constexpr bool supportsMixedPrecision() noexcept {
                return is_mixed_precision;
            }

            /**
             * @brief Checks if data loader uses pinned memory for GPU optimization
             *
             * Compile-time detection of pinned memory usage, indicating optimization
             * for efficient host-to-device memory transfers in GPU training scenarios.
             *
             * @return true if using pinned memory, false for standard CPU memory
             */
            static constexpr bool usesPinnedMemory() noexcept {
                return uses_pinned_memory;
            }

            // ====================================================================
            // Data Loading Operations
            // ====================================================================

            /**
             * @brief Resets the loader to the beginning of the dataset
             *
             * Resets the internal state to start iteration from the first batch.
             * Derived classes may override this method to implement additional
             * reset functionality such as dataset reshuffling or preprocessing
             * pipeline reinitialization.
             *
             * @note Base implementation resets batch counter to zero
             * @note Called automatically at epoch boundaries in training loops
             * @note Override to implement custom reset behavior (shuffling, etc.)
             */
            virtual void reset() {
                current_batch_ = 0;
            }

            /**
             * @brief Loads the next batch of data from the dataset
             *
             * Derived classes must implement this method to load the next batch
             * of data into the input and target tensors. Implementation should
             * handle data preprocessing, memory allocation, and batch composition
             * according to the specific dataset requirements.
             *
             * @throws std::runtime_error If no more batches are available
             * @throws std::runtime_error If data loading fails
             *
             * @note Implementation must update current_batch_ counter after successful load
             * @note Should handle end-of-dataset conditions appropriately
             * @note May involve complex preprocessing pipelines and data augmentation
             */
            virtual void nextBatch() = 0;

            // ====================================================================
            // Tensor Access Methods
            // ====================================================================

            /**
             * @brief Provides mutable access to input tensor for current batch
             *
             * Derived classes must implement this method to provide access to the
             * tensor containing input data for the currently loaded batch. The tensor
             * should be properly shaped and contain valid data after nextBatch() call.
             *
             * @return Mutable reference to input tensor containing current batch data
             *
             * @note Tensor shape should match expected input dimensions for the model
             * @note Data should be preprocessed and ready for model consumption
             * @note Memory layout should be optimized for target compute device
             */
            virtual InputTensor& inputs() = 0;

            /**
             * @brief Provides immutable access to input tensor for current batch
             *
             * Derived classes must implement this method to provide read-only access
             * to the tensor containing input data for the currently loaded batch.
             *
             * @return Const reference to input tensor containing current batch data
             *
             * @note Enables safe access for analysis and debugging without modification risk
             * @note Should return same data as mutable version
             */
            virtual const InputTensor& inputs() const = 0;

            /**
             * @brief Provides mutable access to target tensor for current batch
             *
             * Derived classes must implement this method to provide access to the
             * tensor containing target/label data for the currently loaded batch.
             * The tensor should contain ground truth data corresponding to the inputs.
             *
             * @return Mutable reference to target tensor containing current batch labels
             *
             * @note Target data should align with input batch ordering
             * @note Data format should match model's expected output structure
             * @note For mixed-precision workflows, may use different data type than inputs
             */
            virtual TargetTensor& targets() = 0;

            /**
             * @brief Provides immutable access to target tensor for current batch
             *
             * Derived classes must implement this method to provide read-only access
             * to the tensor containing target/label data for the currently loaded batch.
             *
             * @return Const reference to target tensor containing current batch labels
             *
             * @note Enables safe access for analysis and debugging without modification risk
             * @note Should return same data as mutable version
             */
            virtual const TargetTensor& targets() const = 0;

            // ====================================================================
            // Advanced Data Loading Features
            // ====================================================================

            /**
             * @brief Returns dataset statistics for optimization and analysis
             *
             * Derived classes may override this method to provide dataset-specific
             * statistics such as sample count, class distribution, or data characteristics
             * that can inform training optimization and analysis.
             *
             * @return String containing human-readable dataset statistics
             *
             * @note Default implementation provides basic batch configuration information
             * @note Override to include dataset-specific metrics and characteristics
             */
            virtual std::string getDatasetInfo() const {
                std::ostringstream oss;
                oss << "DataLoader: " << numBatches() << " batches, "
                    << batch_size_ << " samples per batch"
                    << (is_mixed_precision ? " (mixed precision)" : "")
                    << (uses_pinned_memory ? " (pinned memory)" : "");
                return oss.str();
            }

            /**
             * @brief Validates current batch data integrity
             *
             * Derived classes may override this method to implement data validation
             * checks ensuring batch integrity, proper tensor shapes, and valid data ranges.
             * Useful for debugging data loading pipelines and preprocessing issues.
             *
             * @return true if current batch data passes validation, false otherwise
             *
             * @note Default implementation performs basic existence checks
             * @note Override to implement dataset-specific validation logic
             * @note Can be used in debug builds for comprehensive data verification
             */
            virtual bool validateCurrentBatch() const {
                try {
                    const auto& input_tensor = inputs();
                    const auto& target_tensor = targets();

                    return !input_tensor.empty() && !target_tensor.empty() &&
                        input_tensor.size() > 0 && target_tensor.size() > 0;
                }
                catch ( ... ) {
                    return false;
                }
            }

        protected:
            // ====================================================================
            // Protected Helper Methods for Derived Classes
            // ====================================================================

            /**
             * @brief Updates current batch counter
             *
             * Protected helper method for derived classes to update the batch
             * counter after successfully loading a new batch. Ensures consistent
             * state management across all data loader implementations.
             *
             * @param batch_index New batch index to set
             *
             * @note Should be called by derived classes after successful batch loading
             * @note Enables consistent progress tracking across all loader types
             */
            void setCurrentBatch( size_t batch_index ) noexcept {
                current_batch_ = batch_index;
            }

            /**
             * @brief Increments current batch counter
             *
             * Protected helper method for derived classes to increment the batch
             * counter after successfully loading the next batch. Simplifies
             * sequential batch loading implementations.
             *
             * @note Should be called by derived classes after successful nextBatch() operation
             * @note Automatically handles sequential batch progression
             */
            void incrementBatch() noexcept {
                ++current_batch_;
            }

        private:
            // ====================================================================
            // Private Member Variables
            // ====================================================================

            size_t batch_size_;     ///< Number of samples in each batch (immutable after construction)
            size_t current_batch_;  ///< Zero-based index of currently loaded batch
    };

    // ====================================================================
    // Type Aliases for Common Data Loader Configurations
    // ====================================================================

    /**
     * @brief CPU data loader with single precision floating point
     *
     * Convenient alias for data loaders using standard CPU memory with FP32
     * data types for both inputs and targets. Suitable for CPU-only training
     * and development workflows.
     *
     * @tparam TInputDataType Input tensor data type (defaults to FP32)
     * @tparam TTargetDataType Target tensor data type (defaults to input type)
     */
    export template<TensorDataType TInputDataType = TensorDataType::FP32,
        TensorDataType TTargetDataType = TInputDataType>
        using CpuDataLoader = DataLoader<TInputDataType, TTargetDataType, CpuMemoryResource>;

    /**
     * @brief Pinned memory data loader optimized for GPU training
     *
     * Convenient alias for data loaders using CUDA pinned memory for optimal
     * host-to-device transfer performance. Essential for high-performance
     * GPU training workflows with overlapped data transfers.
     *
     * @tparam TInputDataType Input tensor data type (defaults to FP32)
     * @tparam TTargetDataType Target tensor data type (defaults to input type)
     */
    export template<TensorDataType TInputDataType = TensorDataType::FP32,
        TensorDataType TTargetDataType = TInputDataType>
        using PinnedDataLoader = DataLoader<TInputDataType, TTargetDataType, CudaPinnedMemoryResource>;

    /**
     * @brief Mixed-precision data loader for advanced training workflows
     *
     * Convenient alias for data loaders using different precision for inputs
     * and targets, enabling advanced mixed-precision training strategies.
     */
    export template<TensorDataType TInputDataType, TensorDataType TTargetDataType>
        using MixedPrecisionDataLoader = DataLoader<TInputDataType, TTargetDataType, CudaPinnedMemoryResource>;
}