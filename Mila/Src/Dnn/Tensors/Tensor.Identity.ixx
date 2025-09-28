/**
 * @file Tensor.Id.ixx
 * @brief Tensor identification module partition
 *
 * This module partition handles all aspects of tensor identification, including:
 * - Unique identifier generation and management
 * - User-friendly naming functionality
 * - ID retrieval mechanisms
 *
 * The unique ID system ensures each tensor can be distinctly identified during
 * its lifecycle, while the naming system allows for human-readable identification
 * in debugging, visualization, and serialization contexts.
 */

module;
#include <atomic>
#include <stdexcept>
#include <string>

export module Dnn.Tensor:Identity;

export import :Interface;

import Compute.MemoryResource;
import Dnn.TensorTraits;

//namespace Mila::Dnn
//{
//    /**
//     * @brief Generator for unique tensor identifiers
//     *
//     * Provides a thread-safe mechanism for generating sequential unique identifiers
//     * for tensors throughout the application. The implementation uses atomic
//     * operations to ensure uniqueness even in multi-threaded contexts.
//     *
//     * IDs are assigned during tensor construction and remain constant throughout
//     * the tensor's lifetime, providing a reliable way to track tensor instances
//     * across operations and memory transfers.
//     */
//    class UniqueIdGenerator {
//    public:
//        /**
//         * @brief Retrieves the next available unique identifier
//         *
//         * Atomically increments and returns a counter to ensure each call
//         * receives a unique value, even in concurrent execution environments.
//         *
//         * @return size_t A guaranteed unique sequential identifier
//         */
//        static size_t getNextId() {
//            return counter_.fetch_add( 1, std::memory_order_relaxed );
//        }
//
//    private:
//        /**
//         * @brief Thread-safe counter for generating sequential identifiers
//         *
//         * Uses atomic operations to ensure correctness in multi-threaded scenarios
//         */
//        static std::atomic<size_t> counter_;
//    };
//
//    std::atomic<size_t> UniqueIdGenerator::counter_{ 0 };
//
//    /**
//     * @brief Retrieves the tensor's unique identifier
//     *
//     * The unique ID is assigned during construction and never changes during
//     * the tensor's lifetime. This provides a reliable way to identify specific
//     * tensor instances, particularly useful for:
//     * - Tracking tensors across complex operations
//     * - Debugging and logging
//     * - Graph-based execution models
//     * - Serialization/deserialization processes
//     *
//     * @return std::string The tensor's unique identifier in string format ("tensor_NNN")
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    std::string Tensor<TElementType, TMemoryResource>::getUId() const {
//        return uid_;
//    }
//
//    /**
//     * @brief Retrieves the user-assigned name of the tensor
//     *
//     * Unlike the unique ID, the name is optional and can be set by the user
//     * to provide a more meaningful, domain-specific identifier. Names are
//     * particularly useful in:
//     * - Debugging and visualization
//     * - Model checkpointing and serialization
//     * - Layer-wise operations in neural networks
//     * - Integration with external frameworks
//     *
//     * @return std::string The tensor's name (empty if not explicitly set)
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    std::string Tensor<TElementType, TMemoryResource>::getName() const {
//        return name_;
//    }
//
//    /**
//     * @brief Assigns a human-readable name to the tensor
//     *
//     * Sets a custom name that can be used for easier identification in
//     * debugging, visualization, and serialization contexts. The name is
//     * separate from the system-assigned unique ID and can be changed
//     * multiple times during the tensor's lifetime.
//     *
//     * @param value The name to assign to the tensor
//     * @throws std::invalid_argument If the provided name is empty
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    void Tensor<TElementType, TMemoryResource>::setName( const std::string& value ) {
//        if ( value.empty() ) {
//            throw std::invalid_argument( "Tensor name cannot be empty." );
//        }
//
//        name_ = value;
//    }
//
//    /**
//     * @brief Generates a unique identifier string for this tensor instance
//     *
//     * Creates a string-based unique identifier by combining a prefix with
//     * a sequentially generated numeric ID. This method is typically called
//     * during tensor construction to establish the tensor's permanent identity.
//     *
//     * The format follows the pattern "tensor_N" where N is a unique number.
//     *
//     * @return std::string A formatted unique identifier string
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    std::string Tensor<TElementType, TMemoryResource>::setUId() {
//        return "tensor_" + std::to_string( UniqueIdGenerator::getNextId() );
//    }
//
//    //template class Tensor<float, Compute::CpuMemoryResource>;
//	//template class Tensor<float, Compute::CudaMemoryResource>;
//}