/**
 * @file Tensor.ixx
 * @brief Tensor implementation
 *
 * This module defines the Tensor class which provides a unified interface for multi-dimensional
 * tensors across different compute devices (CPU, CUDA, Metal, OpenCL, Vulkan) with explicit
 * device context management. The tensor uses abstract TensorDataType enumeration for type safety
 * while enabling compile-time optimization and device-specific implementations.
 *
 * Key architectural features:
 * - Device context integration for proper multi-GPU support
 * - Abstract data type system prevents host compilation issues with device types
 * - Compile-time dispatch based on TensorDataType enumeration
 * - Device-agnostic memory resource abstraction with device binding
 * - Support for packed sub-byte precision formats (FP4, INT4)
 * - Type-safe memory transfer operations with automatic compatibility validation
 */

module;
#include <utility>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <atomic>
#include <functional>
#include <numeric>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <limits>

export module Dnn.Tensor;

import Dnn.TensorBuffer;
import Dnn.TensorData;
import Dnn.TensorDataType;
import Dnn.TensorPtr;
import Dnn.TensorTraits;
//import Compute.BackendTraitSelector;
import Compute.CpuTensorDataTypeTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Compute.CudaManagedMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.DeviceContext;
import Compute.CpuDeviceContext;
import Compute.CudaDeviceContext;
import Compute.DeviceRegistrar;

namespace Mila::Dnn
{
    namespace detail
    {
        /**
         * @brief Calculates storage size in bytes for given logical element count
         *
         * Computes the required storage bytes for a given number of logical elements
         * of the specified tensor data type, handling potential overflow conditions.
         *
         * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
         * @param logical_size Number of logical elements
         * @return Required storage size in bytes
         * @throws std::overflow_error If calculation would overflow
         */
        template<TensorDataType TDataType>
        constexpr size_t getStorageSize(size_t logical_size) {
            constexpr size_t element_size = TensorDataTypeTraits<TDataType>::size_in_bytes;

            if (logical_size > std::numeric_limits<size_t>::max() / element_size) {
                throw std::overflow_error("Storage size calculation would overflow.");
            }

            return logical_size * element_size;
        }
    }

    /**
     * @brief Thread-safe generator for unique tensor identifiers
     *
     * Provides atomic generation of unique IDs for tensor instances ensuring
     * proper identification and debugging capabilities across the system.
     * Thread-safe design supports concurrent tensor creation in multi-threaded
     * neural network training environments.
     */
    class UniqueIdGenerator {
    public:
        /**
         * @brief Generates the next unique identifier atomically
         *
         * Uses relaxed memory ordering for maximum performance while maintaining
         * uniqueness guarantees across all threads and tensor instances.
         *
         * @return Unique size_t identifier that is thread-safe and monotonically increasing
         */
        static size_t getNextId() {
            return counter_.fetch_add(1, std::memory_order_relaxed);
        }

    private:
        static std::atomic<size_t> counter_; ///< Thread-safe counter for unique ID generation
    };

    std::atomic<size_t> UniqueIdGenerator::counter_{ 0 };

    /**
     * @brief Device-aware N-dimensional tensor with DeviceContext integration
     *
     * Advanced tensor implementation providing unified operations across heterogeneous
     * compute environments (CPU, CUDA, Metal, OpenCL, Vulkan) with explicit device context
     * management for proper multi-GPU support. Uses abstract TensorDataType enumeration for
     * type safety while enabling compile-time optimization and device-specific dispatch.
     *
     * Core architectural principles:
     * - Device context integration enables proper multi-GPU scaling and resource management
     * - Abstract data types prevent host compilation issues with device-only types
     * - Compile-time dispatch optimizes performance across all supported devices
     * - Memory resource abstraction with device binding enables seamless device interoperability
     * - Support for cutting-edge precision formats including FP8, FP4, and packed types
     * - Type-safe operations with automatic memory compatibility validation
     *
     * The tensor class is move-only to prevent accidental expensive copy operations.
     * Explicit operations are provided for data sharing, deep copying, and memory transfers.
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
     * @tparam TMemoryResource Memory resource type defining storage location and access patterns
     *
     * @note All constructors require explicit device context for proper device binding
     * @note Use clone() for deep copies and explicit transfer methods for device migration
     * @note Memory transfers preserve tensor metadata including name and dimensional structure
     * @note Packed data types (FP4, INT4) automatically handle sub-byte storage optimization
     * @note Memory resource management is handled by TensorBuffer - Tensor focuses on tensor semantics
     *
     * @see TensorDataType for supported abstract data type enumeration
     * @see TensorDataTypeTraits for compile-time type characteristics
     * @see ITensorData for the polymorphic base interface
     * @see TensorBuffer for underlying device-specific memory management
     * @see MemoryResource for device memory abstraction layer
     * @see DeviceContext for device binding and resource management
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    class Tensor : public ITensorData
    {
    public:
        // ====================================================================
        // Type Aliases and Compile-Time Properties
        // ====================================================================

        using DataType = TensorDataType;                                           ///< Abstract data type enumeration
        using MemoryResource = TMemoryResource;                                    ///< Memory resource type for this tensor
        using DataTypeTraits = TensorDataTypeTraits<TDataType>;                   ///< Compile-time data type characteristics

        static constexpr TensorDataType data_type = TDataType;                     ///< Compile-time data type constant
        static constexpr size_t element_size = DataTypeTraits::size_in_bytes;      ///< Size per element in bytes
        static constexpr size_t alignment = DataTypeTraits::alignment;             ///< Required memory alignment
        static constexpr bool is_float_type = DataTypeTraits::is_float_type;       ///< Floating-point type classification
        static constexpr bool is_integer_type = DataTypeTraits::is_integer_type;   ///< Integer type classification
        static constexpr bool is_device_only = DataTypeTraits::is_device_only;     ///< Device-only type restriction

        // ====================================================================
        // Construction, Assignment, and Destruction
        // ====================================================================

        /**
         * @brief Creates a tensor with device context and specified shape
         *
         * Constructs a tensor with the given device context and dimensions, allocating appropriate
         * memory using the configured memory resource. The device context ensures proper device
         * binding and resource management for multi-GPU environments.
         *
         * @param device_context Shared pointer to device context for proper device binding
         * @param shape Vector defining the size of each dimension in row-major order
         *
         * @throws std::invalid_argument If device_context is null
         * @throws std::bad_alloc If memory allocation fails
         * @throws std::runtime_error If device context type doesn't match memory resource requirements
         *
         * @note Shape vector defines tensor dimensionality from outermost to innermost
         * @note Memory layout is always row-major (C-style) for maximum compatibility
         * @note Packed data types automatically calculate appropriate storage requirements
         * @note Empty shape vector creates a scalar (0-dimensional) tensor
         */
        explicit Tensor(std::shared_ptr<Compute::DeviceContext> device_context, const std::vector<size_t>& shape)
            : device_context_(device_context), uid_(setUId()), shape_(shape),
            strides_(computeStrides(shape)), size_(computeSize(shape)) {

            if (!device_context_) {
                throw std::invalid_argument("Device context cannot be null");
            }

            validateDeviceContextCompatibility();

            allocateBuffer();
        }

        /**
         * @brief Creates a tensor with device name string and specified shape
         *
         * Constructs a tensor by creating a new DeviceContext internally using the provided
         * device name. This constructor initializes the device registrar to ensure devices
         * are available and creates appropriate device context based on device type.
         *
         * @param device_name Device identifier string (e.g., "CPU", "CUDA:0")
         * @param shape Vector defining the size of each dimension in row-major order
         *
         * @throws std::invalid_argument If device name is invalid or shape is invalid
         * @throws std::runtime_error If device initialization fails or device type doesn't match memory resource
         * @throws std::bad_alloc If memory allocation fails
         *
         * @note Creates new DeviceContext internally - use shared context constructor for shared contexts
         * @note Automatically initializes device registrar to ensure device availability
         */
        explicit Tensor( const std::string& device_name, const std::vector<size_t>& shape )
            : Tensor( createDeviceContext( device_name ), shape ) {
        }

        /**
         * @brief Creates a tensor using externally managed memory buffer with device context
         *
         * Constructs a tensor that wraps existing memory without taking ownership.
         * The external memory must remain valid for the tensor's entire lifetime.
         * Essential for interfacing with external libraries, memory-mapped data,
         * or implementing custom memory management strategies.
         *
         * @param device_context Shared pointer to device context for proper device binding
         * @param shape Vector defining tensor dimensions in row-major order
         * @param data_ptr Shared pointer to pre-allocated memory of appropriate size
         *
         * @throws std::invalid_argument If device_context or data_ptr is null, or shape is invalid
         * @throws std::bad_alloc If wrapper allocation fails
         * @throws std::runtime_error If device context type doesn't match memory resource requirements
         *
         * @warning Caller must ensure memory size matches computed tensor requirements
         * @warning Memory layout must be row-major compatible with computed strides
         * @warning External memory must remain valid for tensor lifetime
         *
         * @note Tensor does not manage the underlying memory lifecycle
         * @note Memory size requirements include packing considerations for sub-byte types
         * @note Alignment requirements must be satisfied by external memory
         */
        Tensor(std::shared_ptr<Compute::DeviceContext> device_context, const std::vector<size_t>& shape, std::shared_ptr<void> data_ptr)
            : device_context_(device_context), uid_(setUId()), shape_(shape),
            strides_(computeStrides(shape)), size_(computeSize(shape)), external_memory_ptr_(data_ptr) {

            if (!device_context_) {
                throw std::invalid_argument("Device context cannot be null");
            }
            if (!external_memory_ptr_) {
                throw std::invalid_argument("data_ptr cannot be null");
            }

            validateDeviceContextCompatibility();

            size_t required_bytes = detail::getStorageSize<TDataType>(size_);
            
            // Create TensorBuffer with external memory using updated constructor signature
            buffer_ = std::make_shared<TensorBuffer<TDataType, TMemoryResource>>(
                device_context_, size_, static_cast<std::byte*>(external_memory_ptr_.get()), required_bytes);
        }

        /**
         * @brief Copy constructor - explicitly deleted for performance safety
         *
         * Tensors cannot be copied implicitly to prevent accidental expensive copy
         * operations in performance-critical neural network code. Use explicit
         * clone() for deep copies or std::move() for efficient ownership transfers.
         */
        Tensor(const Tensor& other) = delete;

        /**
         * @brief Efficiently transfers ownership from another tensor
         *
         * Moves all resources from the source tensor, leaving it in a valid
         * but empty state. No data copying occurs, making this operation
         * extremely efficient for large tensors.
         *
         * @param other Source tensor to move from (will be left in empty state)
         */
        Tensor(Tensor&& other) noexcept
            : device_context_(std::move(other.device_context_)),
            uid_(std::move(other.uid_)),
            name_(std::move(other.name_)),
            size_(other.size_),
            shape_(std::move(other.shape_)),
            strides_(std::move(other.strides_)),
            buffer_(std::move(other.buffer_)),
            external_memory_ptr_(std::move(other.external_memory_ptr_)) {
            other.size_ = 0;
            other.shape_.clear();
            other.strides_.clear();
        }

        /**
         * @brief Copy assignment operator - explicitly deleted for performance safety
         *
         * Tensors cannot be copy-assigned to prevent accidental expensive copy
         * operations. Use explicit clone() for deep copies or std::move() for transfers.
         */
        Tensor& operator=(const Tensor& other) = delete;

        /**
         * @brief Efficiently moves resources from another tensor
         *
         * Transfers ownership without data copying, leaving the source tensor
         * in a valid but empty state. Self-assignment safe implementation.
         *
         * @param other Source tensor to move from
         * @return Reference to this tensor for method chaining
         *
         * @note Self-assignment safe through identity check
         * @note Source tensor becomes empty after successful move
         * @note All metadata including name, shape, and device context are transferred
         */
        Tensor& operator=(Tensor&& other) noexcept {
            if (this != &other) {
                device_context_ = std::move(other.device_context_);
                uid_ = std::move(other.uid_);
                name_ = std::move(other.name_);
                shape_ = std::move(other.shape_);
                strides_ = std::move(other.strides_);
                size_ = other.size_;
                buffer_ = std::move(other.buffer_);
                external_memory_ptr_ = std::move(other.external_memory_ptr_);

                other.size_ = 0;
                other.shape_.clear();
                other.strides_.clear();
            }
            return *this;
        }

        /**
         * @brief Destructor with automatic resource cleanup via RAII
         *
         * Automatically releases all resources through RAII principles.
         * Buffer cleanup is handled by shared_ptr reference counting,
         * ensuring proper cleanup even in complex sharing scenarios.
         */
        ~Tensor() = default;

        // ====================================================================
        // Device Context and Memory Resource Access
        // ====================================================================

		// REVIEW: Should we allow changing the device context after creation? return const ref?
        /**
         * @brief Returns the tensor's device context
         *
         * Provides access to the device context for device operations, stream management,
         * and multi-GPU coordination.
         *
         * @return Shared pointer to the device context
         */
        std::shared_ptr<Compute::DeviceContext> getDeviceContext() const {
            return device_context_;
        }

        // ====================================================================
        // Type Information and Interface Compliance
        // ====================================================================

        /**
         * @brief Returns the tensor's abstract data type identifier
         *
         * Provides the standardized data type enumeration enabling runtime type
         * identification and compatibility checking with external systems without
         * exposing device-specific concrete types.
         *
         * @return TensorDataType enumeration value for this tensor
         *
         * @note Required by ITensorData polymorphic interface
         * @note Determined at compile time from TDataType template parameter
         * @note Enables type-safe operations across device boundaries
         */
        TensorDataType getDataType() const override {
            return TDataType;
        }

        /**
         * @brief Returns human-readable name of the tensor's data type
         *
         * Provides a string representation of the abstract element type for
         * debugging, logging, and user interface display purposes.
         *
         * @return String name of the data type (e.g., "FP32", "FP16", "INT32")
         *
         * @note Required by ITensorData polymorphic interface
         * @note Useful for debugging and runtime introspection
         * @note Consistent across all devices for same abstract type
         */
        std::string getDataTypeName() const override {
            return std::string(DataTypeTraits::type_name);
        }

        /**
         * @brief Returns the size in bytes of each tensor element
         *
         * Provides the storage size per element, accounting for packed data types
         * where multiple elements may share bytes. Essential for memory allocation
         * and transfer calculations.
         *
         * @return Size in bytes per logical element
         *
         * @note For packed types, returns storage size per packed unit
         * @note Use getStorageSize() for total memory requirements
         */
        size_t getElementSizeInBytes() const {
            return DataTypeTraits::size_in_bytes;
        }

        /**
         * @brief Checks if tensor data is accessible from host (CPU) code
         *
         * Determines whether the tensor's memory can be directly accessed
         * from host code without explicit transfers. Essential for choosing
         * appropriate access patterns and optimization strategies.
         *
         * @return true if memory is host-accessible, false otherwise
         *
         * @note Compile-time constant based on memory resource type
         * @note Affects which access methods are available at compile time
         * @note Host-accessible memory enables direct pointer access
         */
        static constexpr bool is_host_accessible() {
            return TMemoryResource::is_host_accessible;
        }

        /**
         * @brief Checks if tensor data is accessible from device (GPU) code
         *
         * Determines whether the tensor's memory can be accessed from device
         * kernels and compute operations. Critical for kernel launch decisions
         * and device operation planning.
         *
         * @return true if memory is device-accessible, false otherwise
         *
         * @note Compile-time constant based on memory resource type
         * @note Some memory types (managed, pinned) are accessible from both host and device
         * @note Device-accessible memory enables kernel operations
         */
        static constexpr bool is_device_accessible() {
            return TMemoryResource::is_device_accessible;
        }

        // =================================================================
        // Element access (host accessible MR )
        // =================================================================

        // Returns a value at indices (host-only), enabled for host-compatible types
        /*template<typename TElement = int>
            requires TMemoryResource::is_host_accessible && (Compute::BackendTraits<TMemoryResource>::template supports<TDataType>())
        auto at(const std::vector<size_t>& indices) const
            -> typename Compute::BackendTraits<TMemoryResource>::template native_type<TDataType>
        {
            validateIndices(indices, "at()");

            using BackendTraits = Compute::BackendTraits<TMemoryResource>;
            using NativeT = typename BackendTraits::template native_type<TDataType>;
            const auto idx = computeFlatIndex(indices);
            const auto* base = static_cast<const NativeT*>(rawData());
            return base[idx];
        }*/

        //// Sets a value at indices (host-only), enabled for host-compatible types
        //template<typename TDummy = int>
        //    requires TMemoryResource::is_host_accessible && (Compute::BackendTraits<TMemoryResource>::template supports<TDataType>())
        //void set(const std::vector<size_t>& indices,
        //    typename Compute::BackendTraits<TMemoryResource>::template native_type<TDataType> value)
        //{
        //    validateIndices(indices, "set()");

        //    using BackendTraits = Compute::BackendTraits<TMemoryResource>;
        //    using NativeT = typename BackendTraits::template native_type<TDataType>;
        //    const auto idx = computeFlatIndex(indices);
        //    auto* base = static_cast<NativeT*>(rawData());
        //    base[idx] = value;
        //}

        // ====================================================================
        // Tensor Properties and Introspection
        // ====================================================================

        /**
         * @brief Returns the tensor's dimensional shape vector
         *
         * Provides the size of each dimension in row-major order. The shape
         * completely defines the tensor's dimensional structure and determines
         * memory layout through computed strides.
         *
         * @return Const reference to vector containing dimension sizes
         *
         * @note Required by ITensorData polymorphic interface
         * @note Empty vector indicates a scalar (0-dimensional) tensor
         * @note Order is from outermost to innermost dimension (row-major)
         * @note Shape determines stride computation and memory indexing
         */
        const std::vector<size_t>& shape() const override {
            return shape_;
        }

        /**
         * @brief Returns the tensor's memory stride information
         *
         * Provides the number of elements to skip when moving one position
         * along each dimension. Essential for multi-dimensional indexing
         * and understanding memory layout patterns.
         *
         * @return Const reference to vector containing stride values
         *
         * @note Strides are computed for row-major (C-style) layout
         * @note Used internally for efficient index computation
         * @note Length matches shape vector length
         * @note Enables efficient multi-dimensional memory access
         */
        const std::vector<size_t>& strides() const {
            return strides_;
        }

        /**
         * @brief Returns the total number of logical elements in the tensor
         *
         * Provides the product of all dimensions, representing the total
         * logical element count regardless of shape configuration or
         * underlying packed storage.
         *
         * @return Total number of logical elements
         *
         * @note Zero for empty tensors
         * @note Product of all shape dimensions
         * @note Logical count may differ from storage bytes for packed types
         */
        size_t size() const {
            return size_;
        }

        /**
         * @brief Checks if the tensor contains no elements
         *
         * Determines whether the tensor has been allocated with zero size,
         * useful for validation and conditional processing in neural network
         * operations.
         *
         * @return true if tensor has no elements, false otherwise
         *
         * @note Equivalent to size() == 0
         * @note Empty tensors can still have a defined shape
         * @note Empty tensors require no storage allocation
         */
        bool empty() const {
            return (size_ == 0);
        }

        /**
         * @brief Returns the number of dimensions in the tensor
         *
         * Provides the tensor's dimensionality, ranging from 0 (scalar)
         * to arbitrarily high dimensions. Essential for understanding
         * tensor structure in neural network operations.
         *
         * @return Number of dimensions
         *
         * @note Equivalent to shape().size()
         * @note Zero indicates a scalar tensor
         * @note Common neural network tensors are 2D (matrices) to 4D (batch, channel, height, width)
         */
        size_t rank() const {
            return shape_.size();
        }

        // ====================================================================
        // Data Access and Raw Pointers
        // ====================================================================

        /**
         * @brief Returns untyped mutable pointer to tensor data
         *
         * Provides raw memory access for low-level operations, device kernel
         * interfacing, and external library integration. Required by ITensorData
         * polymorphic interface for type-erased operations.
         *
         * @return Void pointer to tensor data buffer
         *
         * @warning Requires manual type casting and size calculations
         * @warning No type or bounds safety - caller responsibility
         * @warning For packed types, handle sub-byte element access carefully
         *
         * @note Required by ITensorData polymorphic interface
         * @note Use with appropriate type casting based on getDataType()
         * @note Consider memory access patterns for packed data types
         */
        void* rawData() override {
            return buffer_->rawData();
        }

        /**
         * @brief Returns untyped immutable pointer to tensor data
         *
         * Provides read-only raw memory access for low-level operations
         * and external library integration with const-correctness.
         *
         * @return Const void pointer to tensor data buffer
         *
         * @warning Requires manual type casting and size calculations
         * @warning No type or bounds safety - caller responsibility
         * @warning For packed types, handle sub-byte element access carefully
         *
         * @note Required by ITensorData polymorphic interface
         * @note Use with appropriate type casting based on getDataType()
         * @note Consider memory access patterns for packed data types
         */
        const void* rawData() const override {
            return buffer_->rawData();
        }

        /**
         * @brief Returns memory-type-aware smart pointer to tensor data
         *
         * Provides a type-safe pointer wrapper that enforces memory
         * accessibility rules at compile time, preventing unsafe direct
         * access to device memory from host code.
         *
         * @return HostPtr or DevicePtr depending on memory resource type
         *
         * @note Compile-time selection based on memory accessibility
         * @note Provides additional safety over raw pointers
         * @note Preferred for generic code handling multiple memory types
         * @note Automatically prevents host access to device-only memory
         */
        auto dataPtr() {
            if constexpr (is_host_accessible()) {
                return HostPtr<void>(buffer_->rawData());
            }
            else {
                return DevicePtr<void>(buffer_->rawData());
            }
        }

        /**
         * @brief Returns immutable memory-type-aware smart pointer to tensor data
         *
         * Provides a read-only type-safe pointer wrapper that enforces memory
         * accessibility rules at compile time with const-correctness.
         *
         * @return Const HostPtr or DevicePtr depending on memory resource type
         *
         * @note Compile-time selection based on memory accessibility
         * @note Provides additional safety over raw pointers
         * @note Preferred for generic code handling multiple memory types
         * @note Automatically prevents host access to device-only memory
         */
        const auto dataPtr() const {
            if constexpr (is_host_accessible()) {
                return HostPtr<const void>(buffer_->rawData());
            }
            else {
                return DevicePtr<const void>(buffer_->rawData());
            }
        }

        // ====================================================================
        // Memory Transfer Operations
        // ====================================================================

        /**
         * @brief Transfers tensor data to host (CPU) memory with optional type conversion
         *
         * Creates a new tensor in CPU memory containing a copy of this tensor's data.
         * Supports automatic type conversion during transfer. Always results in a
         * host-accessible tensor using CpuMemoryResource.
         *
         * @tparam TDstDataType Target tensor data type (defaults to current type)
         * @param cpu_context CPU device context for the destination tensor
         * @return New host tensor with transferred and optionally converted data
         *
         * @throws std::invalid_argument If cpu_context is null or not a CPU context
         * @throws std::runtime_error If memory transfer fails
         * @throws std::bad_alloc If host memory allocation fails
         *
         * @note Works from any source memory type (CPU, CUDA, etc.)
         * @note Type conversion performed on host for maximum compatibility
         * @note Preserves tensor shape, strides, and metadata
         */
        template<TensorDataType TDstDataType = TDataType>
			requires isValidTensor<TDstDataType, Compute::CpuMemoryResource>
        Tensor<TDstDataType, Compute::CpuMemoryResource> toHost( std::shared_ptr<Compute::CpuDeviceContext> cpu_context ) const {
            if (!cpu_context) {
                throw std::invalid_argument( "CPU device context cannot be null" );
            }

            Tensor<TDstDataType, Compute::CpuMemoryResource> dst( cpu_context, shape_ );

            if (!name_.empty()) {
                dst.setName( name_ );
            }

            if (size_ == 0) {
                return dst;
            }

            constexpr bool srcHost = TMemoryResource::is_host_accessible;

            if constexpr (srcHost) {
                // Host ? Host: Direct copy or conversion
                if constexpr (TDstDataType == TDataType) {
                    const size_t bytes = detail::getStorageSize<TDataType>( size_ );
                    std::memcpy( dst.rawData(), rawData(), bytes );
                }
                else {
                    performHostTypeConversion<TDstDataType>( dst );
                }
            }
            else {
                // Device ? Host: Transfer then optional conversion
                if constexpr (TDstDataType == TDataType) {
                    // Same type: direct device-to-host transfer
                    const size_t bytes = detail::getStorageSize<TDataType>( size_ );
                    // fixme: dst.buffer_->copyFrom( rawData(), bytes );
                }
                else {
                    // Different types: transfer to temp host, then convert
                    auto temp_cpu_context = std::dynamic_pointer_cast<Compute::CpuDeviceContext>(
                        createDeviceContext( "CPU" ));
                    Tensor<TDataType, Compute::CpuMemoryResource> temp_host( temp_cpu_context, shape_ );
                    const size_t src_bytes = detail::getStorageSize<TDataType>( size_ );
                    
                    // fixme: temp_host.buffer_->copyFrom( rawData(), src_bytes );

                    performHostTypeConversion( temp_host, dst );
                }
            }

            return dst;
        }

        /**
         * @brief Transfers tensor data to device memory with optional type conversion
         *
         * Creates a new tensor in device memory containing a copy of this tensor's data.
         * Supports automatic type conversion during transfer. The memory resource type
         * and device context must be compatible.
         *
         * @tparam TDstDataType Target tensor data type (defaults to current type)
         * @tparam TDeviceMemoryResource Target device memory resource type
         * @param device_context Device context for the destination tensor
         * @return New device tensor with transferred and optionally converted data
         *
         * @throws std::invalid_argument If device_context is null
         * @throws std::runtime_error If device operations fail or contexts are incompatible
         * @throws std::bad_alloc If device memory allocation fails
         *
         * @note Source must be host-accessible (CPU memory)
         * @note Type conversion performed on host before device transfer
         * @note Preserves tensor shape, strides, and metadata
         */
        template<TensorDataType TDstDataType = TDataType, typename TDeviceMemoryResource>
            requires isValidTensor<TDstDataType, TDeviceMemoryResource> && (!TDeviceMemoryResource::is_host_accessible)
        Tensor<TDstDataType, TDeviceMemoryResource> toDevice( std::shared_ptr<Compute::DeviceContext> device_context ) const {
            static_assert(TMemoryResource::is_host_accessible,
                "toDevice() requires source tensor to be host-accessible. Use toHost() first if needed.");

            if (!device_context) {
                throw std::invalid_argument( "Device context cannot be null" );
            }

            Tensor<TDstDataType, TDeviceMemoryResource> dst( device_context, shape_ );

            if (!name_.empty()) {
                dst.setName( name_ );
            }

            if (size_ == 0) {
                return dst;
            }

            if constexpr (TDstDataType == TDataType) {
                // Same type: direct host-to-device transfer
                const size_t bytes = detail::getStorageSize<TDataType>( size_ );
                // Fixme: dst.buffer_->copyFrom( rawData(), bytes );
            }
            else {
                // Different types: convert on host, then transfer
                auto temp_cpu_context = std::dynamic_pointer_cast<Compute::CpuDeviceContext>(
                    createDeviceContext( "CPU" ));
                Tensor<TDstDataType, Compute::CpuMemoryResource> temp_host( temp_cpu_context, shape_ );
                performHostTypeConversion<TDstDataType>( temp_host );

                const size_t dst_bytes = detail::getStorageSize<TDstDataType>( size_ );
                // fixme: dst.buffer_->copyFrom( temp_host.rawData(), dst_bytes );
            }

            return dst;
        }

        // ====================================================================
        // Data Manipulation Operations
        // ====================================================================

        /**
         * @brief Creates an independent deep copy of the tensor
         *
         * Constructs a new tensor with its own memory buffer containing
         * a complete copy of this tensor's data. Unlike copy construction
         * which is disabled, this method ensures complete independence
         * between the original and cloned tensors. Device context is shared.
         *
         * @return New tensor with independent memory containing copied data
         *
         * @throws std::bad_alloc If memory allocation fails
         * @throws std::runtime_error If data copy operation fails
         *
         * @note Creates completely independent tensor with unique identifier
         * @note Preserves name, device context, and all metadata
         * @note Both tensors can be modified independently after cloning
         * @note Handles packed data types and complex memory layouts correctly
         */
        Tensor<TDataType, TMemoryResource> clone() const {
            Tensor<TDataType, TMemoryResource> cloned_tensor(device_context_, shape_);

            if (size_ > 0) {
                cloned_tensor.buffer_->copyFrom(rawData(), buffer_->storageBytes());
            }

            if (!name_.empty()) {
                cloned_tensor.setName(name_);
            }

            return cloned_tensor;
        }

        // ====================================================================
        // Shape Transformation Operations
        // ====================================================================

        /**
         * @brief Modifies tensor shape while preserving total element count
         *
         * Changes the dimensional structure of the tensor without affecting
         * the underlying data or memory layout. The new shape must have the
         * same total number of elements as the current shape.
         *
         * @param new_shape Vector defining the new dimensional structure
         *
         * @throws std::runtime_error If new total size doesn't match current size
         *
         * @note Total element count must remain unchanged
         * @note Empty tensors can be reshaped to any size (will allocate memory)
         * @note Data order in memory remains unchanged
         * @note Automatically recomputes strides for new shape
         * @note Preserves data type, memory resource, and device context
         */
        void reshape(const std::vector<size_t>& new_shape) {
            size_t new_size = computeSize(new_shape);
            if (!empty() && (new_size != size_)) {
                throw std::runtime_error("The new shape must match the size of the tensor or the tensor must be empty.");
            }

            shape_ = new_shape;
            strides_ = computeStrides(new_shape);

            if (empty() && new_size > 0) {
                size_ = new_size;
                allocateBuffer();
            }
        }

        /**
         * @brief Flattens tensor to 2D shape in-place
         *
         * Reshapes the tensor to 2D by collapsing all dimensions except the
         * last into a single dimension. Commonly used for interfacing with
         * linear algebra operations and neural network layers.
         *
         * @return Reference to this tensor (for method chaining)
         *
         * @note For tensors with rank <= 1, no operation is performed
         * @note Result shape is [product_of_first_n-1_dims, last_dim]
         * @note Preserves memory layout and data order
         * @note Modifies this tensor directly
         * @note Essential for neural network layer transitions
         */
        Tensor<TDataType, TMemoryResource>& flatten() {
            if (rank() <= 1) {
                return *this;
            }

            size_t flat_dim = 1;
            for (size_t i = 0; i < shape().size() - 1; i++) {
                flat_dim *= shape()[i];
            }

            std::vector<size_t> new_shape = { flat_dim, shape().back() };
            reshape(new_shape);

            return *this;
        }

        /**
         * @brief Creates a flattened copy of the tensor
         *
         * Returns a new tensor that is a 2D representation of this tensor,
         * leaving the original tensor unchanged. Creates an independent copy
         * with flattened shape suitable for linear algebra operations.
         *
         * @return Flattened tensor with independent memory containing copied data
         *
         * @note Original tensor remains unchanged
         * @note Result is independent copy (deep copy)
         * @note For tensors with rank <= 1, returns clone of original
         * @note Result shape is [product_of_first_n-1_dims, last_dim]
         * @note Useful for neural network layer compatibility
         */
        Tensor<TDataType, TMemoryResource> flattened() const {
            if (rank() <= 1) {
                return clone();
            }

            Tensor<TDataType, TMemoryResource> flattened_tensor = clone();
            flattened_tensor.flatten();

            return flattened_tensor;
        }

        // ====================================================================
        // Identity and Metadata
        // ====================================================================

        /**
         * @brief Returns the tensor's unique identifier
         *
         * Provides the unique string identifier assigned during construction.
         * Useful for tracking, debugging, and logging tensor operations in
         * complex neural network computation graphs.
         *
         * @return Unique identifier string
         *
         * @note Format: "tensor_" followed by sequential number
         * @note Unique across all tensor instances in application lifetime
         * @note Each independently constructed tensor has unique ID
         * @note Preserved during moves but not during cloning
         */
        std::string getUId() const {
            return uid_;
        }

        /**
         * @brief Returns the tensor's optional user-assigned name
         *
         * Retrieves the user-assigned name for the tensor, useful for
         * debugging, visualization, and tracking in complex computations
         * and neural network architectures.
         *
         * @return Tensor name string (empty if no name assigned)
         *
         * @note Names are optional and can be empty
         * @note Names are preserved during memory transfers and cloning
         * @note Useful for debugging and tensor identification in computation graphs
         */
        std::string getName() const {
            return name_;
        }

        /**
         * @brief Assigns a descriptive name to the tensor
         *
         * Sets a user-defined name for the tensor to aid in debugging,
         * visualization, and tracking during computation workflows.
         * Particularly useful in neural network debugging and profiling.
         *
         * @param value New name for the tensor
         *
         * @throws std::invalid_argument If name is empty string
         *
         * @note Names must be non-empty strings
         * @note Names are preserved during memory transfers and cloning
         * @note Useful for debugging and tensor identification
         */
        void setName(const std::string& value) {
            if (value.empty()) {
                throw std::invalid_argument("Tensor name cannot be empty.");
            }
            name_ = value;
        }

        // ====================================================================
        // String Representation and Debugging
        // ====================================================================

        /**
         * @brief Generates comprehensive string representation of the tensor
         *
         * Creates a detailed string description including tensor metadata,
         * shape information, data type details, device context information,
         * and optionally the actual data values for debugging and inspection purposes.
         *
         * @param showBuffer Whether to include actual tensor data in output
         * @return Formatted string representation of the tensor
         *
         * @note Includes UID, name, shape, strides, size, abstract data type, and device info
         * @note Buffer content only shown for host-accessible tensors
         * @note Large tensors show truncated buffer content with ellipsis
         * @note Useful for debugging and neural network introspection
         * @note Handles packed data types appropriately
         */
        std::string toString(bool showBuffer = false) const {
            std::ostringstream oss;
            oss << "Tensor: " << uid_;
            if (!name_.empty())
                oss << "::" << name_;
            oss << ", ";
            oss << outputLayout();

            oss << " Type: " << DataTypeTraits::type_name;

            if (device_context_) {
                oss << ", Device: " << (device_context_->isCudaDevice() ? "CUDA:" + std::to_string(device_context_->getDeviceId()) : "CPU");
            }

            oss << std::endl;

            if (showBuffer) {
                oss << getBufferString();
            }

            return oss.str();
        }

    private:
        // ====================================================================
        // Private Member Variables
        // ====================================================================

        std::shared_ptr<Compute::DeviceContext> device_context_{ nullptr };       ///< Device context for proper device binding and resource management
        std::string uid_;                                                          ///< Unique identifier for this tensor instance
        std::string name_;                                                         ///< Optional user-assigned name for debugging
        size_t size_{ 0 };                                                         ///< Total number of logical elements in the tensor
        std::vector<size_t> shape_{};                                              ///< Dimensional sizes for each tensor dimension
        std::vector<size_t> strides_{};                                            ///< Memory stride values for multi-dimensional indexing
        std::shared_ptr<TensorBuffer<TDataType, TMemoryResource>> buffer_{ nullptr }; ///< Managed buffer containing tensor data
        std::shared_ptr<void> external_memory_ptr_{ nullptr };                    ///< Optional external memory reference

        // ====================================================================
        // Private Helper Methods
        // ====================================================================

        /**
         * @brief Creates appropriate device context based on device name
         *
         * Factory method that uses DeviceContext::create() to instantiate the correct
         * device context type based on the device name. Initializes device registrar
         * to ensure devices are available before creating context.
         *
         * @param device_name Device identifier string (e.g., "CPU", "CUDA:0")
         * @return Shared pointer to appropriate device context
         *
         * @throws std::runtime_error If device name is invalid or device creation fails
         */
        static std::shared_ptr<Compute::DeviceContext> createDeviceContext(const std::string& device_name) {
            // Review: Initialize device registrar to ensure devices are available
            Compute::DeviceRegistrar::instance();

            try {
                return Compute::DeviceContext::create(device_name);
            }
            catch (const std::exception& e) {
                throw std::runtime_error("Failed to create device context for '" + device_name + "': " + e.what());
            }
        }

        /**
         * @brief Validates that device context type matches memory resource requirements
         *
         * Ensures that CUDA memory resources are only used with CUDA device contexts
         * and provides clear error messages for mismatched configurations.
         *
         * @throws std::runtime_error If device context type doesn't match memory resource requirements
         */
        void validateDeviceContextCompatibility() {
            if constexpr (requires { typename TMemoryResource::CompatibleDeviceContext; }) {
                using RequiredContextType = typename TMemoryResource::CompatibleDeviceContext;

                if (!std::dynamic_pointer_cast<RequiredContextType>(device_context_)) {
                    throw std::runtime_error( "Device context type mismatch" );
                }
            }
        }

        /**
         * @brief Allocates and initializes the tensor's data buffer
         *
         * Creates TensorBuffer with device context as required by the latest constructor.
         * The buffer is responsible for device-aware memory allocation and management.
         */
        void allocateBuffer() {
            if (size_ > 0) {
                buffer_ = std::make_shared<TensorBuffer<TDataType, TMemoryResource>>(device_context_, size_);
            }
        }

        /**
         * @brief Performs memory transfer between tensors using buffer operations
         *
         * Handles data copying between tensors with different memory resource types
         * or data types using TensorBuffer's copy operations.
         *
         * @tparam TDstDataType Target tensor data type
         * @tparam TDstMR Target memory resource type
         * @param dst Destination tensor (already allocated)
         */
        template<TensorDataType TDstDataType, typename TDstMR>
        void copyWithMemoryTransfer(Tensor<TDstDataType, TDstMR>& dst) const {
            if constexpr (TDstDataType == TDataType) {
                // Same type: direct memory copy
                const size_t bytes = detail::getStorageSize<TDataType>(size_);
                // FIXME: dst.buffer_->copyFrom(rawData(), bytes);
            }
            else {
                // Different types: requires host-side conversion
                copyWithTypeConversion<TDstDataType, TDstMR>(dst);
            }
        }

        /**
         * @brief Performs type conversion between tensors with different data types
         *
         * Creates temporary host buffer if needed and performs element-wise conversion
         * between different tensor data types.
         *
         * @tparam TDstDataType Target tensor data type
         * @tparam TDstMR Target memory resource type
         * @param dst Destination tensor (already allocated)
         */
        template<TensorDataType TDstDataType, typename TDstMR>
        void copyWithTypeConversion(Tensor<TDstDataType, TDstMR>& dst) const {
            constexpr bool srcHost = TMemoryResource::is_host_accessible;
            constexpr bool dstHost = TDstMR::is_host_accessible;

            if constexpr (srcHost && dstHost) {
                // Both host-accessible: direct conversion
                copyWithHostConversion<TDstDataType, TDstMR>(dst);
            }
            else if constexpr (srcHost && !dstHost) {
                // Host to Device: convert on host, then transfer
                auto temp_cpu_context = createDeviceContext("CPU");
                Tensor<TDstDataType, Compute::CpuMemoryResource> temp_host(temp_cpu_context, shape_);
                copyWithHostConversion(*this, temp_host);

                const size_t dst_bytes = detail::getStorageSize<TDstDataType>(size_);
                dst.buffer_->copyFrom(temp_host.rawData(), dst_bytes);
            }
            else if constexpr (!srcHost && dstHost) {
                // Device to Host: transfer to host, then convert
                auto temp_cpu_context = createDeviceContext("CPU");
                Tensor<TDataType, Compute::CpuMemoryResource> temp_host(temp_cpu_context, shape_);
                const size_t src_bytes = detail::getStorageSize<TDataType>(size_);
                temp_host.buffer_->copyFrom(rawData(), src_bytes);

                copyWithHostConversion(temp_host, dst);
            }
        }

        /**
         * @brief Performs host-side type conversion copy
         *
         * Used when both source and destination are host-accessible but have
         * different data types requiring element-wise conversion.
         */
        template<TensorDataType TDstDataType, typename THostMR>
        void copyWithHostConversion( Tensor<TDstDataType, THostMR>& dst_tensor ) const {
            using SrcType = typename CpuTensorDataTypeTraits::template native_type<TDataType>;
            using DstType = typename CpuTensorDataTypeTraits::template native_type<TDstDataType>;

            const SrcType* src_data = static_cast<const SrcType*>(rawData());
            DstType* dst_data = static_cast<DstType*>(dst_tensor.rawData());

            // Element-wise conversion on host
            for (size_t i = 0; i < size_; ++i) {
                dst_data[i] = static_cast<DstType>( src_data[i] );
            }
        }

        void validateIndices(const std::vector<size_t>& indices, const char* fn) const {
            if (indices.size() != shape_.size()) {
                throw std::runtime_error(std::string(fn) + ": number of indices must match tensor rank");
            }
            for (size_t i = 0; i < indices.size(); ++i) {
                if (indices[i] >= shape_[i]) {
                    throw std::out_of_range(std::string(fn) + ": index " + std::to_string(indices[i]) +
                        " is out of range for dim " + std::to_string(i) + " size " + std::to_string(shape_[i]));
                }
            }
            if constexpr (!TMemoryResource::is_host_accessible) {
                throw std::runtime_error(std::string(fn) + ": direct access requires host-accessible memory. "
                    "Use to<TDataType, Compute::CpuMemoryResource>() first.");
            }
        }

        // Compute flat index using row-major strides
        size_t computeFlatIndex(const std::vector<size_t>& indices) const {
            size_t idx = 0;
            for (size_t d = 0; d < indices.size(); ++d) {
                idx += indices[d] * strides_[d];
            }
            return idx;
        }

        /**
         * @brief Generates formatted layout information string
         * @return String containing shape, strides, and size information
         */
        std::string outputLayout() const {
            std::string result = "TensorData(shape=[";

            for (size_t i = 0; i < shape_.size(); ++i) {
                result += std::to_string(shape_[i]);
                if (i < shape_.size() - 1) result += ",";
            }

            result += "], strides=[";

            for (size_t i = 0; i < strides_.size(); ++i) {
                result += std::to_string(strides_[i]);
                if (i < strides_.size() - 1) result += ",";
            }

            result += "], format=RowMajor";
            result += ", size=" + std::to_string(size_) + ")";

            return result;
        }

        /**
         * @brief Computes total element count from shape vector
         * @param shape Dimensional sizes
         * @return Product of all dimensions
         */
        size_t computeSize(const std::vector<size_t>& shape) {
            if (shape.empty()) {
                return 0;
            }
            return std::accumulate(shape.begin(), shape.end(), 1ull, std::multiplies<size_t>());
        }

        /**
         * @brief Computes row-major memory strides from shape
         * @param shape Dimensional sizes
         * @return Vector of stride values for each dimension
         */
        std::vector<size_t> computeStrides(const std::vector<size_t>& shape) {
            std::vector<size_t> strides(shape.size(), 1);

            if (shape.empty()) {
                return strides;
            }

            for (size_t i = shape.size() - 1; i > 0; --i) {
                strides[i - 1] = strides[i] * shape[i];
            }

            return strides;
        }

        /**
         * @brief Generates unique identifier for new tensor instance
         * @return Unique identifier string
         */
        std::string setUId() {
            return "tensor_" + std::to_string(UniqueIdGenerator::getNextId());
        }

        /**
         * @brief Gets formatted buffer content for string representation
         * @return Formatted buffer content or access warning
         */
        std::string getBufferString() const {
            if constexpr (is_host_accessible()) {
                return "Buffer content display not implemented for abstract data types";
            }
            else {
                return "Tensor is not host-accessible. Cannot output buffer contents.";
            }
        }
    };

    /**
     * @brief Stream insertion operator for tensor output
     *
     * Enables direct streaming of tensor objects to output streams for
     * convenient debugging and logging. Uses the tensor's toString() method
     * with comprehensive metadata display.
     *
     * @tparam TDataType Abstract tensor data type
     * @tparam TMemoryResource Memory resource type
     * @param os Output stream to write to
     * @param tensor Tensor object to output
     * @return Reference to the output stream for chaining
     */
    export template <TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    std::ostream& operator<<(std::ostream& os, const Tensor<TDataType, TMemoryResource>& tensor) {
        os << tensor.toString();
        return os;
    }

    /**
     * @brief Tensor type that uses host (CPU) memory with abstract data types
     *
     * Convenient alias for tensors stored in CPU memory that can be directly
     * accessed from host code without memory transfers. Uses abstract data
     * type system for device independence.
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
     */
    export template <TensorDataType TDataType>
        using HostTensor = Tensor<TDataType, Compute::CpuMemoryResource>;

    /**
     * @brief Tensor type that uses device (GPU) memory with abstract data types
     *
     * Convenient alias for tensors stored in GPU memory optimized for
     * device computation but requiring transfers for host access. Uses
     * abstract data type system for device independence.
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
     */
    export template <TensorDataType TDataType>
        using DeviceTensor = Tensor<TDataType, Compute::CudaMemoryResource>;

    /**
     * @brief Tensor type that uses pinned (page-locked) host memory
     *
     * Convenient alias for tensors using pinned host memory that provides
     * faster transfers to/from GPU while remaining host-accessible. Uses
     * abstract data type system for device independence.
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
     */
    export template <TensorDataType TDataType>
        using PinnedTensor = Tensor<TDataType, Compute::CudaPinnedMemoryResource>;

    /**
     * @brief Tensor type that uses CUDA managed memory
     *
     * Convenient alias for tensors using CUDA managed memory that is
     * accessible from both CPU and GPU with automatic migration. Uses
     * abstract data type system for device independence.
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
     */
    export template <TensorDataType TDataType>
        using UniversalTensor = Tensor<TDataType, Compute::CudaManagedMemoryResource>;
}