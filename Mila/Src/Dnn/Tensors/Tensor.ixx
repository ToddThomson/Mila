/**
 * @file Tensor.ixx
 * @brief Tensor implementation with proper scalar tensor support
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
 * - Proper scalar tensor (rank 0) support with dedicated accessor methods
 *
 * Scalar tensor semantics:
 * - Shape: {} (empty vector) represents rank 0
 * - Size: 1 (single element, per mathematical convention)
 * - Access: Use item() method, not operator[]
 * - Memory: Always allocated (size = 1)
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
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorHostTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.CpuTensorDataTypeTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaManagedMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.DeviceContext;
import Compute.ComputeDevice;
import Compute.CpuDevice;
import Compute.CudaDevice;
import Compute.DeviceRegistrar;
import Compute.DeviceRegistry;

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
        constexpr size_t getStorageSize( size_t logical_size ) {
            constexpr size_t element_size = TensorDataTypeTraits<TDataType>::size_in_bytes;

            if (logical_size > std::numeric_limits<size_t>::max() / element_size) {
                throw std::overflow_error( "Storage size calculation would overflow." );
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
            return counter_.fetch_add( 1, std::memory_order_relaxed );
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
     * - Proper scalar tensor (rank 0) support following mathematical conventions
     *
     * The tensor class is move-only to prevent accidental expensive copy operations.
     * Explicit operations are provided for data sharing, deep copying, and memory transfers.
     *
     * Tensor dimensionality:
     * - Scalar (rank 0): shape {}, size 1, access via item()
     * - Vector (rank 1): shape {n}, size n, access via operator[]
     * - Matrix (rank 2): shape {m, n}, size m*n, access via operator[]
     * - Higher-rank: shape {d1, d2, ...}, size = product of dimensions
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
     * @see ITensor for the polymorphic base interface
     * @see TensorBuffer for underlying device-specific memory management
     * @see MemoryResource for device memory abstraction layer
     * @see DeviceContext for device binding and resource management
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    class Tensor : public ITensor
    {
    public:

        using DataType = TensorDataType;                                           ///< Abstract data type enumeration
        using MemoryResource = TMemoryResource;                                    ///< Memory resource type for this tensor
        using DataTypeTraits = TensorDataTypeTraits<TDataType>;                   ///< Compile-time data type characteristics
        using host_value_t = std::conditional_t<TensorDataTypeTraits<TDataType>::is_integer_type, int32_t, float>; ///< Host value type for scalars

        // ====================================================================
        // Construction, Assignment, and Destruction
        // ====================================================================

        /**
         * @brief Creates a tensor with an existing compute device and specified shape
         *
         * Constructs a tensor using an already-created compute device, avoiding
         * device registry lookups. Useful when multiple tensors share the same
         * device or when working with devices obtained from other sources.
         *
         * @param device Shared pointer to compute device
         * @param shape Vector defining the size of each dimension in row-major order
         *
         * @throws std::invalid_argument If device is null
         * @throws std::runtime_error If device type doesn't match memory resource
         * @throws std::bad_alloc If memory allocation fails
         *
         * @note Device type must match TMemoryResource::device_type
         * @note More efficient than string-based constructor (no registry lookup)
         * @note Empty shape {} creates a scalar (0D tensor) with size 1
         * @note Shape {0} creates an empty 1D tensor with size 0
         *
         * Example:
         * @code
         * auto device = std::make_shared<CpuDevice>();
         *
         * // Create multiple tensors sharing the same device
         * auto tensor1 = Tensor<TensorDataType::FP32, CpuMemoryResource>(device, {3, 4});
         * auto tensor2 = Tensor<TensorDataType::FP32, CpuMemoryResource>(device, {5, 6});
         *
         * // From ExecutionContext
         * auto exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>(0);
         * auto tensor3 = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>(
         *     exec_ctx->getDevice(), {10, 20});
         * @endcode
         */
        explicit Tensor( std::shared_ptr<Compute::ComputeDevice> device, const std::vector<size_t>& shape )
            : device_( validateAndGetDevice( device ) ), uid_( setUId() ), shape_( shape ),
            strides_( computeStrides( shape ) ), size_( computeSize( shape ) ) {

            allocateBuffer();
        }

        /**
         * @brief Creates a tensor with device name and specified shape
         *
         * Constructs a tensor by verifying the device name through DeviceRegistry
         * and creating an appropriate device context internally. The device name is
         * validated to ensure the device exists before tensor construction proceeds.
         *
         * @param device_name Device identifier string (e.g., "CPU", "CUDA:0", METAL:1, etc. )
         * @param shape Vector defining the size of each dimension in row-major order
         *
         * @throws std::invalid_argument If device name is invalid or not registered
         * @throws std::runtime_error If device context creation fails or type mismatch
         * @throws std::bad_alloc If memory allocation fails
         *
         * @note Device name must be registered with DeviceRegistry
         * @note Automatically creates appropriate DeviceContext based on device type
         * @note Empty shape {} creates a scalar (0D tensor) with size 1
         * @note Shape {0} creates an empty 1D tensor with size 0
         *
         * Example:
         * @code
         * // Create scalar on CPU
         * auto scalar = Tensor<TensorDataType::FP32, CpuMemoryResource>("CPU", {});
         *
         * // Create vector on GPU
         * auto vector = Tensor<TensorDataType::FP16, CudaDeviceMemoryResource>("CUDA:0", {100});
         * @endcode
         */
        explicit Tensor( const std::string& device_name, const std::vector<size_t>& shape )
            : device_( createDevice( device_name ) ), uid_( setUId() ), shape_( shape ),
            strides_( computeStrides( shape ) ), size_( computeSize( shape ) ) {

            allocateBuffer();
        }

        /**
         * @brief Copy constructor - explicitly deleted for performance safety
         *
         * Tensors cannot be copied implicitly to prevent accidental expensive copy
         * operations in performance-critical neural network code. Use explicit
         * clone() for deep copies or std::move() for efficient ownership transfers.
         */
        Tensor( const Tensor& other ) = delete;

        /**
         * @brief Efficiently transfers ownership from another tensor
         *
         * Moves all resources from the source tensor, leaving it in a
         * clearly invalid "moved-from" state that is distinguishable from
         * an intentionally empty tensor.
         *
         * @param other Source tensor to move from (will be left in moved-from state)
         *
         * @note Source tensor will be in moved-from state after move
         * @note Moved-from tensors should not be used except for reassignment
         * @note empty() will return true for moved-from tensors
         */
        Tensor( Tensor&& other ) noexcept
            : device_( std::move( other.device_ ) ),
            uid_( std::move( other.uid_ ) ),
            name_( std::move( other.name_ ) ),
            size_( other.size_ ),
            shape_( std::move( other.shape_ ) ),
            strides_( std::move( other.strides_ ) ),
            buffer_( std::move( other.buffer_ ) ) {

            // Leave moved-from object in clearly invalid state
            // Use a sentinel shape that indicates "moved-from" rather than "empty"
            other.size_ = 0;
            other.shape_ = {};
            other.strides_ = {};
            other.device_ = nullptr;
        }

        /**
         * @brief Copy assignment operator - explicitly deleted for performance safety
         *
         * Tensors cannot be copy-assigned to prevent accidental expensive copy
         * operations. Use explicit clone() for deep copies or std::move() for transfers.
         */
        Tensor& operator=( const Tensor& other ) = delete;

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
        Tensor& operator=( Tensor&& other ) noexcept {
            if (this != &other)
            {
                device_ = std::move( other.device_ );
                uid_ = std::move( other.uid_ );
                name_ = std::move( other.name_ );
                shape_ = std::move( other.shape_ );
                strides_ = std::move( other.strides_ );
                size_ = other.size_;
                buffer_ = std::move( other.buffer_ );

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
        
        /**
         * @brief Returns the device type of this tensor's memory resource
         *
         * Provides the device type directly from the memory resource's static
         * device_type member. More efficient than querying device context.
         *
         * @return DeviceType enum value for this tensor's memory resource
         *
         * @note Implements ITensor interface
         * @note Compile-time constant propagated from TMemoryResource
         * @note Equivalent to TMemoryResource::device_type
         *
         * Example:
         * @code
         * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> cuda_tensor(...);
         * assert(cuda_tensor.getDeviceType() == DeviceType::Cuda);
         *
         * Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_tensor(...);
         * assert(cpu_tensor.getDeviceType() == DeviceType::Cpu);
         * @endcode
         */
        Compute::DeviceType getDeviceType() const override {
            return TMemoryResource::device_type;
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
         * @note Required by ITensor polymorphic interface
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
         * @note Required by ITensor polymorphic interface
         * @note Useful for debugging and runtime introspection
         * @note Consistent across all devices for same abstract type
         */
        std::string getDataTypeName() const override {
            return std::string( DataTypeTraits::type_name );
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

        // ====================================================================
        // Scalar Tensor Access (rank 0 only)
        // ====================================================================

        /**
         * @brief Checks if tensor is a scalar (0-dimensional)
         *
         * A scalar tensor has rank 0 (empty shape) and contains exactly
         * one element. Scalars represent single values without dimensional
         * structure. Distinguished from 1D tensors with one element.
         *
         * @return true if tensor is a scalar (rank 0), false otherwise
         *
         * @note Equivalent to rank() == 0
         * @note Scalars: shape {}, rank 0, size 1, NOT empty
         * @note Not scalars: shape {1} (1D vector), shape {0} (empty)
         * @note Use item() to access scalar value
         *
         * Example:
         * @code
         * Tensor<TensorDataType::FP32, CpuMemoryResource> scalar("CPU", {});
         * EXPECT_TRUE(scalar.isScalar());   // rank 0, size 1
         * EXPECT_EQ(scalar.rank(), 0u);
         * EXPECT_EQ(scalar.size(), 1u);
         * EXPECT_FALSE(scalar.empty());     // Scalars are NOT empty
         *
         * Tensor<TensorDataType::FP32, CpuMemoryResource> vector1d("CPU", {1});
         * EXPECT_FALSE(vector1d.isScalar()); // rank 1, not a scalar
         * @endcode
         */
        bool isScalar() const noexcept {
            return rank() == 0;
        }

        /**
         * @brief Gets the scalar value for 0-dimensional tensors
         *
         * Provides direct access to the single value in a scalar tensor.
         * Only available for host-accessible memory resources and 0D tensors.
         * Element type is automatically mapped via TensorHostTypeMap.
         *
         * @return Reference to the scalar value
         *
         * @throws std::runtime_error If tensor is not a scalar (rank != 0)
         *
         * @note Only available for host-accessible memory (compile-time enforced)
         * @note Return type is host_type from TensorHostTypeMap<TDataType>
         * @note For non-scalars, use operator[] with indices
         * @note Scalars are distinguishable from 1D tensors: use isScalar() to check
         *
         * @see TensorHostTypeMap for type mapping rules
         *
         * Example:
         * @code
         * Tensor<TensorDataType::FP32, CpuMemoryResource> scalar("CPU", {});
         * fill(scalar, 3.14f);
         * float value = scalar.item();  // Returns 3.14f
         *
         * scalar.item() = 2.71f;        // Mutable access
         * EXPECT_FLOAT_EQ(scalar.item(), 2.71f);
         * @endcode
         */
        auto& item() requires TMemoryResource::is_host_accessible {
            if (!isScalar()) {
                throw std::runtime_error( "item() can only be called on scalar tensors (rank 0). Use operator[] for higher-rank tensors." );
            }

            using HostType = typename TensorHostTypeMap<TDataType>::host_type;
            return static_cast<HostType*>(buffer_->data())[0];
        }

        /**
         * @brief Gets the scalar value for 0-dimensional tensors (const version)
         *
         * Provides read-only access to the single value in a scalar tensor.
         * Only available for host-accessible memory resources and 0D tensors.
         *
         * @return Const reference to the scalar value
         *
         * @throws std::runtime_error If tensor is not a scalar (rank != 0)
         *
         * @note Only available for host-accessible memory (compile-time enforced)
         * @note Return type is host_type from TensorHostTypeMap<TDataType>
         *
         * @see TensorHostTypeMap for type mapping rules
         */
        const auto& item() const requires TMemoryResource::is_host_accessible {
            if (!isScalar()) {
                throw std::runtime_error( "item() can only be called on scalar tensors (rank 0). Use operator[] for higher-rank tensors." );
            }

            using HostType = typename TensorHostTypeMap<TDataType>::host_type;
            return static_cast<const HostType*>(buffer_->data())[0];
        }

        // ====================================================================
        // Multi-dimensional Tensor Access (rank >= 1)
        // ====================================================================

        /**
         * @brief Accesses tensor element using multi-dimensional indices
         *
         * Provides direct element access using a vector of indices for each dimension.
         * Only available for host-accessible memory resources. Validates indices are
         * within bounds and computes the appropriate flat index for memory access.
         * Element type is automatically mapped via TensorHostTypeMap.
         *
         * @param indices Vector of indices, one per dimension
         * @return Reference to the element at the specified position
         *
         * @throws std::runtime_error If indices size doesn't match tensor rank
         * @throws std::runtime_error If tensor is scalar (use item() instead)
         * @throws std::out_of_range If any index is out of bounds for its dimension
         *
         * @note Only available for host-accessible memory resources at compile time
         * @note For scalars (rank 0), use item() instead
         * @note Return type is host_type from TensorHostTypeMap
         * @note Indices are validated against shape before access
         *
         * @see TensorHostTypeMap for type mapping rules
         *
         * Example:
         * @code
         * Tensor<TensorDataType::FP32, CpuMemoryResource> matrix("CPU", {3, 4});
         * matrix[{0, 0}] = 1.0f;
         * matrix[{2, 3}] = 9.0f;
         *
         * // Scalar tensors must use item()
         * Tensor<TensorDataType::FP32, CpuMemoryResource> scalar("CPU", {});
         * // scalar[{}] throws! Use: scalar.item() = 5.0f;
         * @endcode
         */
        auto& operator[]( const std::vector<size_t>& indices )
            requires TMemoryResource::is_host_accessible {

            if (isScalar()) {
                throw std::runtime_error( "Cannot use operator[] on scalar tensors. Use item() instead." );
            }

            validateIndices( indices, "operator[]" );

            size_t flat_index = computeFlatIndex( indices );
            using HostType = typename TensorHostTypeMap<TDataType>::host_type;

            return static_cast<HostType*>(buffer_->data())[flat_index];
        }

        /**
         * @brief Accesses tensor element using multi-dimensional indices (const version)
         *
         * Provides read-only direct element access using a vector of indices for each
         * dimension. Only available for host-accessible memory resources. Validates
         * indices are within bounds and computes the appropriate flat index.
         *
         * @param indices Vector of indices, one per dimension
         * @return Const reference to the element at the specified position
         *
         * @throws std::runtime_error If indices size doesn't match tensor rank
         * @throws std::runtime_error If tensor is scalar (use item() instead)
         * @throws std::out_of_range If any index is out of bounds for its dimension
         *
         * @note Only available for host-accessible memory resources at compile time
         * @note For scalars (rank 0), use item() instead
         * @note Uses host-compatible type mapping for element access
         * @note Indices are validated against shape before access
         */
        const auto& operator[]( const std::vector<size_t>& indices ) const
            requires TMemoryResource::is_host_accessible {

            if (isScalar()) {
                throw std::runtime_error( "Cannot use operator[] on scalar tensors. Use item() instead." );
            }

            validateIndices( indices, "operator[] const" );

            size_t flat_index = computeFlatIndex( indices );
            using HostType = typename TensorHostTypeMap<TDataType>::host_type;

            return static_cast<const HostType*>(buffer_->data())[flat_index];
        }

        /**
         * @brief Accesses tensor element using variadic indices
         *
         * Provides convenient element access using individual index arguments
         * instead of a vector. Forwards to the vector-based operator[] after
         * converting arguments to appropriate types.
         *
         * @tparam Indices Types of index arguments (must be convertible to size_t)
         * @param indices Individual index values for each dimension
         * @return Reference to the element at the specified position
         *
         * @throws std::runtime_error If number of indices doesn't match tensor rank
         * @throws std::runtime_error If tensor is scalar (use item() instead)
         * @throws std::out_of_range If any index is out of bounds for its dimension
         *
         * @note Only available for host-accessible memory resources at compile time
         * @note For scalars, use item() instead
         * @note All index arguments must be convertible to size_t
         * @note Convenient for accessing known-rank tensors: tensor[i, j, k]
         */
        template<typename... Indices>
        auto& operator[]( Indices... indices )
            requires TMemoryResource::is_host_accessible && (std::convertible_to<Indices, size_t> && ...) {
            std::vector<size_t> idx{ static_cast<size_t>(indices)... };
            return this->operator[]( idx );
        }

        /**
         * @brief Accesses tensor element using variadic indices (const version)
         *
         * Provides convenient read-only element access using individual index
         * arguments instead of a vector. Forwards to the vector-based const
         * operator[] after converting arguments to appropriate types.
         *
         * @tparam Indices Types of index arguments (must be convertible to size_t)
         * @param indices Individual index values for each dimension
         * @return Const reference to the element at the specified position
         *
         * @throws std::runtime_error If number of indices doesn't match tensor rank
         * @throws std::runtime_error If tensor is scalar (use item() instead)
         * @throws std::out_of_range If any index is out of bounds for its dimension
         *
         * @note Only available for host-accessible memory resources at compile time
         * @note For scalars, use item() instead
         * @note All index arguments must be convertible to size_t
         * @note Convenient for accessing known-rank tensors: tensor[i, j, k]
         */
        template<typename... Indices>
        const auto& operator[]( Indices... indices ) const
            requires TMemoryResource::is_host_accessible &&
        (std::convertible_to<Indices, size_t> && ...) {
            std::vector<size_t> idx{ static_cast<size_t>(indices)... };
            return this->operator[]( idx );
        }

        // ====================================================================
        // Tensor Properties and Introspection
        // ====================================================================

        /**
         * @brief Gets the compute device associated with this tensor.
         *
         * Returns the device that this tensor's memory is bound to, providing
         * access to device properties and capabilities.
         *
         * @return Shared pointer to the compute device
         *
         * @note Used by TensorOps for device-aware operations
         * @note Enables device property queries without accessing context
         * @note Device is set during construction and remains constant
         */
        std::shared_ptr<Compute::ComputeDevice> getDevice() const {
            return device_;
        }

        /**
         * @brief Returns the tensor's dimensional shape vector
         *
         * Provides the size of each dimension in row-major order. The shape
         * completely defines the tensor's dimensional structure and determines
         * memory layout through computed strides.
         *
         * @return Const reference to vector containing dimension sizes
         *
         * @note Required by ITensor polymorphic interface
         * @note Empty vector {} indicates a scalar (0-dimensional) tensor
         * @note Order is from outermost to innermost dimension (row-major)
         * @note Shape determines stride computation and memory indexing
         *
         * Shape interpretation:
         * - {} = scalar (rank 0, size 1)
         * - {n} = vector (rank 1, size n)
         * - {m, n} = matrix (rank 2, size m*n)
         * - {0} = empty 1D vector (rank 1, size 0)
         * - {0, n} = empty 2D matrix (rank 2, size 0)
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
         * @note Empty for scalars (rank 0)
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
         * @note 1 for scalars (rank 0)
         * @note 0 for empty tensors (any dimension is 0)
         * @note Product of all shape dimensions
         * @note Logical count may differ from storage bytes for packed types
         *
         * Examples:
         * - shape {} (scalar) -> size 1
         * - shape {5} -> size 5
         * - shape {3, 4} -> size 12
         * - shape {0} -> size 0 (empty)
         * - shape {2, 0, 3} -> size 0 (empty)
         */
        size_t size() const override {
            return size_;
        }

        /**
         * @brief Checks if the tensor contains no elements
         *
         * Determines whether the tensor has been allocated with zero size.
         * A tensor is empty when its size is 0, which occurs when any
         * dimension in the shape is 0.
         *
         * @return true if tensor has no elements (size == 0), false otherwise
         *
         * A tensor is empty when any dimension is 0, resulting in size() == 0.
         * 
         * Construction:
         *   Tensor<FP32, CpuMemoryResource> empty(device, {0});         // Valid
         *   Tensor<FP32, CpuMemoryResource> empty2d(device, {0, 5});    // Valid
         *   Tensor<FP32, CpuMemoryResource> empty3d(device, {3, 0, 4}); // Valid
         * 
         * Properties:
         *   empty.size() == 0
         *   empty.empty() == true
         *   empty.rank() == number of dimensions (? 1)
         *   empty.shape() contains the dimensions (at least one is 0)
         *   empty.data() == nullptr (no allocation)
         * 
         * Operations:
         *   - Reduction ops (sum, mean, etc.) return identity or throw
         *   - Element-wise ops propagate emptiness
         *   - Reshape allowed if size remains 0
         *   - Concatenation allowed (concatenating with non-empty changes size)
         * 
         * Not Empty:
         *   Tensor<FP32, CpuMemoryResource> scalar(device, {});  // NOT empty!
         *   scalar.size() == 1   // Scalar has one element
         *   scalar.empty() == false
         * 
         * Moved-From State:
         *   Should be distinguishable from intentionally empty tensors.
         *   Consider: moved-from -> shape {0}, empty -> shape {0}, but different semantics
         * 
         * @note Scalars (rank 0) are NOT empty - they have size 1
         * @note Empty shape {} -> scalar (size 1) -> NOT empty
         * @note Shape {0} -> empty 1D vector (size 0) -> empty
         * @note Shape {2, 0, 3} -> empty 3D tensor (size 0) -> empty
         * @note Empty tensors require no storage allocation
         * @note Equivalent to size() == 0
         *
         * Example:
         * @code
         * Tensor<TensorDataType::FP32, CpuMemoryResource> scalar("CPU", {});
         * EXPECT_FALSE(scalar.empty());  // Scalars are NOT empty
         * EXPECT_EQ(scalar.size(), 1u);
         *
         * Tensor<TensorDataType::FP32, CpuMemoryResource> empty("CPU", {0});
         * EXPECT_TRUE(empty.empty());    // Zero-size tensors ARE empty
         * EXPECT_EQ(empty.size(), 0u);
         * @endcode
         */
        bool empty() const {
            return (size_ == 0);
        }

        /**
         * @brief Returns the number of dimensions in the tensor
         *
         * Provides the tensor's dimensionality (rank), ranging from 0 (scalar)
         * to arbitrarily high dimensions. Essential for understanding
         * tensor structure in neural network operations.
         *
         * @return Number of dimensions (rank)
         *
         * @note Equivalent to shape().size()
         * @note Rank 0 -> scalar (single value, size 1)
         * @note Rank 1 -> vector (1D array)
         * @note Rank 2 -> matrix (2D array)
         * @note Common neural network tensors: 2D (matrices) to 4D (batch, channel, height, width)
         *
         * Example:
         * @code
         * Tensor<TensorDataType::FP32, CpuMemoryResource> scalar("CPU", {});
         * EXPECT_EQ(scalar.rank(), 0u);  // Scalar
         *
         * Tensor<TensorDataType::FP32, CpuMemoryResource> vector("CPU", {5});
         * EXPECT_EQ(vector.rank(), 1u);  // Vector
         *
         * Tensor<TensorDataType::FP32, CpuMemoryResource> matrix("CPU", {3, 4});
         * EXPECT_EQ(matrix.rank(), 2u);  // Matrix
         * @endcode
         */
        size_t rank() const {
            return shape_.size();
        }

        /**
         * @brief Check if tensor is in a valid state (not moved-from)
         */
        bool isValid() const noexcept {
            return device_ != nullptr;
        }

        // ====================================================================
        // Data Pointers
        // ====================================================================
       
        /**
         * @brief Returns type-safe pointer to tensor data with concrete host type
         *
         * Provides a type-safe pointer wrapper that automatically uses the concrete
         * host-compatible type corresponding to the tensor's abstract data type.
         * Only available for host-accessible memory resources. Return type is
         * TensorPtr<host_type> where host_type is determined by TensorHostTypeMap.
         *
         * @return TensorPtr<host_type> with automatic type mapping
         *
         * @note Only available for host-accessible memory resources (compile-time enforced)
         * @note Return type automatically mapped: FP16->float, INT8->int8_t, etc.
         * @note Provides type-safe access without manual casting
         * @note Use auto for automatic type deduction: auto ptr = tensor.data();
         * @note Works with scalars - returns pointer to single element
         *
         * @see TensorPtr for pointer wrapper operations
         * @see TensorHostTypeMap for type mapping rules
         *
         * Example:
         * @code
         * Tensor<TensorDataType::FP16, CpuMemoryResource> t("CPU", {10});
         * auto ptr = t.data();  // TensorPtr<float>, not TensorPtr<__half>
         * ptr[0] = 3.14f;       // Type-safe float access
         *
         * Tensor<TensorDataType::FP32, CpuMemoryResource> scalar("CPU", {});
         * auto sptr = scalar.data();  // Points to single element
         * @endcode
         */
        [[nodiscard]] constexpr auto* data() noexcept requires TMemoryResource::is_host_accessible {
            using HostType = typename TensorHostTypeMap<TDataType>::host_type;
            return static_cast<HostType*>(buffer_ ? buffer_->data() : nullptr);
        }

        /**
         * @brief Returns type-safe immutable pointer to tensor data with concrete host type
         *
         * Provides a read-only type-safe pointer wrapper that automatically uses the
         * concrete host-compatible type corresponding to the tensor's abstract data type.
         * Only available for host-accessible memory resources.
         *
         * @return Const TensorPtr<host_type> for safe read operations
         *
         * @note Only available for host-accessible memory resources (compile-time enforced)
         * @note Return type automatically mapped from abstract TensorDataType
         * @note Provides type-safe read access without manual casting
         * @note Works with scalars - returns pointer to single element
         *
         * @see TensorPtr for pointer wrapper operations
         * @see TensorHostTypeMap for type mapping rules
         */
        [[nodiscard]] constexpr const auto* data() const noexcept requires TMemoryResource::is_host_accessible {
            using HostType = typename TensorHostTypeMap<TDataType>::host_type;
            return static_cast<const HostType*>(buffer_ ? buffer_->data() : nullptr);
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
         * @throws std::runtime_error If new total size doesn't match current size (except for empty tensors)
         *
         * @note Total element count must remain unchanged (unless empty)
         * @note Empty tensors can be reshaped to any size (will allocate memory)
         * @note Data order in memory remains unchanged
         * @note Automatically recomputes strides for new shape
         * @note Preserves data type, memory resource, and device context
         * @note Scalars can be reshaped to vectors and vice versa if size matches
         *
         * Scalar reshaping examples:
         * @code
         * Tensor<TensorDataType::FP32, CpuMemoryResource> scalar("CPU", {});
         * // scalar.reshape({1}) would work - both have size 1
         * // scalar.reshape({2}) would throw - size mismatch
         *
         * Tensor<TensorDataType::FP32, CpuMemoryResource> vec1("CPU", {1});
         * vec1.reshape({});  // Convert to scalar - both have size 1
         * EXPECT_TRUE(vec1.isScalar());
         * @endcode
         */
        void reshape( const std::vector<size_t>& new_shape ) {
            size_t new_size = computeSize( new_shape );
            if (!empty() && (new_size != size_)) {
                throw std::runtime_error( "The new shape must match the size of the tensor or the tensor must be empty." );
            }

            shape_ = new_shape;
            strides_ = computeStrides( new_shape );

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
         * @note For tensors with rank <= 1 (scalars and vectors), no operation is performed
         * @note Result shape is [product_of_first_n-1_dims, last_dim]
         * @note Preserves memory layout and data order
         * @note Modifies this tensor directly
         * @note Essential for neural network layer transitions
         * @note Scalars remain unchanged (rank 0)
         */
        Tensor<TDataType, TMemoryResource>& flatten() {
            throw std::runtime_error( "Tensor::flattened() is not implemented. Use TensorOps for deep copy and shape transformation." );
        }

        Tensor<TDataType, TMemoryResource> flattened() const {
            throw std::runtime_error( "Tensor::flattened() is not implemented. Use TensorOps for deep copy and shape transformation." );
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
        void setName( const std::string& value ) {
            if (value.empty()) {
                throw std::invalid_argument( "Tensor name cannot be empty." );
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
         * @note Scalars display with empty shape and single stride
         */
        std::string toString( bool showBuffer = false ) const {
            std::ostringstream oss;
            oss << "Tensor: " << uid_;
            if (!name_.empty())
                oss << "::" << name_;
            oss << ", ";
            oss << outputLayout();

            oss << " Type: " << DataTypeTraits::type_name;
            oss << ", Device: " << device_->getDeviceName();
            oss << std::endl;

            if (showBuffer) {
                oss << getBufferString();
            }

            return oss.str();
        }

    protected:
        /**
         * @brief Returns raw pointer to tensor data (implements ITensor protected API)
         *
         * Provides type-erased access to tensor memory for TensorOps implementations.
         * This is the internal API that allows device-specific operations to access
         * tensor data uniformly across all memory resource types.
         *
         * @return Raw void pointer to tensor data
         * @note Implements ITensor::rawData()
         */
        void* rawData() override {
            return buffer_ ? buffer_->data() : nullptr;
        }

        /**
         * @brief Returns raw pointer to tensor data (const version)
         */
        const void* rawData() const override {
            return buffer_ ? buffer_->data() : nullptr;
        }

        /**
         * @brief Returns pointer to the memory resource managing this tensor's storage
         *
         * Provides efficient access to the memory resource for dispatch optimization,
         * zero-copy operations when memory resources are compatible, and type-safe
         * downcasting to specific tensor types.
         *
         * @return Pointer to memory resource (never null for valid tensors)
         *
         * @note Implements ITensor interface
         * @note Return type is base MemoryResource pointer for polymorphic access
         * @note Can be safely cast to TMemoryResource* based on template parameter
         * @note Enables efficient memory resource compatibility checks
         *
         * Example:
         * @code
         * ITensor& tensor = getTensor();
         * auto* mr = tensor.getMemoryResource();
         *
         * // Type-safe downcast check
         * if (auto* cuda_mr = dynamic_cast<CudaDeviceMemoryResource*>(mr)) {
         *     // Use CUDA-specific operations
         * }
         * @endcode
         */
        Compute::MemoryResource* getMemoryResource() const override {
            return buffer_ ? buffer_->getMemoryResource() : nullptr;
        }

private:

        std::shared_ptr<Compute::ComputeDevice> device_{ nullptr };       ///< Device for proper device binding and resource management
        std::string uid_;                                                          ///< Unique identifier for this tensor instance
        std::string name_;                                                         ///< Optional user-assigned name for debugging
        size_t size_{ 0 };                                                         ///< Total number of logical elements in the tensor
        std::vector<size_t> shape_{};                                              ///< Dimensional sizes for each tensor dimension
        std::vector<size_t> strides_{};                                            ///< Memory stride values for multi-dimensional indexing
        std::shared_ptr<TensorBuffer<TDataType, TMemoryResource>> buffer_{ nullptr }; ///< Managed buffer containing tensor data

        // ====================================================================
        // Private Helper Methods
        // ====================================================================

        /**
         * @brief Validates device and returns it if valid
         *
         * Ensures the provided device is not null and matches the memory resource's
         * required device type.
         *
         * @param device Device to validate
         * @return The validated device
         * @throws std::invalid_argument If device is null
         * @throws std::runtime_error If device type doesn't match memory resource
         */
        static std::shared_ptr<Compute::ComputeDevice> validateAndGetDevice(
            std::shared_ptr<Compute::ComputeDevice> device )
        {
            if (!device)
            {
                throw std::invalid_argument( "Device cannot be null" );
            }

            // Validate device type matches memory resource requirement
            constexpr Compute::DeviceType required_type = TMemoryResource::device_type;

            if (device->getDeviceType() != required_type)
            {
                throw std::runtime_error(
                    "Device type mismatch: Memory resource requires " +
                    deviceToString( required_type ) +
                    " but provided device is " +
                    deviceToString( device->getDeviceType() )
                );
            }

            return device;
        }

        /**
         * @brief Creates appropriate device context based on device name
         *
         * Factory method that verifies the device name through DeviceRegistry,
         * then uses DeviceContext::create() to instantiate the correct device
         * context type. Initializes device registrar to ensure devices are available.
         *
         * @param device_name Device identifier string (e.g., "CPU", "CUDA:0", "METAL:0")
         * @return Shared pointer to appropriate device context
         *
         * @throws std::invalid_argument If device name is not registered with DeviceRegistry
         * @throws std::runtime_error If device context creation fails
         *
         * @note Verifies device exists in registry before creating context
         * @note Initializes device registrar automatically
         */
        static std::shared_ptr<Compute::ComputeDevice> createDevice( const std::string& device_name ) {
            // Lazy Initialize device registrar to ensure all devices are available upon first use
            Compute::DeviceRegistrar::instance();

            std::shared_ptr<Compute::ComputeDevice> device;
            
            try {
                device = Compute::DeviceRegistry::instance().getDevice( device_name );
            }
            catch (const std::exception& e) {
                throw std::runtime_error(
                    "Failed to create device for '" + device_name + "': " + e.what()
                );
            }

            // Validate device type matches memory resource requirement
            constexpr Compute::DeviceType required_type = TMemoryResource::device_type;

            if (device->getDeviceType() != required_type) {
                throw std::runtime_error(
                    "Device type mismatch: Memory resource requires " +
                    deviceToString( required_type ) +
                    " but device '" + device_name + "' is " +
                    deviceToString( device->getDeviceType() )
                );
            }

            return device;
        }

        /**
         * @brief Allocates and initializes the tensor's data buffer
         *
         * Creates TensorBuffer with device context. Handles zero-size tensors
         * efficiently - no allocation occurs for truly empty tensors (size = 0),
         * but scalars (empty shape, size = 1) are allocated normally.
         */
        void allocateBuffer() {
            if (size_ > 0) {
                buffer_ = std::make_shared<TensorBuffer<TDataType, TMemoryResource>>( device_->getDeviceId(), size_);
            }
        }

        /**
         * @brief Validates multi-dimensional indices against tensor shape
         *
         * Ensures that the provided indices match the tensor rank and that
         * each index is within bounds for its corresponding dimension.
         *
         * @param indices Vector of indices to validate
         * @param fn Function name for error messages (e.g., "operator[]")
         *
         * @throws std::runtime_error If indices size doesn't match tensor rank
         * @throws std::out_of_range If any index exceeds its dimension's size
         */
        void validateIndices( const std::vector<size_t>& indices, const char* fn ) const {
            if (indices.size() != shape_.size()) {
                throw std::runtime_error( std::string( fn ) + ": number of indices must match tensor rank" );
            }
            for (size_t i = 0; i < indices.size(); ++i) {
                if (indices[i] >= shape_[i]) {
                    throw std::out_of_range( std::string( fn ) + ": index " + std::to_string( indices[i] ) +
                        " is out of range for dim " + std::to_string( i ) + " size " + std::to_string( shape_[i] ) );
                }
            }
        }

        /**
         * @brief Computes flat memory index from multi-dimensional indices
         *
         * Converts multi-dimensional indices to a single linear index using
         * row-major (C-style) memory layout and precomputed strides.
         *
         * @param indices Vector of indices (must be pre-validated)
         * @return Flat index into contiguous memory
         *
         * @note Assumes indices are already validated
         * @note Uses row-major stride computation
         */
        size_t computeFlatIndex( const std::vector<size_t>& indices ) const {
            size_t idx = 0;
            for (size_t d = 0; d < indices.size(); ++d) {
                idx += indices[d] * strides_[d];
            }
            return idx;
        }

        /**
         * @brief Generates formatted layout information string
         *
         * Creates a string representation of tensor layout including shape,
         * strides, format, and size for debugging and display purposes.
         *
         * @return String containing shape, strides, and size information
         */
        std::string outputLayout() const {
            std::string result = "TensorData(shape=[";

            for (size_t i = 0; i < shape_.size(); ++i) {
                result += std::to_string( shape_[i] );
                if (i < shape_.size() - 1) result += ",";
            }

            result += "], strides=[";

            for (size_t i = 0; i < strides_.size(); ++i) {
                result += std::to_string( strides_[i] );
                if (i < strides_.size() - 1) result += ",";
            }

            result += "], format=RowMajor";
            result += ", size=" + std::to_string( size_ ) + ")";

            return result;
        }

        /**
         * @brief Computes total element count from shape vector
         *
         * Calculates the product of all dimensions, representing the total number
         * of logical elements in the tensor. Empty shape represents a scalar
         * (0-dimensional) tensor with size 1, following mathematical convention
         * where the product of an empty sequence is 1 (multiplicative identity).
         *
         * @param shape Dimensional sizes
         * @return Product of all dimensions (1 for empty shape = scalar)
         *
         * @note Empty shape {} -> size 1 (scalar, rank 0)
         * @note Shape {0} -> size 0 (empty 1D vector, rank 1)
         * @note Shape {0, 5} -> size 0 (empty 2D matrix, rank 2)
         * @note Shape {2, 3} -> size 6 (2x3 matrix, rank 2)
         * @note Shape {3, 4, 5} -> size 60 (3x4x5 tensor, rank 3)
         *
         * Mathematical rationale:
         * - Product of empty sequence = 1 (multiplicative identity)
         * - This correctly treats scalars as having one element
         * - Empty tensors have at least one dimension with size 0
         *
         * Examples:
         * @code
         * computeSize({})        -> 1   // Scalar (0D)
         * computeSize({5})       -> 5   // 1D vector
         * computeSize({0})       -> 0   // Empty 1D vector
         * computeSize({3, 4})    -> 12  // 2D matrix
         * computeSize({0, 5})    -> 0   // Empty 2D matrix
         * computeSize({2, 3, 4}) -> 24  // 3D tensor
         * @endcode
         */
        size_t computeSize( const std::vector<size_t>& shape ) {
            // Product of empty sequence is 1 (multiplicative identity) for scalar construction
            return std::accumulate( shape.begin(), shape.end(), 1ull, std::multiplies<size_t>() );
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
        using DeviceTensor = Tensor<TDataType, Compute::CudaDeviceMemoryResource>;

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