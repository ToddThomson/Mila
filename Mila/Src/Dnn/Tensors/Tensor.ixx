/**
 * @file Tensor.ixx
 * @brief Device-aware tensor type with scalar support
 *
 * Provides a device-bound N-dimensional tensor with explicit memory-resource
 * abstraction and host/device access semantics. Scalars use empty shape {}.
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
#include <exception>
#include <optional>

export module Dnn.Tensor;

import Dnn.TensorBuffer;
import Dnn.ITensor;
import Dnn.TensorTypes;
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
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
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
     * Provides atomic generation of unique IDs for tensor instances.
     */
    class UniqueIdGenerator {
    public:
        /**
         * @brief Generates the next unique identifier atomically
         *
         * Uses relaxed memory ordering for uniqueness across threads.
         *
         * @return Unique size_t identifier
         */
        static size_t getNextId() {
            return counter_.fetch_add( 1, std::memory_order_relaxed );
        }

    private:
        static std::atomic<size_t> counter_; ///< Thread-safe counter for unique ID generation
    };

    std::atomic<size_t> UniqueIdGenerator::counter_{ 0 };

    /**
     * @brief Device-aware N-dimensional tensor
     *
     * Move-only tensor parameterized by an abstract TensorDataType and a MemoryResource.
     * Scalars are represented by an empty shape ({}), which yields size() == 1.
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
     * @tparam TMemoryResource Memory resource type defining storage location and access patterns
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
         * @brief Creates a tensor bound to the specified device id and shape
         *
         * Constructs a tensor using a DeviceId. Empty shape {} creates a scalar
         * (0D tensor) with size 1. Shape containing a zero produces an empty tensor.
         *
         * @param device_id Device identifier (type + index)
         * @param shape Vector defining the size of each dimension in row-major order
         *
         * @throws std::invalid_argument If device id is invalid (checked by validateDeviceId)
         * @throws std::runtime_error If device type doesn't match memory resource
         * @throws std::bad_alloc If memory allocation fails
         */
        explicit Tensor( Compute::DeviceId device_id, const shape_t& shape )
            : device_id_( validateDeviceId( device_id ) ), uid_( setUId() ), shape_( shape ),
            strides_( computeStrides( shape ) ), size_( computeSize( shape ) ) {

            allocateBuffer();
        }

        /**
         * @brief Copy constructor - explicitly deleted for performance safety
         *
         * Use clone() for deep copies or std::move() for ownership transfer.
         */
        Tensor( const Tensor& other ) = delete;

        /**
         * @brief Efficiently transfers ownership from another tensor
         *
         * Moves resources from the source tensor, leaving it in a moved-from state.
         */
        Tensor( Tensor&& other ) noexcept
            : device_id_( std::move( other.device_id_ ) ),
            uid_( std::move( other.uid_ ) ),
            name_( std::move( other.name_ ) ),
            size_( other.size_ ),
            shape_( std::move( other.shape_ ) ),
            strides_( std::move( other.strides_ ) ),
            buffer_( std::move( other.buffer_ ) ) {

            // Leave moved-from object in clearly invalid state
            other.size_ = 0;
            other.shape_ = {};
            other.strides_ = {};
            other.device_id_ = {};
        }

        /**
         * @brief Copy assignment operator - explicitly deleted
         *
         * Use clone() for deep copies or std::move() for transfers.
         */
        Tensor& operator=( const Tensor& other ) = delete;

        /**
         * @brief Efficiently moves resources from another tensor
         *
         * Self-assignment safe implementation.
         */
        Tensor& operator=( Tensor&& other ) noexcept {
            if (this != &other)
            {
                device_id_ = std::move( other.device_id_ );
                uid_ = std::move( other.uid_ );
                name_ = std::move( other.name_ );
                shape_ = std::move( other.shape_ );
                strides_ = std::move( other.strides_ );
                size_ = other.size_;
                buffer_ = std::move( other.buffer_ );

                other.size_ = 0;
                other.shape_.clear();
                other.strides_.clear();
                // FIXME: other.device_id_.clear();
            }

            return *this;
        }

        /**
         * @brief Destructor with automatic resource cleanup via RAII
         */
        ~Tensor() = default;
        
        /**
         * @brief Returns the device type of this tensor's memory resource
         *
         * Equivalent to TMemoryResource::device_type.
         */
        Compute::DeviceType getDeviceType() const override {
            return TMemoryResource::device_type;
        }

        // ====================================================================
        // Type Information and Interface Compliance
        // ====================================================================
        
        /**
         * @brief Returns the size in bytes of a single tensor element
         */
        size_t elementSize() const override
        {
            return DataTypeTraits::size_in_bytes;
        }

        /**
         * @brief Returns the tensor's abstract data type identifier
         */
        TensorDataType getDataType() const override {
            return TDataType;
        }

        /**
         * @brief Returns human-readable name of the tensor's data type
         */
        std::string getDataTypeName() const override {
            return std::string( DataTypeTraits::type_name );
        }

        /**
         * @brief Checks if tensor data is accessible from host (CPU) code
         */
        static constexpr bool is_host_accessible() {
            return TMemoryResource::is_host_accessible;
        }

        /**
         * @brief Checks if tensor data is accessible from device (GPU) code
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
         * Scalars: shape {} and size() == 1.
         */
        bool isScalar() const noexcept override {
            return rank() == 0;
        }

        /**
         * @brief Gets the scalar value for 0-dimensional tensors
         *
         * Only available for host-accessible memory resources.
         *
         * @throws std::runtime_error If tensor is not a scalar (rank != 0)
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
         * Only available for host-accessible memory resources.
         *
         * @throws std::runtime_error If tensor is not a scalar (rank != 0)
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
         * Only available for host-accessible memory resources.
         */
        auto& operator[]( const std::vector<int64_t>& indices )
            requires TMemoryResource::is_host_accessible {

            if (isScalar()) {
                throw std::runtime_error( "Cannot use operator[] on scalar tensors. Use item() instead." );
            }

            validateIndices( indices, "operator[]" );

            int64_t flat_index = computeFlatIndex( indices );
            using HostType = typename TensorHostTypeMap<TDataType>::host_type;

            return static_cast<HostType*>(buffer_->data())[flat_index];
        }

        /**
         * @brief Accesses tensor element using multi-dimensional indices (const version)
         */
        const auto& operator[]( const std::vector<int64_t>& indices ) const
            requires TMemoryResource::is_host_accessible {

            if (isScalar()) {
                throw std::runtime_error( "Cannot use operator[] on scalar tensors. Use item() instead." );
            }

            validateIndices( indices, "operator[] const" );

            int64_t flat_index = computeFlatIndex( indices );
            using HostType = typename TensorHostTypeMap<TDataType>::host_type;

            return static_cast<const HostType*>(buffer_->data())[flat_index];
        }

        /**
         * @brief Variadic index access (non-const)
         */
        template<typename... Indices>
        auto& operator[]( Indices... indices )
            requires TMemoryResource::is_host_accessible && (std::convertible_to<Indices, int64_t> && ...) {
            std::vector<int64_t> idx{ static_cast<int64_t>(indices)... };
            return this->operator[]( idx );
        }

        /**
         * @brief Variadic index access (const)
         */
        template<typename... Indices>
        const auto& operator[]( Indices... indices ) const
            requires TMemoryResource::is_host_accessible && (std::convertible_to<Indices, int64_t> && ...)
        {
            index_t idx{ static_cast<int64_t>(indices)... };

            return this->operator[]( idx );
        }

        // ====================================================================
        // Tensor Properties and Introspection
        // ====================================================================

        Compute::DeviceId getDeviceId() const override
        {
            return device_id_;
        }

        /**
         * @brief Gets the compute device associated with this tensor.
         *
         * Performs an on-demand lookup from the DeviceRegistry and returns a
         * shared_ptr to the device. Returns nullptr if the device cannot be created.
         */
        //std::shared_ptr<Compute::Device> getDevice() const override {
        //    if (!device_id_.has_value()) {
        //        return nullptr;
        //    }

        //    // Ensure device registrar ran so factories are registered
        //    Compute::DeviceRegistrar::instance();

        //    try {
        //        return Compute::DeviceRegistry::instance().getDevice( device_id_.value() );
        //    }
        //    catch ( ... ) {
        //        return nullptr;
        //    }
        //}

        /**
         * @brief Returns the tensor's dimensional shape vector
         */
        const shape_t& shape() const override {
            return shape_;
        }

        /**
         * @brief Returns the tensor's memory stride information
         */
        const stride_t& strides() const {
            return strides_;
        }

        /**
         * @brief Returns the total number of logical elements in the tensor
         */
        size_t size() const override {
            return size_;
        }

        /**
         * @brief Checks if the tensor contains no elements
         *
         * Scalars are NOT empty (size == 1). Empty tensors have size == 0.
         */
        bool empty() const {
            return (size_ == 0);
        }

        /**
         * @brief Returns the number of dimensions in the tensor
         */
        size_t rank() const {
            return shape_.size();
        }

        /**
         * @brief Check if tensor is in a valid state (not moved-from)
         */
        bool isValid() const noexcept {
            return true; // FIXME: Do we need a moved from state? device_id_;
        }

        // ====================================================================
        // Data Pointers
        // ====================================================================
       
        /**
         * @brief Returns type-safe pointer to tensor data with concrete host type
         *
         * Only available for host-accessible memory resources.
         */
        [[nodiscard]] constexpr auto* data() noexcept requires TMemoryResource::is_host_accessible {
            using HostType = typename TensorHostTypeMap<TDataType>::host_type;
            return static_cast<HostType*>(buffer_ ? buffer_->data() : nullptr);
        }

        /**
         * @brief Returns type-safe immutable pointer to tensor data with concrete host type
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
         * The new shape must have the same total number of elements as the current shape,
         * unless the tensor is empty.
         */
        void reshape( const shape_t& new_shape ) {
            int64_t new_size = computeSize( new_shape );
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
         * No-op for scalars and vectors.
         */
        Tensor<TDataType, TMemoryResource>& flatten() 
        {
            if (rank() <= 1)
            {
                return *this;
            }

            int64_t first = 1;
            
            for (int64_t i = 0; i + 1 < shape_.size(); ++i)
            {
                first *= shape_[i];
            }
            
            int64_t last = shape_.back();

            this->reshape( shape_t{ first, last } );

            return *this;
        }

        // ====================================================================
        // Identity and Metadata
        // ====================================================================

        /**
         * @brief Returns the tensor's unique identifier
         */
        std::string getUId() const {
            return uid_;
        }

        /**
         * @brief Returns the tensor's optional user-assigned name
         */
        std::string getName() const {
            return name_;
        }

        /**
         * @brief Assigns a descriptive name to the tensor
         *
         * Names must be non-empty strings.
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
         * @brief Generates string representation of the tensor
         *
         * Includes UID, name, shape, data type and device information.
         */
        std::string toString( bool showBuffer = false ) const {
            std::ostringstream oss;
            oss << "Tensor: " << uid_;
            if (!name_.empty())
                oss << "::" << name_;
            oss << ", ";
            oss << outputLayout();

            oss << " Type: " << DataTypeTraits::type_name;

            oss << " Device: " << device_id_.toString();
            
            oss << std::endl;

            if (showBuffer) {
                oss << getBufferString();
            }

            return oss.str();
        }

    //protected:

		// TJT: Review: Should these be protected or public? 

        /**
         * @brief Returns raw pointer to tensor data (implements ITensor protected API)
         */
        void* rawData() override {
			// TJT: Review: can buffer_ be null here?
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
         */
        Compute::MemoryResource* getMemoryResource() const override {
            return buffer_ ? buffer_->getMemoryResource() : nullptr;
        }

    private:
        
        Compute::DeviceId device_id_;       ///< DeviceId for proper device binding and resource management
        
        std::string uid_;                                                          ///< Unique identifier for this tensor instance
        std::string name_;                                                         ///< Optional user-assigned name for debugging
        size_t size_{ 0 };                                                         ///< Total number of logical elements in the tensor
        std::vector<int64_t> shape_{};                                              ///< Dimensional sizes for each tensor dimension
        std::vector<int64_t> strides_{};                                            ///< Memory stride values for multi-dimensional indexing
        std::shared_ptr<TensorBuffer<TDataType, TMemoryResource>> buffer_{ nullptr }; ///< Managed buffer containing tensor data

        // ====================================================================
        // Private Helper Methods
        // ====================================================================

        /**
         * @brief Validates device id against the memory resource requirement
         *
         * Throws if types do not match.
         */
        static Compute::DeviceId validateDeviceId( Compute::DeviceId device_id )
        {
            constexpr Compute::DeviceType required_type = TMemoryResource::device_type;

            if (device_id.type != required_type)
            {
                throw std::runtime_error(
                    "Device type mismatch: Memory resource requires " +
                    deviceTypeToString( required_type ) +
                    " but provided device is " +
                    deviceTypeToString( device_id.type )
                );
            }

            return device_id;
        }

        /**
         * @brief Allocates and initializes the tensor's data buffer
         *
         * Scalars allocate normally (size==1); truly empty tensors allocate nothing.
         */
        void allocateBuffer() {
            if (size_ > 0) {
                buffer_ = std::make_shared<TensorBuffer<TDataType, TMemoryResource>>( device_id_.index, size_);
            }
        }

        /**
         * @brief Validates multi-dimensional indices against tensor shape
         */
        void validateIndices( const std::vector<int64_t>& indices, const char* fn ) const {
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
         */
        int64_t computeFlatIndex( const std::vector<int64_t>& indices ) const {
            int64_t idx = 0;

            for (size_t d = 0; d < indices.size(); ++d) {
                idx += indices[d] * strides_[d];
            }
            
            return idx;
        }

        /**
         * @brief Generates formatted layout information string
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
         */
        int64_t computeSize( const std::vector<int64_t>& shape ) {
            // Product of empty sequence is 1 (multiplicative identity) for scalar construction
            return std::accumulate( shape.begin(), shape.end(), 1ull, std::multiplies<int64_t>() );
        }

        /**
         * @brief Computes row-major memory strides from shape
         */
        std::vector<int64_t> computeStrides(const std::vector<int64_t>& shape) {
            std::vector<int64_t> strides(shape.size(), 1);

            if (shape.empty()) {
                return strides;
            }

            for (int64_t i = shape.size() - 1; i > 0; --i) {
                strides[i - 1] = strides[i] * shape[i];
            }

            return strides;
        }

        /**
         * @brief Generates unique identifier for new tensor instance
         */
        std::string setUId() {
            return "tensor_" + std::to_string(UniqueIdGenerator::getNextId());
        }

        /**
         * @brief Gets formatted buffer content for string representation
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
     */
    export template <TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    std::ostream& operator<<(std::ostream& os, const Tensor<TDataType, TMemoryResource>& tensor) {
        os << tensor.toString();
        return os;
    }

    /**
     * @brief Host tensor alias
     */
    export template <TensorDataType TDataType>
        using HostTensor = Tensor<TDataType, Compute::CpuMemoryResource>;

    /**
     * @brief Device tensor alias
     */
    export template <TensorDataType TDataType>
        using DeviceTensor = Tensor<TDataType, Compute::CudaDeviceMemoryResource>;

    /**
     * @brief Pinned tensor alias
     */
    export template <TensorDataType TDataType>
        using PinnedTensor = Tensor<TDataType, Compute::CudaPinnedMemoryResource>;

    /**
     * @brief Universal (managed) tensor alias
     */
    export template <TensorDataType TDataType>
        using UniversalTensor = Tensor<TDataType, Compute::CudaManagedMemoryResource>;
}