module;
#include <vector>  
#include <iostream>
#include <sstream>
#include <iomanip>
#include <variant>  
#include <memory>
#include <mdspan>
#include <type_traits>
#include <cuda_fp16.h>
#include <atomic>
#include <string>
#include <stdexcept>
#include <numeric>

export module Dnn.Tensor;

import Dnn.TensorType;  
import Dnn.TensorBuffer; 
import Dnn.TensorTraits;

import Compute.ComputeDevice;
import Compute.DeviceContext;

import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Compute.CudaManagedMemoryResource;
import Compute.CudaPinnedMemoryResource;

namespace Mila::Dnn
{
	class UniqueIdGenerator {
	public:
		static size_t getNextId() {
			return counter_.fetch_add( 1, std::memory_order_relaxed );
		}

	private:
		static std::atomic<size_t> counter_;
	};

	std::atomic<size_t> UniqueIdGenerator::counter_{ 0 };

	export template<typename T, typename TMemoryResource = Compute::DeviceMemoryResource>
		requires ValidTensorType<T> && std::is_base_of_v<Compute::MemoryResource, TMemoryResource>
	class Tensor {
	public:
		using scalar_t = std::variant<int64_t, int32_t, half, float>;

		using Extent1d = std::dextents<size_t, 1>;
		using Extent2d = std::dextents<size_t, 2>;
		using Extent3d = std::dextents<size_t, 3>;
		using Extent4d = std::dextents<size_t, 4>;

        /**
        * @brief Constructs a tensor with the given shape and initializes it with the specified value.
        *
        * This constructor initializes the tensor with the provided shape and fills it with the given value.
        * If no value is provided, the tensor is initialized with the default value of the type T.
        *
        * @param shape The shape of the tensor.
        * @param value The value to initialize the tensor with. Defaults to the default value of type T.
        */
		Tensor( const std::vector<size_t>& shape, T value = T{} )
			: uid_{ set_uid() }, shape_( shape ), strides_( computeStrides( shape ) ), size_( computeSize( shape ) ) {
			allocateBuffer( value );
		}

		/**
		* @brief Constructs a tensor with the given shape and a shared pointer to allocated memory.
		*
		* This constructor initializes the tensor with the provided shape and uses the given shared pointer to allocated memory.
		* The tensor does not take ownership of the memory.
		*
		* @param shape The shape of the tensor.
		* @param data_ptr Shared pointer to the allocated memory.
		*/
		Tensor( const std::vector<size_t>& shape, std::shared_ptr<T> data_ptr )
			: uid_{ set_uid() }, shape_( shape ), strides_( computeStrides( shape ) ), size_( computeSize( shape ) ), data_ptr_( data_ptr ) {
			if ( !data_ptr_ ) {
				throw std::invalid_argument( "data_ptr cannot be null." );
			}
			buffer_ = std::make_shared<TensorBuffer<T, TMemoryResource>>( size_, data_ptr_.get() );
			data_type_ = tensor_type_of( data_ptr_.get() );
		}

		Tensor()
			: uid_{ set_uid() }, shape_(), strides_( computeStrides( shape_ ) ), size_() {
			allocateBuffer( T{} );
		}

		/**
		* @brief Converts this tensor to use a different memory resource.
		*
		* This method creates a new tensor with the same shape and data but using a different memory resource.
		* The data is copied from the current tensor to the new tensor using the appropriate memory transfer
		* mechanism based on the source and destination memory resource types.
		*
		* @tparam TNewMR The target memory resource type.
		* @return Tensor<T, TNewMR> A new tensor with the specified memory resource type.
		* @throws std::runtime_error If a CUDA memory transfer operation fails.
		*/
		template<typename TNewMR>
		Tensor<T, TNewMR> to() const {
			static_assert(std::is_base_of_v<Compute::MemoryResource, TNewMR>,
				"NewTMemoryResource must be derived from Compute::MemoryResource");

			Tensor<T, TNewMR> new_tensor( shape_ );

			if ( !name_.empty() ) {
				new_tensor.setName( name_ );
			}

			if ( size_ == 0 ) {
				return new_tensor;
			}

			// Determine the appropriate copy method based on memory resource types
			cudaError_t status = cudaSuccess;

			if constexpr ( is_device_accessible<TMemoryResource>() && is_device_accessible<TNewMR>() ) {
				status = cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyDeviceToDevice );
			}
			else if constexpr ( is_host_accessible<TMemoryResource>() && is_device_accessible<TNewMR>() ) {
				status = cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyHostToDevice );
			}
			else if constexpr ( is_device_accessible<TMemoryResource>() && is_host_accessible<TNewMR>() ) {
				status = cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyDeviceToHost );
			}
			else {
				// Host to host transfer (use standard copy)
				std::copy( this->data(), this->data() + size_, new_tensor.data() );
				return new_tensor;
			}

			// Check for CUDA errors
			if ( status != cudaSuccess ) {
				throw std::runtime_error( "CUDA memory transfer failed: " +
					std::string( cudaGetErrorString( status ) ) );
			}

			return new_tensor;
		}

		template<typename HostAccessibleMR = Compute::HostMemoryResource>
		Tensor<T, HostAccessibleMR> toHostAccessible() const {
			if constexpr ( std::is_same_v<TMemoryResource, Compute::HostMemoryResource> ||
				std::is_same_v<TMemoryResource, Compute::PinnedMemoryResource> ||
				std::is_same_v<TMemoryResource, Compute::ManagedMemoryResource> ) {
				// Create a shallow copy if the memory is already host-accessible
				Tensor<T, TMemoryResource> result( *this );
				return result.template to<HostAccessibleMR>();
			}
			else {
				// Create a new host-accessible tensor and copy the data
				return this->template to<HostAccessibleMR>();
			}
		}

		T at( const std::vector<size_t>& indices ) const {
			validateIndices( indices, "at()" );

			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				size_t index = computeIndex( indices );
				return buffer_->data()[ index ];
			}
			else {
				auto host_tensor = to<Compute::HostMemoryResource>();
				return host_tensor.at( indices );
			}
		}

		void set( const std::vector<size_t>& indices, T value ) {
			validateIndices( indices, "set()" );

			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				size_t index = computeIndex( indices );
				buffer_->data()[ index ] = value;
			}
			else {
				auto host_tensor = to<Compute::HostMemoryResource>();
				host_tensor.set( indices, value );
				*this = host_tensor.template to<TMemoryResource>();
			}
		}

		/**
		* @brief Copies data from another tensor into this tensor.
		*
		* This method copies the contents of the source tensor to this tensor. Both tensors must have
		* the same shape. The data is copied using the appropriate memory transfer mechanism based on
		* the source and destination memory resource types.
		*
		* @tparam SrcMemoryResource The memory resource type of the source tensor.
		* @param src The source tensor to copy data from.
		* @throws std::runtime_error If the shapes don't match or if a CUDA memory transfer operation fails.
		*/
		template<typename SrcMemoryResource>
		void copyFrom( const Tensor<T, SrcMemoryResource>& src ) {
			if ( shape_ != src.shape() ) {
				throw std::runtime_error( "Cannot copy from tensor with different shape." );
			}

			if ( size_ == 0 ) {
				return;
			}

			// Determine the appropriate copy method based on memory resource types
			if constexpr ( is_device_accessible<TMemoryResource>() && is_device_accessible<SrcMemoryResource>() ) {
				cudaError_t status = cudaMemcpy( data(), src.data(), size_ * sizeof( T ), cudaMemcpyDeviceToDevice );
				if ( status != cudaSuccess ) {
					throw std::runtime_error( "CUDA memory transfer failed: " +
						std::string( cudaGetErrorString( status ) ) );
				}
			}
			else if constexpr ( is_host_accessible<TMemoryResource>() && is_device_accessible<SrcMemoryResource>() ) {
				// Host destination, device source - need to bring to host first
				auto host_src = src.template to<Compute::HostMemoryResource>();
				std::copy( host_src.data(), host_src.data() + size_, buffer_->data() );
			}
			else if constexpr ( is_device_accessible<TMemoryResource>() && is_host_accessible<SrcMemoryResource>() ) {
				// Device destination, host source
				cudaError_t status = cudaMemcpy( data(), src.data(), size_ * sizeof( T ), cudaMemcpyHostToDevice );
				if ( status != cudaSuccess ) {
					throw std::runtime_error( "CUDA memory transfer failed: " +
						std::string( cudaGetErrorString( status ) ) );
				}
			}
			else {
				// Host to host transfer (use standard copy)
				std::copy( src.data(), src.data() + size_, buffer_->data() );
			}

			// Copy name if source has one and this tensor doesn't
			if ( name_.empty() && !src.getName().empty() ) {
				setName( src.getName() );
			}
		}

		/**
		* @brief Creates a deep copy of this tensor.
		*
		* Unlike the copy constructor which shares the underlying buffer,
		* this method creates a completely independent copy with its own data buffer.
		*
		* @return Tensor<T, TMemoryResource> A deep copy of this tensor
		*/
		Tensor<T, TMemoryResource> clone() const {
			// Create a new tensor with the same shape
			Tensor<T, TMemoryResource> result( shape_ );

			// Copy data from the current tensor to the new tensor
			if ( size_ > 0 ) {
				if constexpr ( is_host_accessible<TMemoryResource>() ) {
					std::copy( data(), data() + size_, result.data() );
				}
				else {
					cudaMemcpy( result.data(), data(), size_ * sizeof( T ), cudaMemcpyDeviceToDevice );
				}
			}

			if ( !name_.empty() ) {
				result.setName( name_ );
			}

			return result;
		}

		void reshape( const std::vector<size_t>& new_shape ) {
			size_t new_size = computeSize( new_shape );
			if ( !empty() && (new_size != size_) ) {
				throw std::runtime_error( "The new shape must match the size of the tensor or the tensor must be empty." );
			}

			shape_ = new_shape;
			strides_ = computeStrides( new_shape );

			if ( empty() ) {
				buffer_->resize( new_size );
				size_ = new_size;
			}
		}

		auto vectorSpan() {
			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				return std::mdspan<T, Extent1d>( buffer_->data(), size_ );
			}
			else {
				throw std::runtime_error( "vectorSpan() requires host-accessible memory. Use to<CpuMemoryResource>() first." );
			}
		}

		auto matrixSpan( const std::vector<size_t>& shape ) {
			if ( shape.size() != 2 ) {
				throw std::runtime_error( "matrixSpan: The shape must have exactly 2 dimensions." );
			}
			size_t total_size = shape[ 0 ] * shape[ 1 ];
			if ( total_size > size_ ) {
				throw std::runtime_error( "matrixSpan: The specified shape exceeds the tensor size." );
			}

			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				return std::mdspan<T, Extent2d>( buffer_->data(), shape[ 0 ], shape[ 1 ] );
			}
			else {
				throw std::runtime_error( "matrixSpan() requires host-accessible memory. Use to<CpuMemoryResource>() first." );
			}
		}

		template<typename... Args>
		T& operator[]( Args... args ) {
			static_assert(sizeof...(args) > 0, "operator[]: At least one index must be provided.");
			const size_t num_args = sizeof...(args);
			if ( num_args != shape_.size() ) {
				throw std::runtime_error( "operator[]: Number of indices must match the tensor rank." );
			}

			// Validate indices are within bounds
			std::vector<size_t> indices = { static_cast<size_t>(args)... };

			for ( size_t i = 0; i < indices.size(); ++i ) {
				if ( indices[ i ] >= shape_[ i ] ) {
					throw std::out_of_range( "operator[]: Index " + std::to_string( indices[ i ] ) +
						" is out of range for dimension " + std::to_string( i ) +
						" with size " + std::to_string( shape_[ i ] ) );
				}
			}

			size_t index = computeIndex( { static_cast<size_t>(args)... } );
			if ( index >= size_ ) {
				throw std::out_of_range( "operator[]: Index out of range." );
			}

			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				return buffer_->data()[ index ];
			}
			else {
				throw std::runtime_error( "Direct tensor access requires host-accessible memory. Use to<CpuMemoryResource>() first." );
			}
		}

		template<typename... Args>
		const T& operator[]( Args... args ) const {
			static_assert(sizeof...(args) > 0, "operator[]: At least one index must be provided.");
			const size_t num_args = sizeof...(args);
			if ( num_args != shape_.size() ) {
				throw std::runtime_error( "operator[]: Number of indices must match the tensor rank." );
			}
			// Validate indices are within bounds
			std::vector<size_t> indices = { static_cast<size_t>(args)... };

			for ( size_t i = 0; i < indices.size(); ++i ) {
				if ( indices[ i ] >= shape_[ i ] ) {
					throw std::out_of_range( "operator[]: Index " + std::to_string( indices[ i ] ) +
						" is out of range for dimension " + std::to_string( i ) +
						" with size " + std::to_string( shape_[ i ] ) );
				}
			}

			size_t index = computeIndex( { static_cast<size_t>(args)... } );
			if ( index >= size_ ) {
				throw std::out_of_range( "operator[]: Index out of range." );
			}
			return buffer_->data()[ index ];
		}

		// Properties..
		const bool empty() const {
			return (size_ == 0);
		}

		const std::vector<size_t>& shape() const {
			return shape_;
		}

		const std::vector<size_t>& strides() const {
			return strides_;
		}

		size_t size() const {
			return size_;
		}

		size_t rank() const {
			return shape_.size();
		}

		T* data() {
			return buffer_->data();
		}

		const T* data() const {
			return buffer_->data();
		}

		void fill( const T& value ) {
			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				// Direct fill for host-accessible memory
				std::fill( buffer_->data(), buffer_->data() + size_, value );
			}
			else {
				// Create a temporary host tensor, fill it, then copy back
				auto host_tensor = to<Compute::HostMemoryResource>();
				host_tensor.fill( value );
				*this = host_tensor.template to<TMemoryResource>();
			}
		}

		template <typename MR = TMemoryResource>
		static constexpr bool is_host_accessible() {
			return MR::is_host_accessible;
		}

		template <typename MR = TMemoryResource>
		static constexpr bool is_device_accessible() {
			return MR::is_device_accessible;
		}

		std::string	getName() const {
			return name_;
		}

        void setName( const std::string& value ) {
            if ( value.empty() ) {
                throw std::invalid_argument("Tensor name cannot be empty.");
            }
            name_ = value;
        }

		std::string get_uid() const {
			return uid_;
		}

		std::string toString( bool showBuffer = false ) const {
			std::ostringstream oss;
			oss << "Tensor: " << uid_;
			if ( !name_.empty() ) oss << "::" << name_;
			oss << ", ";
			oss << "Shape: (";
			for ( size_t i = 0; i < shape_.size(); ++i ) {
				oss << shape_[ i ];
				if ( i != shape_.size() - 1 ) {
					oss << ",";
				}
			}
			oss << ")";
			oss << " Size: " << size_;
			oss << " Type: " << to_string( data_type_ ) << std::endl;

			if ( showBuffer ) {
				oss << getBufferString( 0, 0 );
			}

			return oss.str();
		}

		std::string getBufferString( size_t start_index = 0, size_t depth = 0 ) const {
			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				return outputBuffer( start_index, depth );
			}
			else {
				auto host_tensor = to<Compute::HostMemoryResource>();
				// Now we need a way to get buffer content without calling private method
				// We'll use a different approach: recursively get string representation
				return host_tensor.getBufferString( start_index, depth );
			}
		}

		friend std::ostream& operator<<( std::ostream& os, const Tensor& tensor ) {
			os << tensor.toString();
			return os;
		}

		/**
		* @brief Copy constructor (creates a shallow copy).
		*
		* This constructor creates a new tensor that shares the underlying data buffer with
		* the original tensor. Modifications to one tensor's data will affect the other.
		* For a deep, independent copy, use the clone() method instead.
		*
		 * @param other The tensor to copy from.
		*/
		Tensor( const Tensor& other )
			: uid_( other.uid_ ), name_( other.name_ ), shape_( other.shape_ ), strides_( other.strides_ ),
			size_( other.size_ ), data_type_( other.data_type_ ), buffer_( other.buffer_ ) {}
		
        /**
        * @brief Move constructor.
        *
        * This constructor moves the contents of the given tensor to this tensor.
        *
        * @param other The tensor to move from.
        */
		Tensor( Tensor&& other ) noexcept
			: uid_( std::move( other.uid_ ) ),
			name_( std::move( other.name_ ) ),
			size_( other.size_ ),
			data_type_( other.data_type_ ),
			shape_( std::move( other.shape_ ) ),
			strides_( std::move( other.strides_ ) ),
			buffer_( std::move( other.buffer_ ) ) {
			other.size_ = 0;
			other.data_type_ = TensorType::FP16;
		}

		/**
		* @brief Move assignment operator.
		*
		* This operator moves the contents of the given tensor to this tensor.
		*
		* @param other The tensor to move from.
		* @return Tensor& A reference to this tensor.
		*/
		Tensor& operator=( Tensor&& other ) noexcept {
			if ( this != &other ) {
				uid_ = std::move( other.uid_ );
				name_ = std::move( other.name_ );
				shape_ = std::move( other.shape_ );
				strides_ = std::move( other.strides_ );
				size_ = other.size_;
				data_type_ = other.data_type_;
				buffer_ = std::move( other.buffer_ );

				other.size_ = 0;
				other.data_type_ = TensorType::FP16;
			}
			return *this;
		}

		/**
		* @brief Copy assignment operator.
		*
		* This operator copies the contents of the given tensor to this tensor.
		*
		* @param other The tensor to copy from.
		* @return Tensor& A reference to this tensor.
		*/
		Tensor& operator=( const Tensor& other ) {
			if ( this != &other ) {
				uid_ = other.uid_;
				name_ = other.name_;
				shape_ = other.shape_;
				strides_ = other.strides_;
				size_ = other.size_;
				data_type_ = other.data_type_;
				buffer_ = other.buffer_;
			}
			return *this;
		}

		~Tensor() = default;

	private:
		std::string uid_;
		std::string name_;
		size_t size_{ 0 };
		TensorType data_type_;
		std::vector<size_t> shape_{};
		std::vector<size_t> strides_{};
		std::shared_ptr<TensorBuffer<T, TMemoryResource>> buffer_{ nullptr };
		std::shared_ptr<T> data_ptr_{ nullptr }; // Shared pointer to external memory (non-owning)

		void allocateBuffer( T value ) {
			buffer_ = std::make_shared<TensorBuffer<T, TMemoryResource>>( size_, value );
			data_type_ = tensor_type_of( buffer_->data() );
		}

        std::string outputBuffer( size_t index, size_t depth ) const {
            std::ostringstream oss;
            if ( depth == shape_.size() - 1 ) {
                for ( size_t i = 0; i < shape_[ depth ]; ++i ) {
                    if ( i < 3 || i >= shape_[ depth ] - 3 ) {
                        oss << std::setw( 10 ) << buffer_->data()[ index + i ] << " ";
                    }
                    else if ( i == 3 ) {
                        oss << "... ";
                    }
                }
            }
            else {
                for ( size_t i = 0; i < shape_[ depth ]; ++i ) {
                    if ( i < 3 || i >= shape_[ depth ] - 3 ) {
                        oss << "[ ";
                        oss << outputBuffer( index + i * shape_[ depth + 1 ], depth + 1 );
                        oss << "]" << std::endl;
                    }
                    else if ( i == 3 ) {
                        oss << "[ ... ]" << std::endl;
                        i = shape_[ depth ] - 4;
                    }
                }
            }
            return oss.str();
        }
        
		static std::vector<size_t> computeStrides( const std::vector<size_t>& shape ) {
			std::vector<size_t> strides( shape.size(), 1 );

			if ( shape.empty() ) {
				return strides;
			}

			for ( int i = shape.size() - 2; i >= 0; --i ) {
				strides[ i ] = strides[ i + 1 ] * shape[ i + 1 ];
			}

			return strides;
		}

        static size_t computeSize(const std::vector<size_t>& shape) {
           if (shape.empty()) {
               return 0;
           }
           return std::accumulate(shape.begin(), shape.end(), 1ull, std::multiplies<size_t>());
        }

		size_t computeIndex( const std::vector<size_t>& indices ) const {
			size_t index = 0;
			for ( size_t i = 0; i < indices.size(); ++i ) {
				index += indices[ i ] * strides_[ i ];
			}
			return index;
		}
		// Inside the Tensor class

		void validateIndices( const std::vector<size_t>& indices, const std::string& method_name ) const {
			if ( indices.size() != shape_.size() ) {
				throw std::runtime_error( method_name + ": Number of indices must match the tensor rank." );
			}

			for ( size_t i = 0; i < indices.size(); ++i ) {
				if ( indices[ i ] >= shape_[ i ] ) {
					throw std::out_of_range( method_name + ": Index " + std::to_string( indices[ i ] ) +
						" is out of range for dimension " + std::to_string( i ) +
						" with size " + std::to_string( shape_[ i ] ) );
				}
			}
		}

		std::string set_uid() {
			return "tensor_" + std::to_string( UniqueIdGenerator::getNextId() );
		}
	};

    /**
    * @brief Tensor type that uses host (CPU) memory.
    *
    * HostTensor is a specialized tensor that allocates and stores data in regular
    * host memory that is accessible by the CPU. This is suitable for data that 
    * needs to be frequently accessed by the host and doesn't require GPU acceleration.
    *
    * @tparam T The data type of the tensor elements.
    */
    export template <typename T>
    using HostTensor = Tensor<T, Compute::HostMemoryResource>;

    /**
    * @brief Tensor type that uses device (GPU) memory.
    *
    * DeviceTensor is a specialized tensor that allocates and stores data in GPU memory.
    * This type is optimized for operations performed on the GPU and provides the best
    * performance for CUDA accelerated computations, but the data is not directly
    * accessible from CPU code.
    *
    * @tparam T The data type of the tensor elements.
    */
    export template <class T>
    using DeviceTensor = Tensor<T, Compute::DeviceMemoryResource>;

    /**
    * @brief Tensor type that uses pinned (page-locked) host memory.
    *
    * PinnedTensor is a specialized tensor that allocates and stores data in pinned memory,
    * which is host memory that is locked to prevent paging. This memory type enables faster
    * transfers between CPU and GPU compared to regular host memory, but consumes a limited
    * resource that should be used judiciously.
    *
    * @tparam T The data type of the tensor elements.
    */
    export template <class T>
    using PinnedTensor = Tensor<T, Compute::PinnedMemoryResource>;

    /**
    * @brief Tensor type that uses CUDA managed memory accessible from both CPU and GPU.
    *
    * UniversalTensor is a specialized tensor that allocates and stores data in CUDA managed memory,
    * which provides a single memory space accessible by both CPU and GPU. The CUDA runtime
    * automatically migrates data between host and device as needed. This offers programming
    * convenience at the cost of potentially lower performance compared to explicitly managed memory.
    *
    * @tparam T The data type of the tensor elements.
    */
    export template <class T>
    using UniversalTensor = Tensor<T, Compute::ManagedMemoryResource>;
}