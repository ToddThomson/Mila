module;
#include <vector>  
#include <iostream>
#include <sstream>
#include <iomanip>
#include <variant>  
#include <memory>
//#include <mdspan>
#include <type_traits>
#include <cuda_fp16.h>
#include <atomic>
#include <string>
#include <stdexcept>
#include <numeric>

export module Dnn.Tensor;

import Dnn.TensorType;  
import Dnn.TensorBuffer; 
import Dnn.TensorPtr;
import Dnn.TensorTraits;

import Compute.ComputeDevice;
import Compute.DeviceContext;

import Compute.MemoryResource;
import Compute.DynamicMemoryResource;
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

	export template<typename TElementType, typename TMemoryResource = Compute::CudaMemoryResource>
		requires ValidTensorType<TElementType> && std::is_base_of_v<Compute::MemoryResource, TMemoryResource>
	class Tensor {
	public:
		using MR = TMemoryResource;
		using scalar_t = std::variant<int64_t, int32_t, half, float>;

		/*using Extent1d = std::dextents<size_t, 1>;
		using Extent2d = std::dextents<size_t, 2>;
		using Extent3d = std::dextents<size_t, 3>;
		using Extent4d = std::dextents<size_t, 4>;*/

        /**
        * @brief Constructs a tensor with the given shape and initializes it with the specified value.
        *
        * This constructor initializes the tensor with the provided shape and fills it with the given value.
        * If no value is provided, the tensor is initialized with the default value of the type TElementType.
        *
        * @param shape The shape of the tensor.
        * @param value The value to initialize the tensor with. Defaults to the default value of type TElementType.
        */
		explicit Tensor( const std::vector<size_t>& shape, TElementType value = TElementType{} )
			: uid_{ set_uid() }, shape_( shape ), strides_( computeStrides( shape ) ), size_( computeSize( shape ) ) {
			allocateBuffer( value );
		}

		/*Tensor( std::initializer_list<size_t> shape, TElementType value = TElementType{} )
			: Tensor( std::vector<size_t>( shape ), value ) {}*/

		/**
		* @brief Constructs a tensor with the given shape and a shared pointer to allocated memory.
		*
		* This constructor initializes the tensor with the provided shape and uses the given shared pointer to allocated memory.
		* The tensor does not take ownership of the memory.
		*
		* @param shape The shape of the tensor.
		* @param data_ptr Shared pointer to the allocated memory.
		*/
		Tensor( const std::vector<size_t>& shape, std::shared_ptr<TElementType> data_ptr )
			: uid_{ set_uid() }, shape_( shape ), strides_( computeStrides( shape ) ), size_( computeSize( shape ) ), external_memory_ptr_( data_ptr ) {
			if ( !external_memory_ptr_ ) {
				throw std::invalid_argument( "data_ptr cannot be null." );
			}
			buffer_ = std::make_shared<TensorBuffer<TElementType, TMemoryResource>>( size_, external_memory_ptr_.get() );
			//data_type_ = tensor_type_of( external_memory_ptr_.get() );
		}

		Tensor()
			: uid_{ set_uid() }, shape_(), strides_( computeStrides( shape_ ) ), size_() {
			allocateBuffer( TElementType{} );
		}

		/**
		* @brief Converts this tensor to use a different memory resource.
		*
		* This method creates a new tensor with the same shape and data but using a different memory resource.
		* The data is copied from the current tensor to the new tensor using the appropriate memory transfer
		* mechanism based on the source and destination memory resource types.
		*
		* @tparam TNewMR The target memory resource type.
		* @return Tensor<TElementType, TNewMR> A new tensor with the specified memory resource type.
		* @throws std::runtime_error If a CUDA memory transfer operation fails.
		*/
		template<typename TNewMR>
		Tensor<TElementType, TNewMR> to() const {
			static_assert(std::is_base_of_v<Compute::MemoryResource, TNewMR>,
				"NewTMemoryResource must be derived from Compute::MemoryResource");

			Tensor<TElementType, TNewMR> new_tensor( shape_ );

			if ( !name_.empty() ) {
				new_tensor.setName( name_ );
			}

			if ( size_ == 0 ) {
				return new_tensor;
			}

			// Determine the appropriate copy method based on memory resource types
			cudaError_t status = cudaSuccess;

			if constexpr ( is_device_accessible<TMemoryResource>() && is_device_accessible<TNewMR>() ) {
				status = cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( TElementType ), cudaMemcpyDeviceToDevice );
			}
			else if constexpr ( is_host_accessible<TMemoryResource>() && is_device_accessible<TNewMR>() ) {
				status = cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( TElementType ), cudaMemcpyHostToDevice );
			}
			else if constexpr ( is_device_accessible<TMemoryResource>() && is_host_accessible<TNewMR>() ) {
				status = cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( TElementType ), cudaMemcpyDeviceToHost );
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
		Tensor<TElementType, HostAccessibleMR> toHostAccessible() const {
			if constexpr ( std::is_same_v<TMemoryResource, Compute::HostMemoryResource> ||
				std::is_same_v<TMemoryResource, Compute::CudaPinnedMemoryResource> ||
				std::is_same_v<TMemoryResource, Compute::CudaManagedMemoryResource> ) {
				// Create a shallow copy if the memory is already host-accessible
				Tensor<TElementType, TMemoryResource> result( *this );
				return result.template to<HostAccessibleMR>();
			}
			else {
				// Create a new host-accessible tensor and copy the data
				return this->template to<HostAccessibleMR>();
			}
		}

		TElementType at( const std::vector<size_t>& indices ) const {
			validateIndices( indices, "at()" );

			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				size_t index = computeIndex( indices );
				return buffer_->data()[ index ];
			}
			else {
				// Get a single element directly instead of copying the entire tensor
				TElementType result;
				size_t index = computeIndex( indices );

				cudaError_t status = cudaMemcpy( &result, buffer_->data() + index,
					sizeof( TElementType ), cudaMemcpyDeviceToHost );

				if ( status != cudaSuccess ) {
					throw std::runtime_error( "CUDA memory transfer failed in at(): " +
						std::string( cudaGetErrorString( status ) ) );
				}

				return result;
			}
		}

		void set( const std::vector<size_t>& indices, TElementType value ) {
			validateIndices( indices, "set()" );

			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				size_t index = computeIndex( indices );
				buffer_->data()[ index ] = value;
			}
			else {
				// Set a single element directly instead of copying the entire tensor
				size_t index = computeIndex( indices );

				cudaError_t status = cudaMemcpy( buffer_->data() + index, &value,
					sizeof( TElementType ), cudaMemcpyHostToDevice );

				if ( status != cudaSuccess ) {
					throw std::runtime_error( "CUDA memory transfer failed in set(): " +
						std::string( cudaGetErrorString( status ) ) );
				}
			}
		}

		/**
		* @brief Converts a float tensor to half precision.
		*
		* This method creates a new tensor with the same shape but with half precision elements.
		* The conversion is performed in a type-safe manner, handling both host and device memory appropriately.
		*
		* @tparam T The original element type (must be float)
		* @tparam TMR The memory resource type
		* @return Tensor<half, TMR> A new tensor with half precision values
		* @throws std::runtime_error If the source tensor is not of float type or if a CUDA operation fails
		*/
		template<typename T = TElementType, typename TMR = TMemoryResource>
		std::enable_if_t<std::is_same_v<T, float>, Tensor<half, TMR>> toHalf() const {
			Tensor<half, TMR> result( shape_ );

			if ( !name_.empty() ) {
				result.setName( name_ + "_half" );
			}

			if ( size_ == 0 ) {
				return result;
			}

			// For host-accessible memory, perform explicit conversion
			if constexpr ( is_host_accessible<TMR>() ) {
				// Create temporary host tensors if needed
				if constexpr ( !is_host_accessible<TMemoryResource>() ) {
					auto host_tensor = this->template to<Compute::HostMemoryResource>();
					auto host_result = Tensor<half, Compute::HostMemoryResource>( shape_ );

					// Convert float to half on host
					for ( size_t i = 0; i < size_; ++i ) {
						host_result.raw_data()[ i ] = __float2half( host_tensor.raw_data()[ i ] );
					}

					// Copy back to device if needed
					result.copyFrom( host_result );
				}
				else {
					// Direct host-to-host conversion
					for ( size_t i = 0; i < size_; ++i ) {
						result.raw_data()[ i ] = __float2half( this->raw_data()[ i ] );
					}
				}
			}
			// For device memory, use optimized CUDA conversion
			else {
				cudaError_t status = cudaSuccess;

				// If the source is not already on device, we need to bring it there first
				if constexpr ( !is_device_accessible<TMemoryResource>() ) {
					auto device_tensor = this->template to<Compute::CudaMemoryResource>();
					// TODO: Use CUDA kernel for float to half conversion
					// For now, we'll use a host intermediary
					auto host_tensor = device_tensor.template to<Compute::HostMemoryResource>();
					auto host_result = Tensor<half, Compute::HostMemoryResource>( shape_ );

					// Convert float to half on host
					for ( size_t i = 0; i < size_; ++i ) {
						host_result.raw_data()[ i ] = __float2half( host_tensor.raw_data()[ i ] );
					}

					// Copy back to device
					result.copyFrom( host_result );
				}
				else {
					// Both source and destination are on device
					// TODO: Use CUDA kernel for float to half conversion
					// For now, fallback to host intermediary
					auto host_tensor = this->template to<Compute::HostMemoryResource>();
					auto host_result = Tensor<half, Compute::HostMemoryResource>( shape_ );

					// Convert float to half on host
					for ( size_t i = 0; i < size_; ++i ) {
						host_result.raw_data()[ i ] = __float2half( host_tensor.raw_data()[ i ] );
					}

					// Copy back to device
					result.copyFrom( host_result );
				}

				if ( status != cudaSuccess ) {
					throw std::runtime_error( "CUDA float to half conversion failed: " +
						std::string( cudaGetErrorString( status ) ) );
				}
			}

			return result;
		}

		/**
		* @brief Converts a half precision tensor to float.
		*
		* This method creates a new tensor with the same shape but with float precision elements.
		* The conversion is performed in a type-safe manner, handling both host and device memory appropriately.
		*
		* @tparam T The original element type (must be half)
		* @tparam TMR The memory resource type
		* @return Tensor<float, TMR> A new tensor with float precision values
		* @throws std::runtime_error If the source tensor is not of half type or if a CUDA operation fails
		*/
		template<typename T = TElementType, typename TMR = TMemoryResource>
		std::enable_if_t<std::is_same_v<T, half>, Tensor<float, TMR>> toFloat() const {
			Tensor<float, TMR> result( shape_ );

			if ( !name_.empty() ) {
				result.setName( name_ + "_float" );
			}

			if ( size_ == 0 ) {
				return result;
			}

			// For host-accessible memory, perform explicit conversion
			if constexpr ( is_host_accessible<TMR>() ) {
				// Create temporary host tensors if needed
				if constexpr ( !is_host_accessible<TMemoryResource>() ) {
					auto host_tensor = this->template to<Compute::HostMemoryResource>();
					auto host_result = Tensor<float, Compute::HostMemoryResource>( shape_ );

					// Convert half to float on host
					for ( size_t i = 0; i < size_; ++i ) {
						host_result.raw_data()[ i ] = __half2float( host_tensor.raw_data()[ i ] );
					}

					// Copy back to device if needed
					result.copyFrom( host_result );
				}
				else {
					// Direct host-to-host conversion
					for ( size_t i = 0; i < size_; ++i ) {
						result.raw_data()[ i ] = __half2float( this->raw_data()[ i ] );
					}
				}
			}
			// For device memory, use optimized CUDA conversion
			else {
				cudaError_t status = cudaSuccess;

				// If the source is not already on device, we need to bring it there first
				if constexpr ( !is_device_accessible<TMemoryResource>() ) {
					auto device_tensor = this->template to<Compute::CudaMemoryResource>();
					// TODO: Use CUDA kernel for half to float conversion
					// For now, we'll use a host intermediary
					auto host_tensor = device_tensor.template to<Compute::HostMemoryResource>();
					auto host_result = Tensor<float, Compute::HostMemoryResource>( shape_ );

					// Convert half to float on host
					for ( size_t i = 0; i < size_; ++i ) {
						host_result.raw_data()[ i ] = __half2float( host_tensor.raw_data()[ i ] );
					}

					// Copy back to device
					result.copyFrom( host_result );
				}
				else {
					// Both source and destination are on device
					// TODO: Use CUDA kernel for half to float conversion
					// For now, fallback to host intermediary
					auto host_tensor = this->template to<Compute::HostMemoryResource>();
					auto host_result = Tensor<float, Compute::HostMemoryResource>( shape_ );

					// Convert half to float on host
					for ( size_t i = 0; i < size_; ++i ) {
						host_result.raw_data()[ i ] = __half2float( host_tensor.raw_data()[ i ] );
					}

					// Copy back to device
					result.copyFrom( host_result );
				}

				if ( status != cudaSuccess ) {
					throw std::runtime_error( "CUDA half to float conversion failed: " +
						std::string( cudaGetErrorString( status ) ) );
				}
			}

			return result;
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
		void copyFrom( const Tensor<TElementType, SrcMemoryResource>& src ) {
			if ( shape_ != src.shape() ) {
				throw std::runtime_error( "Cannot copy from tensor with different shape." );
			}

			if ( size_ == 0 ) {
				return;
			}

			// Determine the appropriate copy method based on memory resource types
			if constexpr ( is_device_accessible<TMemoryResource>() && is_device_accessible<SrcMemoryResource>() ) {
				// Device to device transfer - use raw pointers for CUDA operations
				cudaError_t status = cudaMemcpy( raw_data(), src.raw_data(),
					size_ * sizeof( TElementType ),
					cudaMemcpyDeviceToDevice );
				if ( status != cudaSuccess ) {
					throw std::runtime_error( "CUDA memory transfer failed: " +
						std::string( cudaGetErrorString( status ) ) );
				}
			}
			else if constexpr ( is_host_accessible<TMemoryResource>() && is_device_accessible<SrcMemoryResource>() ) {
				// Host destination, device source - need to bring to host first
				auto host_src = src.template to<Compute::HostMemoryResource>();
				std::copy( host_src.raw_data(), host_src.raw_data() + size_, raw_data() );
			}
			else if constexpr ( is_device_accessible<TMemoryResource>() && is_host_accessible<SrcMemoryResource>() ) {
				// Device destination, host source
				cudaError_t status = cudaMemcpy( raw_data(), src.raw_data(),
					size_ * sizeof( TElementType ),
					cudaMemcpyHostToDevice );
				if ( status != cudaSuccess ) {
					throw std::runtime_error( "CUDA memory transfer failed: " +
						std::string( cudaGetErrorString( status ) ) );
				}
			}
			else {
				// Host to host transfer (use standard copy)
				std::copy( src.raw_data(), src.raw_data() + size_, raw_data() );
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
		* @return Tensor<TElementType, TMemoryResource> A deep copy of this tensor
		*/
		Tensor<TElementType, TMemoryResource> clone() const {
			// Create a new tensor with the same shape
			Tensor<TElementType, TMemoryResource> result( shape_ );

			// Copy data from the current tensor to the new tensor
			if ( size_ > 0 ) {
				if constexpr ( is_host_accessible<TMemoryResource>() ) {
					std::copy( data(), data() + size_, result.data() );
				}
				else {
					cudaMemcpy( result.data(), data(), size_ * sizeof( TElementType ), cudaMemcpyDeviceToDevice );
				}
			}

			if ( !name_.empty() ) {
				result.setName( name_ );
			}

			return result;
		}

		/**
		 * @brief Flattens the tensor to a 2D tensor by combining all dimensions except the last one.
		 *
		 * This method reshapes the tensor from [D1, D2, ..., Dn-1, Dn] to [D1*D2*...*Dn-1, Dn].
		 * This is particularly useful for operations that treat all but the last dimension as batch dimensions,
		 * such as Fully Connected operations.
		 *
		 * @return A reference to this tensor after flattening.
		 */
		Tensor<TElementType, TMemoryResource>& flatten() {
			if ( shape().size() <= 1 ) {
				return *this; // Already flat or scalar
			}

			// Calculate the product of all dimensions except the last
			size_t flat_dim = 1;
			for ( size_t i = 0; i < shape().size() - 1; i++ ) {
				flat_dim *= shape()[ i ];
			}

			// The new shape is [flat_dim, last_dim]
			std::vector<size_t> new_shape = { flat_dim, shape().back() };
			reshape( new_shape );

			return *this;
		}

		/**
		 * @brief Creates a flattened copy of this tensor.
		 *
		 * Similar to flatten(), but returns a new tensor instead of modifying this one.
		 *
		 * @return A new tensor with flattened shape.
		 */
		Tensor<TElementType, TMemoryResource> flattened() const {
			if ( shape().size() <= 1 ) {
				return *this; // Already flat or scalar
			}

			// Calculate the product of all dimensions except the last
			size_t flat_dim = 1;
			for ( size_t i = 0; i < shape().size() - 1; i++ ) {
				flat_dim *= shape()[ i ];
			}

			// The new shape is [flat_dim, last_dim]
			std::vector<size_t> new_shape = { flat_dim, shape().back() };

			// Create a shallow copy and reshape it
			Tensor<TElementType, TMemoryResource> result( *this );
			result.reshape( new_shape );

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

		// TJT: std::mdspan is not available in the GNU C++ compiler so these convenience methods are commented out for now.

		/*auto vectorSpan() {
			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				return std::mdspan<TElementType, Extent1d>( buffer_->data(), size_ );
			}
			else {
				throw std::runtime_error( "vectorSpan() requires host-accessible memory. Use to<CpuMemoryResource>() first." );
			}
		}*/

		/*auto matrixSpan( const std::vector<size_t>& shape ) {
			if ( shape.size() != 2 ) {
				throw std::runtime_error( "matrixSpan: The shape must have exactly 2 dimensions." );
			}
			size_t total_size = shape[ 0 ] * shape[ 1 ];
			if ( total_size > size_ ) {
				throw std::runtime_error( "matrixSpan: The specified shape exceeds the tensor size." );
			}

			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				return std::mdspan<TElementType, Extent2d>( buffer_->data(), shape[ 0 ], shape[ 1 ] );
			}
			else {
				throw std::runtime_error( "matrixSpan() requires host-accessible memory. Use to<CpuMemoryResource>() first." );
			}
		}*/

		template<typename... Args>
		TElementType& operator[]( Args... args ) {
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
		const TElementType& operator[]( Args... args ) const {
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

			size_t index = computeIndex( { static_cast<size_t>( args )... } );
			if ( index >= size_ ) {
				throw std::out_of_range( "operator[]: Index out of range." );
			}

			// Add the same check as in the non-const version
			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				return buffer_->data()[ index ];
			}
			else {
				throw std::runtime_error( "Direct tensor access requires host-accessible memory. Use to<HostMemoryResource>() first." );
			}
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

		/**
		* @brief Gets a pointer to the tensor data with memory type safety.
		*
		* Returns a memory-type-aware pointer that prevents unsafe host access
		* to device memory at compile time.
		*
		* @return A TensorPtr wrapper that enforces memory access safety
		*/
		auto data() {
			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				return HostPtr<TElementType>( buffer_->data() );
			}
			else {
				return DevicePtr<TElementType>( buffer_->data() );
			}
		}

		/**
		 * @brief Gets a const pointer to the tensor data with memory type safety.
		 *
		 * Returns a memory-type-aware pointer that prevents unsafe host access
		 * to device memory at compile time.
		 *
		 * @return A const TensorPtr wrapper that enforces memory access safety
		 */
		auto data() const {
			if constexpr ( is_host_accessible<TMemoryResource>() ) {
				return HostPtr<const TElementType>( buffer_->data() );
			}
			else {
				return DevicePtr<const TElementType>( buffer_->data() );
			}
		}

		/**
		 * @brief Gets a raw pointer to the tensor data for use in CUDA kernels.
		 *
		 * This method is intended for internal use by CUDA operation implementations.
		 * Use with caution as it bypasses memory type safety.
		 *
		 * @return Raw pointer to the tensor data
		 */
		TElementType* raw_data() {
			return buffer_->data();
		}

		/**
		 * @brief Gets a const raw pointer to the tensor data for use in CUDA kernels.
		 *
		 * This method is intended for internal use by CUDA operation implementations.
		 * Use with caution as it bypasses memory type safety.
		 *
		 * @return Const raw pointer to the tensor data
		 */
		const TElementType* raw_data() const {
			return buffer_->data();
		}

		void fill( const TElementType& value ) {
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

		// The relevant part of the Tensor class that needs to be modified
		template <typename MR = TMemoryResource>
		static constexpr bool is_host_accessible() {
			if constexpr ( std::is_same_v<MR, Compute::DynamicMemoryResource> ) {
				// For DynamicMemoryResource, we need to create an instance to check
				// Since we don't have access to a runtime instance here, we'll make a conservative choice
				// using std::is_same to avoid compilation errors
				return true;  // This is a conservative choice - assuming it might be host accessible
			}
			else {
				// For other memory resources, use the static constexpr member
				return MR::is_host_accessible;
			}
		}

		template <typename MR = TMemoryResource>
		static constexpr bool is_device_accessible() {
			if constexpr ( std::is_same_v<MR, Compute::DynamicMemoryResource> ) {
				// Same conservative approach for device accessibility
				return true;  // This is a conservative choice - assuming it might be device accessible
			}
			else {
				// For other memory resources, use the static constexpr member
				return MR::is_device_accessible;
			}
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
			oss << " Type: " << TensorTrait<TElementType>::type_name << std::endl;

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
			size_( other.size_ ), buffer_( other.buffer_ ) {}
		
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
			shape_( std::move( other.shape_ ) ),
			strides_( std::move( other.strides_ ) ),
			buffer_( std::move( other.buffer_ ) ) {
			other.size_ = 0;
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
				buffer_ = std::move( other.buffer_ );

				other.size_ = 0;
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
				buffer_ = other.buffer_;
			}
			return *this;
		}

		~Tensor() = default;

	private:
		std::string uid_;
		std::string name_;
		size_t size_{ 0 };

		// TODO: Redundant with template parameter TElementType
		//TensorType data_type_;
		
		std::vector<size_t> shape_{};
		std::vector<size_t> strides_{};
		std::shared_ptr<TensorBuffer<TElementType, TMemoryResource>> buffer_{ nullptr };
		std::shared_ptr<TElementType> external_memory_ptr_{ nullptr }; // Shared pointer to external memory (non-owning)

		void allocateBuffer( TElementType value ) {
			buffer_ = std::make_shared<TensorBuffer<TElementType, TMemoryResource>>( size_, value );
			//data_type_ = tensor_type_of( buffer_->data() );
		}

        std::string outputBuffer( size_t index, size_t depth ) const {
            std::ostringstream oss;
            if ( depth == shape_.size() - 1 ) {
                for ( size_t i = 0; i < shape_[ depth ]; ++i ) {
                    if ( i < 3 || i >= shape_[ depth ] - 3 ) {
						TElementType value = buffer_->data()[ index + i ];

						if constexpr ( std::is_same_v<TElementType, half> ) {
							oss << std::setw( 10 ) << static_cast<float>( __half2float( value ) ) << " ";
						}
						else {
							oss << std::setw( 10 ) << value << " ";
						}
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
 * HostTensor stores data in regular CPU memory that is directly accessible from host code.
 * This type is suitable for:
 * - Data that needs frequent host-side access
 * - Input/output data processing
 * - Debugging and inspection of tensor contents
 * - Operations that primarily run on CPU
 *
 * Memory safety:
 * - Safe to access directly through data() method, operator[], at() method
 * - Direct dereference operations will work correctly
 * - No memory transfers required for host access
 *
 * Performance considerations:
 * - Fast host access, but slower for GPU operations
 * - Requires memory transfers when used with GPU operations
 *
 * @tparam TPrecision The data type of the tensor elements.
 */
	export template <typename T>
		using HostTensor = Tensor<T, Compute::HostMemoryResource>;

	/**
	 * @brief Tensor type that uses device (GPU) memory.
	 *
	 * DeviceTensor stores data in GPU memory for optimal performance with CUDA operations.
	 * This type is suitable for:
	 * - Neural network weights, activations, and gradients
	 * - Data used in compute-intensive GPU operations
	 * - Performance-critical processing paths
	 *
	 * Memory safety:
	 * - Cannot be directly accessed from host code
	 * - Attempting direct element access will trigger runtime errors
	 * - Must use to<HostMemoryResource>() to create a host-accessible copy
	 * - Safe for use with CUDA kernels through raw_data() method
	 *
	 * Performance considerations:
	 * - Fastest for GPU operations
	 * - Requires explicit memory transfers for host access
	 * - Most efficient when kept on device throughout processing
	 *
	 * @tparam TPrecision The data type of the tensor elements.
	 */
	export template <class T>
		using DeviceTensor = Tensor<T, Compute::CudaMemoryResource>;

	/**
	 * @brief Tensor type that uses pinned (page-locked) host memory.
	 *
	 * PinnedTensor stores data in page-locked host memory that cannot be swapped to disk.
	 * This type is suitable for:
	 * - Data that needs to be frequently transferred between CPU and GPU
	 * - Input tensors that will be copied to GPU
	 * - Output tensors that need to be read back from GPU
	 *
	 * Memory safety:
	 * - Safe to access directly from host code (data(), operator[], at())
	 * - Provides direct dereference operations like HostTensor
	 * - No runtime safety issues for host access
	 *
	 * Performance considerations:
	 * - Faster host-device transfers than regular host memory
	 * - Consumes a limited system resource (pinned memory)
	 * - Should be used judiciously as excessive use can degrade system performance
	 * - Host access is typically slower than regular host memory
	 *
	 * @tparam TPrecision The data type of the tensor elements.
	 */
	export template <class T>
		using PinnedTensor = Tensor<T, Compute::CudaPinnedMemoryResource>;

	/**
	 * @brief Tensor type that uses CUDA managed memory accessible from both CPU and GPU.
	 *
	 * UniversalTensor uses CUDA's Unified Memory, which is automatically migrated between
	 * host and device as needed by the CUDA runtime. This type is suitable for:
	 * - Data that needs to be accessed from both host and device code
	 * - Prototyping and development where memory management simplicity is preferred
	 * - Cases where optimal data placement isn't known in advance
	 *
	 * Memory safety:
	 * - Safe to access from both host and device code
	 * - No explicit memory transfers needed
	 * - Provides the simplest programming model with automatic data migration
	 *
	 * Performance considerations:
	 * - More convenient but typically lower performance than explicit memory management
	 * - Access patterns that frequently alternate between CPU and GPU may cause thrashing
	 * - Best used with CUDA devices that support hardware page faulting (Pascal or newer)
	 * - May incur overhead from the runtime system managing page migrations
	 *
	 * @tparam TPrecision The data type of the tensor elements.
	 */
	export template <class T>
		using UniversalTensor = Tensor<T, Compute::CudaManagedMemoryResource>;

	// Future backends would add their own aliases
	// export template <class T>
	// using MetalTensor = Tensor<T, Compute::MetalMemoryResource>;
}