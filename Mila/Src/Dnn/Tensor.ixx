module;
#include <vector>  
#include <iostream>
#include <iomanip>
#include <variant>  
#include <memory>
#include <mdspan>
#include <optional>
#include <cuda_fp16.h>

export module Dnn.Tensor;

import Dnn.TensorType;  
import Dnn.TensorBuffer; 
import Dnn.TensorTag;

import Compute.ComputeDevice;
import Compute.DeviceContext;

import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.DeviceMemoryResource;
import Compute.ManagedMemoryResource;
import Compute.PinnedMemoryResource;

namespace Mila::Dnn
{
	export template<typename T, typename MR> requires std::is_base_of_v<Compute::MemoryResource, MR>
	class Tensor : TensorTag {
	public:

		using scalar_t = std::variant<int64_t, int32_t, half, float>;

		using Extent1d = std::dextents<size_t, 1>;
		using Extent2d = std::dextents<size_t, 2>;
		using Extent3d = std::dextents<size_t, 3>;
		using Extent4d = std::dextents<size_t, 4>;

		Tensor( const std::vector<size_t>& shape, std::shared_ptr<Compute::ComputeDevice> device_ptr = nullptr )
			: shape_( shape ), strides_( computeStrides( shape ) ), size_( computeSize( shape ) ), device_ptr_( setDevice( device_ptr ) ) {
			allocateBuffer();
		}

		Tensor( const std::vector<size_t>& shape, const T& value, std::shared_ptr<Compute::ComputeDevice> device_ptr = nullptr )
			: shape_( shape ), strides_( computeStrides( shape ) ), size_( computeSize( shape ) ), device_ptr_( setDevice( device_ptr ) ) {
			allocateBuffer( value );
		}

		Tensor()
			: shape_(), strides_( computeStrides( shape_ ) ), size_(), device_ptr_( setDevice( nullptr ) ) {
			allocateBuffer();
		}

		// Copy constructor for shallow copy
		Tensor( const Tensor& other )
			: shape_( other.shape_ ), strides_( other.strides_ ), size_( other.size_ ), data_type_( other.data_type_ ),
			buffer_( other.buffer_ ), device_ptr_( other.device_ptr_ ) {}

		/*Tensor( float const& scalar )
			: scalar_value_( scalar ), is_scalar_( true ), shape_{ 1 }, strides_{ 1 }, data_type_( TensorType::kFP32 ){
		}

		Tensor( half const& scalar )
			: scalar_value_( scalar ), is_scalar_( true ), shape_{ 1 }, strides_{ 1 }, data_type_( TensorType::kFP16 ) {
		}

		Tensor( int32_t const& scalar )
			: scalar_value_( scalar ), is_scalar_( true ), shape_{ 1 }, strides_{ 1 }, data_type_( TensorType::kINT32 ) {}

		Tensor( int64_t const& scalar )
			: scalar_value_( scalar ), is_scalar_( true ), shape_{ 1 }, strides_{ 1 }, data_type_( TensorType::kINT64 ) {
		}*/

		// Copy asignement operator
		Tensor& operator=( const Tensor& other ) {
			if ( this != &other ) {
				shape_ = other.shape_;
				strides_ = other.strides_;
				size_ = other.size_;
				data_type_ = other.data_type_;
				buffer_ = other.buffer_;
				device_ptr_ = other.device_ptr_;
			}
			return *this;
		}

		template<typename NewMR>
		Tensor<T, NewMR> to() const {
			static_assert(std::is_base_of_v<Compute::MemoryResource, NewMR>, "NewMR must be derived from Compute::MemoryResource");

			// Create a new tensor with the same shape and the new memory resource
			Tensor<T, NewMR> new_tensor( shape_, device_ptr_ );

			// Copy data from the current tensor to the new tensor
			if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> && std::is_same_v<NewMR, Compute::DeviceMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyHostToDevice );
			}
			else if constexpr ( std::is_same_v<MR, Compute::DeviceMemoryResource> && std::is_same_v<NewMR, Compute::CpuMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyDeviceToHost );
			}
			else if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> && std::is_same_v<NewMR, Compute::PinnedMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyHostToHost );
			}
			else if constexpr ( std::is_same_v<MR, Compute::PinnedMemoryResource> && std::is_same_v<NewMR, Compute::CpuMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyHostToHost );
			}
			else if constexpr ( std::is_same_v<MR, Compute::DeviceMemoryResource> && std::is_same_v<NewMR, Compute::PinnedMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyDeviceToHost );
			}
			else if constexpr ( std::is_same_v<MR, Compute::PinnedMemoryResource> && std::is_same_v<NewMR, Compute::DeviceMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyHostToDevice );
			}
			else if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> && std::is_same_v<NewMR, Compute::ManagedMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyHostToDevice );
			}
			else if constexpr ( std::is_same_v<MR, Compute::ManagedMemoryResource> && std::is_same_v<NewMR, Compute::CpuMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyDeviceToHost );
			}
			else {
				std::copy( this->data(), this->data() + size_, new_tensor.data() );
			}

			return new_tensor;
		}

		//template<typename NewMR>
		//Tensor<T, NewMR> to() const {
		//	Tensor<T, NewMR> new_tensor( shape_, device_ptr_ );
		//	if ( constexpr (std::is_same_v<MR, NewMR>) ) {
		//		std::copy( buffer_->data(), buffer_->data() + size_, new_tensor.data() );
		//	}
		//	else {
		//		// Copy data from one memory resource to another
		//		std::copy( buffer_->data(), buffer_->data() + size_, new_tensor.data() );
		//	}
		//	
		//	return new_tensor;
		//}

		void reshape( const std::vector<size_t>& new_shape ) {
			size_t new_size = computeSize( new_shape );
			if ( this->empty() || new_size == size_ ) {
				shape_ = new_shape;
				if ( empty() ) {
					buffer_->resize( new_size );
					size_ = new_size;
				}
				return;
			}

			throw std::runtime_error( "The new shape must match the size of the tensor or the tensor must be empty." );
		}

		auto vectorSpan() {
			return std::mdspan<T, Extent1d>( buffer_->data(), size_ );
		}

		auto matrixSpan( const std::vector<size_t>& shape ) {
			if ( shape.size() != 2 ) {
				throw std::runtime_error( "matrixSpan: The shape must have exactly 2 dimensions." );
			}
			size_t total_size = shape[ 0 ] * shape[ 1 ];
			if ( total_size > size_ ) {
				throw std::runtime_error( "matrixSpan: The specified shape exceeds the tensor size." );
			}
			return std::mdspan<T, Extent2d>( buffer_->data(), shape[ 0 ], shape[ 1 ] );
		}

		/*T_ELEM& at( const std::vector<size_t>& indices ) {
			size_t offset = 0;
			for ( size_t i = 0; i < indices.size(); ++i ) {
				offset += indices[ i ] * strides_[ i ];
			}
			return data()[ offset ];
		}*/

		template<typename... Args>
		T& operator[]( Args... args ) {
			size_t index = computeIndex( { static_cast<size_t>(args)... } );
			return buffer_->data()[ index ];
		}

		template<typename... Args>
		const T& operator[]( Args... args ) const {
			const size_t num_args = sizeof...(args);
			if ( num_args != shape_.size() ) {
				throw std::runtime_error( "Number of indices must match the tensor rank." );
			}
			size_t index = computeIndex( { static_cast<size_t>(args)... } );
			return buffer_->data()[ index ];
		}

		T& operator[]( size_t index ) {
			if ( rank() != 1 ) {
				throw std::runtime_error( "Operator[]: The rank of the tensor must be 1." );
			}
			return buffer_->data()[ index ];
		}

		const T& operator[]( size_t index ) const {
			if ( rank() != 1 ) {
				throw std::runtime_error( "Operator[]: The rank of the tensor must be 1." );
			}
			return buffer_->data()[ index ];
		}

		T& operator[]( size_t row, size_t col ) {
			if ( rank() != 2 ) {
				throw std::runtime_error( "Operator[]: The rank of the tensor must be 2." );
			}

			size_t index = row * shape_[ 1 ] + col;
			if ( index >= size_ ) {
				throw std::runtime_error( "Index out or range." );
			}
			return buffer_->data()[ index ];
		}

		const T& operator[]( size_t row, size_t col ) const {
			if ( rank() != 2 ) {
				throw std::runtime_error( "Operator[]: The rank of the tensor must be 2." );
			}

			size_t index = row * shape_[ 1 ] + col;
			if ( index >= size_ ) {
				throw std::runtime_error( "Index out or range." );
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

		const std::shared_ptr<Compute::ComputeDevice>& device() const {
			return device_ptr_;
		}

		/*template<typename TValue>
		TValue scalar() const {
			if ( is_scalar_ && scalar_value_.has_value() ) {
				auto scalar_var = scalar_value_.value();
				return std::get<TValue>(scalar_var);
			}
			else {
				throw std::runtime_error( "Tensor does not hold a scalar value." );
			}
		}*/


		T* data() {
			return buffer_->data();
		}

		const T* data() const {
			return buffer_->data();
		}

		void fill( const T& value ) {
			if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> || std::is_same_v<MR, Compute::ManagedMemoryResource> ) {
				std::fill( buffer_->data(), buffer_->data() + size_, value );
			}
			else {
				// TODO: Implement fill for other memory resources
				throw std::runtime_error( "Fill is only supported for CPU and Managed memory." );
			}
		}

		void print() const {
			std::cout << "Tensor of shape: ";
			for ( auto dim : shape_ ) {
				std::cout << dim << " ";
			}
			std::cout << std::endl;
			std::cout << "Tensor type::" << to_string( data_type_ ) << std::endl;
			//std::cout << "Device: " << device_->getName() << std::endl;
			std::cout << "Size: " << size_ << std::endl;
			std::cout << "Data:" << std::endl;

			if ( data_type_ == TensorType::kFP32 ) {
				printBuffer( 0, 0 );
			}
		}

	private:
		std::optional<scalar_t> scalar_value_{ std::nullopt };
		bool is_scalar_{ false };
		size_t size_{ 0 };
		TensorType data_type_{ TensorType::kNotSet };
		std::vector<size_t> shape_{};
		std::vector<size_t> strides_{};
		std::shared_ptr<Compute::ComputeDevice> device_ptr_{ nullptr };
		MR mr_{};
		std::shared_ptr<TensorBuffer<T, MR>> buffer_{ nullptr };

		void allocateBuffer() {
			buffer_ = std::make_shared<TensorBuffer<T, MR>>( size_ );
			data_type_ = tensor_type_of( buffer_->data() );
		}
		void allocateBuffer( T value ) {
			buffer_ = std::make_shared<TensorBuffer<T, MR>>( size_, value );
			data_type_ = tensor_type_of( buffer_->data() );
		}

		void printBuffer( size_t index, size_t depth ) const {
			if ( depth == shape_.size() - 1 ) {
				for ( size_t i = 0; i < shape_[ depth ]; ++i ) {
					if ( i < 3 || i >= shape_[ depth ] - 3 ) {
						std::cout << std::setw( 10 ) << buffer_->data()[ index + i ] << " ";
					}
					else if ( i == 3 ) {
						std::cout << "... ";
					}
				}
			}
			else {
				for ( size_t i = 0; i < shape_[ depth ]; ++i ) {
					if ( i < 3 || i >= shape_[ depth ] - 3 ) {
						std::cout << "[ ";
						printBuffer( index + i * strides_[ depth ], depth + 1 );
						std::cout << "]" << std::endl;
					}
					else if ( i == 3 ) {
						std::cout << "[ ... ]" << std::endl;
						i = shape_[ depth ] - 4;
					}
				}
			}
		}

		static std::vector<size_t> computeStrides( const std::vector<size_t>& shape ) {
			std::vector<size_t> strides( shape.size() );

			size_t stride = 1;
			for ( int i = shape.size() - 1; i >= 0; --i ) {
				strides[ i ] = stride;
				stride *= shape[ i ];
			}

			return strides;
		}

		static size_t computeSize( const std::vector<size_t>& shape ) {
			size_t size = 1;
			for ( size_t dim : shape ) {
				size *= dim;
			}
			return size;
		}

		std::shared_ptr<Compute::ComputeDevice> setDevice( std::shared_ptr<Compute::ComputeDevice> device_ptr ) {
			if ( device_ptr ) {
				return device_ptr;
			}
			else {
				return Compute::DeviceContext::instance().getDevice();
			}
		}

		size_t computeIndex( const std::vector<size_t>& indices ) const {
			size_t index = 0;
			for ( size_t i = 0; i < indices.size(); ++i ) {
				index += indices[ i ] * strides_[ i ];
			}
			return index;
		}
	};

	export template <class T>
		using HostTensor = Tensor<T, Compute::CpuMemoryResource>;

	export template <class T>
		using DeviceTensor = Tensor<T, Compute::DeviceMemoryResource>;

	export template <class T>
		using PinnedTensor = Tensor<T, Compute::PinnedMemoryResource>;

	export template <class T>
		using UniversalTensor = Tensor<T, Compute::ManagedMemoryResource>;
}