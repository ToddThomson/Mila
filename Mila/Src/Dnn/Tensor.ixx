module;
#include <vector>  
#include <iostream>
#include <sstream>
#include <iomanip>
#include <variant>  
#include <memory>
#include <mdspan>
#include <optional>
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

	export template<typename T, typename TMemoryResource = Compute::CpuMemoryResource>
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

		template<typename NewTMemoryResource>
		Tensor<T, NewTMemoryResource> to() const {
			static_assert(std::is_base_of_v<Compute::MemoryResource, NewTMemoryResource>, "NewTMemoryResource must be derived from Compute::MemoryResource");

			// Create a new tensor with the same shape and the new memory resource
			Tensor<T, NewTMemoryResource> new_tensor( shape_ );

			// Copy data from the current tensor to the new tensor
			if constexpr ( std::is_same_v<TMemoryResource, Compute::CpuMemoryResource> && std::is_same_v<NewTMemoryResource, Compute::CudaMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyHostToDevice );
			}
			else if constexpr ( std::is_same_v<TMemoryResource, Compute::CudaMemoryResource> && std::is_same_v<NewTMemoryResource, Compute::CpuMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyDeviceToHost );
			}
			else if constexpr ( std::is_same_v<TMemoryResource, Compute::CpuMemoryResource> && std::is_same_v<NewTMemoryResource, Compute::CudaPinnedMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyHostToHost );
			}
			else if constexpr ( std::is_same_v<TMemoryResource, Compute::CudaPinnedMemoryResource> && std::is_same_v<NewTMemoryResource, Compute::CpuMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyHostToHost );
			}
			else if constexpr ( std::is_same_v<TMemoryResource, Compute::CudaMemoryResource> && std::is_same_v<NewTMemoryResource, Compute::CudaPinnedMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyDeviceToHost );
			}
			else if constexpr ( std::is_same_v<TMemoryResource, Compute::CudaPinnedMemoryResource> && std::is_same_v<NewTMemoryResource, Compute::CudaMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyHostToDevice );
			}
			else if constexpr ( std::is_same_v<TMemoryResource, Compute::CpuMemoryResource> && std::is_same_v<NewTMemoryResource, Compute::CudaManagedMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyHostToDevice );
			}
			else if constexpr ( std::is_same_v<TMemoryResource, Compute::CudaManagedMemoryResource> && std::is_same_v<NewTMemoryResource, Compute::CpuMemoryResource> ) {
				cudaMemcpy( new_tensor.data(), this->data(), size_ * sizeof( T ), cudaMemcpyDeviceToHost );
			}
			else {
				std::copy( this->data(), this->data() + size_, new_tensor.data() );
			}

			return new_tensor;
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
			static_assert(sizeof...(args) > 0, "operator[]: At least one index must be provided.");
			const size_t num_args = sizeof...(args);
			if ( num_args != shape_.size() ) {
				throw std::runtime_error( "operator[]: Number of indices must match the tensor rank." );
			}
			size_t index = computeIndex( { static_cast<size_t>(args)... } );
			if ( index >= size_ ) {
				throw std::out_of_range( "operator[]: Index out of range." );
			}
			return buffer_->data()[ index ];
		}

		template<typename... Args>
		const T& operator[]( Args... args ) const {
			static_assert(sizeof...(args) > 0, "operator[]: At least one index must be provided.");
			const size_t num_args = sizeof...(args);
			if ( num_args != shape_.size() ) {
				throw std::runtime_error( "operator[]: Number of indices must match the tensor rank." );
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
			if constexpr ( std::is_same_v<TMemoryResource, Compute::CpuMemoryResource> || std::is_same_v<TMemoryResource, Compute::CudaManagedMemoryResource> ) {
				std::fill( buffer_->data(), buffer_->data() + size_, value );
			}
			else {
				// TODO: Implement fill for other memory resources
				throw std::runtime_error( "Fill is only supported for CPU and Managed memory." );
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
			oss << " Type: " << to_string( data_type_ ) << std::endl;

			if ( showBuffer ) {
				oss << outputBuffer( 0, 0 );
			}

			return oss.str();
		}

		friend std::ostream& operator<<( std::ostream& os, const Tensor& tensor ) {
			os << tensor.toString();
			return os;
		}

		/**
		* @brief Copy constructor.
		*
		* This constructor creates a new tensor by copying the contents of the given tensor.
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
			scalar_value_( std::move( other.scalar_value_ ) ),
			is_scalar_( other.is_scalar_ ),
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
		std::optional<scalar_t> scalar_value_{ std::nullopt };
		bool is_scalar_{ false };
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

		std::string set_uid() {
			return "tensor_" + std::to_string( UniqueIdGenerator::getNextId() );
		}
	};

	export template <typename T>
		using CpuTensor = Tensor<T, Compute::CpuMemoryResource>;

	export template <class T>
		using CudaTensor = Tensor<T, Compute::CudaMemoryResource>;

	export template <class T>
		using PinnedTensor = Tensor<T, Compute::CudaPinnedMemoryResource>;

	export template <class T>
		using UniversalTensor = Tensor<T, Compute::CudaManagedMemoryResource>;
}