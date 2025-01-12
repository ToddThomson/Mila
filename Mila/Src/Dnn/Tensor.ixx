module;
#include <vector>  
#include <numeric>  
#include <iostream>
#include <iomanip>
#include <cassert>  
#include <variant>  
#include <memory>
#include <mdspan>
#include <functional>
#include <cuda_fp16.h>

export module Dnn.Tensor;

import Dnn.TensorType;  
import Dnn.TensorBuffer; 
import Dnn.TensorTag;
import Compute.DeviceRegistry;
import Compute.DeviceInterface;
//import Compute.CudaDevice;

namespace Mila::Dnn
{
	export template<typename T>
		class Tensor : TensorTag {
		public:

			using ElementType = T;

			using Extent1d = std::dextents<size_t, 1>;
			using Extent2d = std::dextents<size_t, 2>;
			using Extent3d = std::dextents<size_t, 3>;
			using Extent4d = std::dextents<size_t, 4>;

			Tensor( const std::vector<size_t>& shape, const std::string& device_name = "CPU", int device_id = 0 )
				:shape_( shape ), strides_( computeStrides( shape ) ), size_( computeSize( shape ) ) {
				setDevice( device_name, device_id );
				allocateBuffer();

				// TJT: Feature 
				//if ( initializer ) {
				//	initializer( this );
				//}
			}

			// Creates an empty tensor with zero size
			Tensor()
				: shape_(), strides_( computeStrides( shape_ ) ), size_() {
				//const std::string& device_name = "CPU";
				//int device_id = 0;
				//setDevice( device_name, device_id );
				allocateBuffer();
			}

			// Copy asignement operator
			Tensor& operator=( const Tensor& other ) {
				if ( this != &other ) {
					shape_ = other.shape_;
					strides_ = other.strides_;
					size_ = other.size_;
					data_type_ = other.data_type_;
					buffer_ = other.buffer_;
					device_ = other.device_; // ->clone();
				}
				return *this;
			}

			// Copy constructor for shallow copy
			Tensor( const Tensor& other )
				: shape_( other.shape_ ), strides_( other.strides_ ), size_( other.size_ ), data_type_( other.data_type_ ), buffer_( other.buffer_ ), device_( other.device_ ) {}

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
				return std::mdspan<ElementType, Extent1d>( buffer_->data(), size_ );
			}

			auto matrixSpan( const std::vector<size_t>& shape ) {
				if ( shape.size() != 2 ) {
					throw std::runtime_error( "matrixSpan: The shape must have exactly 2 dimensions." );
				}
				size_t total_size = shape[ 0 ] * shape[ 1 ];
				if ( total_size > size_ ) {
					throw std::runtime_error( "matrixSpan: The specified shape exceeds the tensor size." );
				}
				return std::mdspan<ElementType, Extent2d>( buffer_->data(), shape[ 0 ], shape[ 1 ] );
			}

			/*T_ELEM& at( const std::vector<size_t>& indices ) {
				size_t offset = 0;
				for ( size_t i = 0; i < indices.size(); ++i ) {
					offset += indices[ i ] * strides_[ i ];
				}
				return data()[ offset ];
			}*/

			/*template<typename... Args>
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
			}*/

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

			const std::shared_ptr<Compute::DeviceInterface>& device() const {
				return device_;
			}

			T* data() {
				return buffer_->data();
			}

			const T* data() const {
				return buffer_->data();
			}

			void fill( const T& value ) {
				buffer_->fill( value );
			}

			void print() {
				std::cout << "Tensor of shape: ";
				for ( auto dim : shape_ ) {
					std::cout << dim << " ";
				}
				std::cout << std::endl;
				std::cout << "TensorType::" << to_string( data_type_ ) << std::endl;
				std::cout << "Data:" << std::endl;

				if ( data_type_ == TensorType::kFP32 ) {
					printBuffer<float>( 0, 0 );
				}
			}

		private:
			size_t size_{ 0 };
			TensorType data_type_{ TensorType::kEmptyType };
			std::vector<size_t> shape_;
			std::vector<size_t> strides_;
			std::shared_ptr<TensorBuffer<T>> buffer_;
			std::shared_ptr<Compute::DeviceInterface> device_;

			void allocateBuffer() {
				buffer_ = std::make_shared<TensorBuffer<T>>( size_ );
				data_type_ = tensor_type_of( buffer_->data() );
			}

			template <typename T>
			void printBuffer( size_t index, size_t depth ) {
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
							printBuffer<T>( index + i * strides_[ depth ], depth + 1 );
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

			size_t computeIndex( const std::vector<size_t>& indices ) const {
				size_t index = 0;
				for ( size_t i = 0; i < indices.size(); ++i ) {
					index += indices[ i ] * strides_[ i ];
				}
				return index;
			}

			void setDevice( const std::string& device_name, int device_id ) {
				auto dev = Compute::DeviceRegistry::instance().createDevice( device_name );
				if ( dev ) {
					device_ = std::move( dev );
				}
				else {
					throw std::runtime_error( "Invalid device name." );
				}
			}
	};
} // namespace Mila::Dnn
