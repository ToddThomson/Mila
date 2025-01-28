#pragma once

#include <thrust/host_vector.h>

// TJT: Adding the device_vector include causes a compile error
//#include <thrust/device_vector.h>

#include "TensorBufferBase.h"

namespace Mila::Dnn
{
	template<typename T, template<typename> class VectorType = thrust::host_vector>
	class TensorBuffer : public TensorBufferBase<T> {
		public:
			TensorBuffer( size_t buffer_size )
				: buffer_( buffer_size ) {}

			size_t size() const override {
				return buffer_.size();
			}

			const T* data() const override {
				//if constexpr ( std::is_same_v<VectorType<T>, thrust::device_vector<T>> ) {
				//	return thrust::raw_pointer_cast( buffer_.data() );
				//}
				//else {
					return buffer_.data();
				//}
			}

			T* data() override {
				//if constexpr ( std::is_same_v<VectorType<T>, thrust::device_vector<T>> ) {
				//	return thrust::raw_pointer_cast( buffer_.data() );
				//}
				//else {
					return buffer_.data();
				//}
			}

			void fill( const T& value ) override {
				thrust::fill( buffer_.begin(), buffer_.end(), value );
			}

			void resize( size_t size ) override {
				buffer_.resize( size );
			}

		private:
			VectorType<T> buffer_;
	};
}