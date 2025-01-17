module;
#include <vector>
#include <memory>

export module Dnn.TensorBuffer;

import Dnn.TensorType;
//import Dnn.TensorBufferBase;

namespace Mila::Dnn
{
	export template<typename T>
	class TensorBuffer {//: public TensorBufferBase {
    public:
        TensorBuffer( size_t buffer_size )
			: buffer_( buffer_size ) {
        }

		/*TensorBuffer()
			: buffer_( 0 ) {
		}*/

        size_t size() const {
            return buffer_.size();
        }

		const T* data() const {
			return buffer_.data();
		}

		T * data() {
			return buffer_.data();
		}

		void fill(const T& value) {
			std::fill(buffer_.begin(), std::end(buffer_), value);
		}

		void resize( size_t size ) {
			buffer_.resize( size );
		}


    private:
		//thrust::host_vector<std::byte> buffer_;
		std::vector<T> buffer_;
    };
} // namespace Mila::Dnn