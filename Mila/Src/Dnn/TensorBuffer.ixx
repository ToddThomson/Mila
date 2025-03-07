module;
#include <memory>
#include <vector>
#include <cuda_runtime.h>

export module Dnn.TensorBuffer;

import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn
{
	/**
	* @brief A buffer for storing tensor data.
	* 
	* @tparam T The type of the elements stored in the buffer.
	*/
	export template <typename T, typename MR> 
		requires std::is_base_of_v<Compute::MemoryResource, MR>
	class TensorBuffer {
	public:
		/**
		* @brief Construct a new TensorBuffer object with optional value initialization.
		* 
		* @param size The number of elements in the buffer.
		* @param value Value to initialize the buffer with.
		*/
		explicit TensorBuffer( size_t size, T value = T{} )
			: size_( size ), mr_( std::make_unique<MR>() ) {
			data_ = static_cast<T*>(mr_->allocate( size_ * sizeof( T ) ));
			initializeBuffer( value );
		}

		/**
		* @brief Destroy the TensorBuffer object.
		*/
		~TensorBuffer() {
			mr_->deallocate(data_, size_ * sizeof(T));
		}

		/**
		* @brief Resize the buffer.
		* 
		* @param size The new size of the buffer.
		*/
		void resize(size_t size) {
			if (size_ != size) {
				mr_->deallocate(data_, size_ * sizeof(T));
				size_ = size;
				data_ = static_cast<T*>(mr_->allocate(size_ * sizeof(T)));
				initializeBuffer();
			}
		}

		/**
		* @brief Access an element in the buffer.
		* 
		* @param i The index of the element.
		* @return T& A reference to the element.
		*/
		/*T& operator[](size_t i) {
			return data_[i];
		}*/

		/**
		* @brief Get a pointer to the data.
		* 
		* @return T* A pointer to the data.
		*/
		T* data() {
			return data_;
		}

		/**
		* @brief Get the size of the buffer.
		* 
		* @return size_t The number of elements in the buffer.
		*/
		size_t size() const {
			return size_;
		}

	private:
		size_t size_{0}; ///< The number of elements in the buffer.
		T* data_{nullptr}; ///< A pointer to the data.
		std::unique_ptr<Compute::MemoryResource> mr_{nullptr}; ///< A unique pointer to the memory resource.

		/**
		* @brief Initialize the buffer with a value.
		* 
		* @param value Value to initialize the buffer with.
		*/
		void initializeBuffer(T value = T{}) {
			if constexpr (std::is_same_v<MR, Compute::CpuMemoryResource>) {
				std::fill(data_, data_ + size_, value);
			} else if constexpr (std::is_same_v<MR, Compute::CudaMemoryResource>) {
				std::vector<T> temp(size_, value);
				cudaMemcpy(data_, temp.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
			}
		}
	};
}