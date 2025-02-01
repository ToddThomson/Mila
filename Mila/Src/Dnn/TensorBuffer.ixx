module;
#include <memory>

export module Dnn.TensorBuffer;

import Compute.MemoryResource;

namespace Mila::Dnn
{
	/**
	* @brief A buffer for storing tensor data.
	* 
	* @tparam T The type of the elements stored in the buffer.
	*/
	export template <typename T>
	class TensorBuffer {
	public:
		/**
		* @brief Construct a new TensorBuffer object.
		* 
		* @param size The number of elements in the buffer.
		* @param mr A shared pointer to a memory resource for allocation.
		*/
		explicit TensorBuffer( size_t size, std::shared_ptr<Compute::MemoryResource> mr )
			: size_( size ), mr_( mr ) {
			data_ = static_cast<T*>( mr_->allocate( size_ * sizeof( T ) ) );
		}

		/**
		* @brief Destroy the TensorBuffer object.
		*/
		~TensorBuffer() {
			mr_->deallocate( data_, size_ * sizeof(T) );
		}

		/**
		* @brief Resize the buffer.
		* 
		* @param size The new size of the buffer.
		*/
		void resize( size_t size ) {
			if ( size_ != size ) {
				mr_->deallocate( data_, size_ * sizeof( T ) );
				size_ = size;
				data_ = static_cast<T*>( mr_->allocate( size_ * sizeof(T) ) );
			}
		}

		/**
		* @brief Access an element in the buffer.
		* 
		* @param i The index of the element.
		* @return T& A reference to the element.
		*/
		T& operator[]( size_t i ) {
			return data_[ i ];
		}



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
		size_t size_{ 0 }; ///< The number of elements in the buffer.
		T* data_{ nullptr }; ///< A pointer to the data.
		std::shared_ptr<Compute::MemoryResource> mr_{ nullptr }; ///< A shared pointer to the memory resource.
	};
}