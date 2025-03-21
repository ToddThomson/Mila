module;
#include <memory>
#include <vector>
#include <limits>
#include <stdexcept>
#include <cuda_runtime.h>
#include <type_traits>

export module Dnn.TensorBuffer;

import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn
{
	namespace detail
	{
		// CUDA warp size alignment (32 threads)
		constexpr size_t CUDA_WARP_SIZE = 32;
		// AVX-512 alignment for CPU operations
		constexpr size_t CPU_SIMD_ALIGN = 64;

		template<typename MR>
		constexpr size_t get_alignment() {
			if constexpr ( std::is_same_v<MR, Compute::CudaMemoryResource> ) {
				return CUDA_WARP_SIZE * sizeof( float ); // Typical tensor element size
			}
			else {
				return CPU_SIMD_ALIGN;
			}
		}
	}

	/**
	 * @brief A buffer for storing tensor data with configurable memory management.
	 *
	 * TensorBuffer provides a generic container for tensor data that can be stored
	 * either in CPU or GPU memory based on the memory resource type. It supports
	 * both owned and externally managed memory.
	 *
	 * @tparam T The type of the elements stored in the buffer.
	 * @tparam MR The memory resource type that determines where and how memory is allocated.
	 *           Must derive from Compute::MemoryResource.
	 *
	 * @note Thread Safety: The buffer is not thread-safe. Multiple threads accessing
	 *       the same buffer must be synchronized externally.
	 *
	 * @note Exception Safety: Basic guarantee - if an exception occurs during operations,
	 *       no memory is leaked but the buffer may be left in an undefined state.
	 *
	 * Example usage:
	 * @code
	 * // Create a CPU buffer with 100 floats initialized to 0.0f
	 * TensorBuffer<float, Compute::CpuMemoryResource> cpuBuffer(100, 0.0f);
	 *
	 * // Create a GPU buffer with 100 floats
	 * TensorBuffer<float, Compute::CudaMemoryResource> gpuBuffer(100);
	 * @endcode
	 */
	export template <typename T, typename MR> 
		requires std::is_base_of_v<Compute::MemoryResource, MR>
	class TensorBuffer {
	public:
		static constexpr size_t alignment = detail::get_alignment<MR>();

		/**
		* @brief Construct a new TensorBuffer object with optional value initialization.
		* 
		* @param size The number of elements in the buffer.
		* @param value Value to initialize the buffer with.
		*/
		explicit TensorBuffer( size_t size, T value = T{} )
			: size_( size ), mr_( std::make_unique<MR>() ) {
			// Check for overflow
			if ( size_ > (std::numeric_limits<size_t>::max() - alignment + 1) / sizeof( T ) ) {
				throw std::overflow_error( "Size too large, causing overflow in aligned_size calculation." );
			}
			size_t aligned_size = (size_ * sizeof( T ) + alignment - 1) & ~(alignment - 1);
			data_ = static_cast<T*>(mr_->allocate( aligned_size, alignment ));
			
			fillBuffer( value );
		}

		/**
		* @brief Construct a new TensorBuffer object with a pointer to pre-allocated memory.
		*
		* @param size The number of elements in the buffer.
		* @param data_ptr Pointer to the pre-allocated memory.
		*/
		TensorBuffer( size_t size, T* data_ptr )
			: size_( size ), data_( data_ptr ), mr_( nullptr ) {
			if ( data_ == nullptr ) {
				throw std::invalid_argument( "data_ptr cannot be null." );
			}
		}

		/**
		* @brief Destroy the TensorBuffer object.
		*/
		~TensorBuffer() {
			if ( mr_ ) {
				mr_->deallocate( data_, size_ * sizeof( T ) );
			}
		}
		/**
		* @brief Resize the buffer.
		*
		* @param new_size The new size of the buffer. Can be zero.
		* @throws std::runtime_error If the buffer is externally managed
		* @throws std::bad_alloc If memory allocation fails
		*/
		void resize( size_t new_size ) {
			if ( size_ == new_size ) {
				return;
			}

			if ( !mr_ ) {
				throw std::runtime_error( "Cannot resize an externally managed buffer." );
			}
			// Check for overflow
			if ( new_size > (std::numeric_limits<size_t>::max() - alignment + 1) / sizeof( T ) ) {
				throw std::overflow_error( "New size too large, causing overflow in aligned_size calculation." );
			}

			size_t aligned_size = (new_size * sizeof( T ) + alignment - 1) & ~(alignment - 1);
			T* new_data = static_cast<T*>(mr_->allocate( aligned_size, alignment ));

			if ( data_ ) {
				// Copy existing data if needed
				size_t copy_size = std::min( size_, new_size ) * sizeof( T );
				if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> ) {
					std::memcpy( new_data, data_, copy_size );
				}
				else {
					cudaMemcpy( new_data, data_, copy_size, cudaMemcpyDeviceToDevice );
				}
				mr_->deallocate( data_, (size_ * sizeof( T ) + alignment - 1) & ~(alignment - 1) );
			}

			data_ = new_data;
			size_ = new_size;

			// Fill remaining space if expanding
			if ( new_size > size_ ) {
				fillBuffer();
			}
		}

		bool is_aligned() const noexcept {
			return reinterpret_cast<std::uintptr_t>(data_) % alignment == 0;
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
		
		// Since this is internal API used by Tensor, explicitly delete copy operations
		TensorBuffer( const TensorBuffer& ) = delete;
		TensorBuffer& operator=( const TensorBuffer& ) = delete;

	private:
		size_t size_{0}; ///< The number of elements in the buffer.
		T* data_{nullptr}; ///< A pointer to the data.
		std::unique_ptr<Compute::MemoryResource> mr_{nullptr}; ///< A unique pointer to the memory resource.

		/**
		* @brief Initialize the buffer with a value.
		* 
		* @param value Value to initialize the buffer with.
		*/
		void fillBuffer( T value = T{} ) {
			if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> ) {
				if constexpr ( sizeof( T ) >= 4 && std::is_trivially_copyable_v<T> ) {
					// Use aligned operations for larger trivially copyable types
				#if defined(__AVX512F__)
				// Use AVX-512 operations
				// Implementation depends on your specific needs
					std::fill( data_, data_ + size_, value );
				#elif defined(__AVX2__)
				// Use AVX2 operations
					std::fill( data_, data_ + size_, value );
				#else
				// Fall back to standard fill
					std::fill( data_, data_ + size_, value );
				#endif
				}
				else {
					std::fill( data_, data_ + size_, value );
				}
			}
			else if constexpr ( std::is_same_v<MR, Compute::CudaMemoryResource> ) {
				std::vector<T> temp( size_, value );
				if ( cudaMemcpy( data_, temp.data(), size_ * sizeof( T ),
					cudaMemcpyHostToDevice ) != cudaSuccess ) {
					throw std::runtime_error( "cudaMemcpy failed during fillBuffer." );
				}
			}
		}
	};
}