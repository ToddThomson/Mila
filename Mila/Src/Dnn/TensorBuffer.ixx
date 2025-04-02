/**
 * @file TensorBuffer.ixx
 * @brief Memory management layer for tensor data with support for CPU and GPU storage.
 */

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
import Cuda.Helpers;

namespace Mila::Dnn
{
	namespace detail
	{
		/**
		 * @brief CUDA warp size alignment (32 threads)
		 *
		 * This constant defines the alignment required for optimal memory access
		 * patterns when using CUDA warp-level operations.
		 */
		constexpr size_t CUDA_WARP_SIZE = 32;

		/**
		 * @brief AVX-512 alignment for CPU operations
		 *
		 * This constant defines the alignment required for optimal SIMD operations
		 * using AVX-512 instructions on CPU.
		 */
		constexpr size_t CPU_SIMD_ALIGN = 64;

		/**
		 * @brief Determines the appropriate memory alignment based on the memory resource type.
		 *
		 * For GPU memory resources, alignment is based on CUDA warp size.
		 * For CPU memory resources, alignment is based on AVX-512 requirements.
		 *
		 * @tparam MR The memory resource type
		 * @return constexpr size_t The required alignment in bytes
		 */
		template<typename MR>
		constexpr size_t get_alignment() {
			if constexpr ( std::is_same_v<MR, Compute::DeviceMemoryResource> ) {
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
	 * @tparam TElementType The type of the elements stored in the buffer.
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
	 * TensorBuffer<float, Compute::HostMemoryResource> cpuBuffer(100, 0.0f);
	 *
	 * // Create a GPU buffer with 100 floats
	 * TensorBuffer<float, Compute::DeviceMemoryResource> gpuBuffer(100);
	 * @endcode
	 */
	export template <typename T, typename MR>
		requires std::is_base_of_v<Compute::MemoryResource, MR>
	class TensorBuffer {
	public:
		/**
		 * @brief The alignment boundary for memory allocation based on the memory resource type.
		 *
		 * This value is determined at compile time to ensure optimal memory access patterns
		 * for the specific memory resource type (CPU or GPU).
		 */
		static constexpr size_t alignment = detail::get_alignment<MR>();

		/**
		 * @brief Constructs a new TensorBuffer object with owned memory and optional value initialization.
		 *
		 * This constructor allocates memory using the specified memory resource and optionally
		 * initializes all elements to the provided value.
		 *
		 * @param size The number of elements in the buffer.
		 * @param value Value to initialize the buffer with (default: default-constructed TElementType).
		 *
		 * @throws std::overflow_error If size is too large, causing overflow in aligned size calculation.
		 * @throws std::bad_alloc If memory allocation fails.
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
		 * @brief Constructs a new TensorBuffer object with externally managed memory.
		 *
		 * This constructor does not allocate memory but instead wraps the provided
		 * pre-allocated memory. The buffer does not take ownership of this memory.
		 *
		 * @param size The number of elements in the buffer.
		 * @param data_ptr Pointer to the pre-allocated memory.
		 *
		 * @throws std::invalid_argument If data_ptr is null.
		 */
		TensorBuffer( size_t size, T* data_ptr )
			: size_( size ), data_( data_ptr ), mr_( nullptr ) {
			if ( data_ == nullptr ) {
				throw std::invalid_argument( "data_ptr cannot be null." );
			}
		}

		/**
		 * @brief Destroys the TensorBuffer object and deallocates memory if owned.
		 *
		 * If the buffer owns its memory (mr_ is not null), it will deallocate
		 * the memory using the memory resource. If the buffer uses externally
		 * managed memory, it will not deallocate anything.
		 */
		~TensorBuffer() {
			if ( mr_ ) {
				mr_->deallocate( data_, size_ * sizeof( T ) );
			}
		}

		/**
		 * @brief Resizes the buffer to a new size, preserving existing data when possible.
		 *
		 * This method allocates a new buffer of the specified size, copies data from
		 * the old buffer to the new one (up to the smaller of the old and new sizes),
		 * and then deallocates the old buffer. If the new size is larger than the old size,
		 * the additional elements are default-initialized.
		 *
		 * @param new_size The new size of the buffer. Can be zero.
		 *
		 * @throws std::runtime_error If the buffer is externally managed.
		 * @throws std::overflow_error If new_size is too large, causing overflow in aligned size calculation.
		 * @throws std::bad_alloc If memory allocation fails.
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
				if constexpr ( std::is_same_v<MR, Compute::HostMemoryResource> ) {
					std::memcpy( new_data, data_, copy_size );
				}
				else {
					Compute::cudaCheckStatus( cudaMemcpy( new_data, data_, copy_size, cudaMemcpyDeviceToDevice ) );
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

		/**
		 * @brief Checks if the buffer's data pointer is properly aligned.
		 *
		 * This method verifies that the buffer's data pointer is aligned according
		 * to the alignment requirements of the memory resource.
		 *
		 * @return bool True if the data pointer is properly aligned, false otherwise.
		 */
		bool is_aligned() const noexcept {
			return reinterpret_cast<std::uintptr_t>(data_) % alignment == 0;
		}

		/**
		 * @brief Gets a pointer to the buffer's data.
		 *
		 * @return TElementType* A pointer to the data stored in the buffer.
		 */
		T* data() {
			return data_;
		}

		/**
		 * @brief Gets the size of the buffer.
		 *
		 * @return size_t The number of elements in the buffer.
		 */
		size_t size() const {
			return size_;
		}

		/**
		 * @brief Copy operations are explicitly deleted.
		 *
		 * TensorBuffer does not support copying since it may involve
		 * deep copying of large memory blocks across different memory spaces.
		 */
		TensorBuffer( const TensorBuffer& ) = delete;
		TensorBuffer& operator=( const TensorBuffer& ) = delete;

	private:
		/**
		 * @brief The number of elements in the buffer.
		 */
		size_t size_{ 0 };

		/**
		 * @brief A pointer to the allocated memory containing the buffer's data.
		 */
		T* data_{ nullptr };

		/**
		 * @brief Memory resource used for allocation/deallocation (null for externally managed buffers).
		 */
		std::unique_ptr<Compute::MemoryResource> mr_{ nullptr };

		/**
		 * @brief Initializes the buffer with a specified value.
		 *
		 * This method fills the buffer with the specified value, using different
		 * approaches depending on whether the buffer is in CPU or GPU memory.
		 *
		 * @param value Value to initialize the buffer with (default: default-constructed TElementType).
		 *
		 * @note For CPU memory, SIMD optimizations may be used for larger trivially copyable types.
		 * @note For GPU memory, a temporary host buffer is created and copied to the device.
		 */
		void fillBuffer( T value = T{} ) {
			if constexpr ( std::is_same_v<MR, Compute::HostMemoryResource> ) {
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
			else if constexpr ( std::is_same_v<MR, Compute::DeviceMemoryResource> ) {
				std::vector<T> temp( size_, value );
				Compute::cudaCheckStatus( cudaMemcpy( data_, temp.data(), size_ * sizeof( T ), cudaMemcpyHostToDevice ) );
			}
		}
	};
}
