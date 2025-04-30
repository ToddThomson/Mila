/**
 * @file TensorBuffer.ixx
 * @brief Memory management layer for tensor data with support for CPU and GPU storage.
 */

module;
#include <memory>
#include <vector>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <type_traits>

export module Dnn.TensorBuffer;

import Compute.MemoryResource;
import Compute.MemoryResourceTracker;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Cuda.Error;

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
	 * TensorBuffer<float, Compute::CudaMemoryResource> gpuBuffer(100);
	 * @endcode
	 */
	export template <typename T, typename MR, bool TrackMemory = true>
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
			: size_( size ), aligned_size_( 0 ), mr_( createMemoryResource() ) {
			if ( size_ == 0 ) {
				data_ = nullptr;
				return;
			}

			// Check for overflow
			if ( size_ > (std::numeric_limits<size_t>::max() - alignment + 1) / sizeof( T ) ) {
				throw std::overflow_error( "Size too large, causing overflow in aligned_size calculation." );
			}

			aligned_size_ = calculateAlignedSize( size_ );

			std::cout << "Allocating buffer of size: " << size_ << " with aligned size: " << aligned_size_ << std::endl;

			data_ = static_cast<T*>(mr_->allocate( aligned_size_, alignment ));

			std::cout << "Allocated buffer of size: " << aligned_size_
				<< " Pointer: " << std::hex << reinterpret_cast<uintptr_t>( data_ ) << std::dec
				<< " (Total: " << Compute::MemoryStats::currentUsage << ")" << std::endl;

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
			: size_( size ), aligned_size_( 0 ), data_( data_ptr ), mr_( nullptr ) {
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
			if ( mr_ && data_ ) {
				std::cout << "~TensorBuffer() calling deallocate of aligned size: " << aligned_size_
					<< " (Total: " << Compute::MemoryStats::currentUsage << ")" << std::endl;

				mr_->deallocate( data_, aligned_size_ );
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

			// Special case for resizing to zero
			if ( new_size == 0 ) {
				if ( data_ ) {
					mr_->deallocate( data_, aligned_size_ );
					data_ = nullptr;
				}
				size_ = 0;
				aligned_size_ = 0;
				return;
			}

			// Check for overflow
			if ( new_size > (std::numeric_limits<size_t>::max() - alignment + 1) / sizeof( T ) ) {
				throw std::overflow_error( "New size too large, causing overflow in aligned_size calculation." );
			}

			size_t new_aligned_size = calculateAlignedSize( new_size );
			T* new_data = static_cast<T*>(mr_->allocate( new_aligned_size, alignment ));

			if ( data_ ) {
				// Copy existing data if needed
				size_t copy_size = std::min( size_, new_size ) * sizeof( T );
				if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> ) {
					std::memcpy( new_data, data_, copy_size );
				}
				else {
					Compute::cudaCheckStatus( cudaMemcpy( new_data, data_, copy_size, cudaMemcpyDeviceToDevice ) );
				}

				mr_->deallocate( data_, aligned_size_ );
			}

			data_ = new_data;
			size_ = new_size;
			aligned_size_ = new_aligned_size;

			// Fill remaining space if expanding from a previous non-zero size
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
		 * @brief Gets the aligned size of the buffer in bytes.
		 *
		 * This is the actual memory size allocated, which may be larger than
		 * size() * sizeof(TPrecision) due to alignment padding.
		 *
		 * @return size_t The aligned size in bytes.
		 */
		size_t aligned_size() const {
			return aligned_size_;
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
		 * @brief The actual size in bytes of the allocated memory, including alignment padding.
		 */
		size_t aligned_size_{ 0 };

		/**
		 * @brief A pointer to the allocated memory containing the buffer's data.
		 */
		T* data_{ nullptr };

		/**
		 * @brief Memory resource used for allocation/deallocation (null for externally managed buffers).
		 */
		std::unique_ptr<Compute::MemoryResource> mr_{ nullptr };

		/**
		 * @brief Calculates the aligned size in bytes for a given number of elements.
		 *
		 * @param num_elements The number of elements.
		 * @return size_t The aligned size in bytes.
		 */
		size_t calculateAlignedSize( size_t num_elements ) const {
			return (num_elements * sizeof( T ) + alignment - 1) & ~(alignment - 1);
		}

		/**
		 * @brief Creates a memory resource based on the template parameter.
		 *
		 * @return std::unique_ptr<Compute::MemoryResource> The created memory resource.
		 */
		std::unique_ptr<Compute::MemoryResource> createMemoryResource() {
			if constexpr ( TrackMemory ) {
				auto resource = std::make_unique<MR>();
				return std::make_unique<Compute::TrackedMemoryResource>( resource.release() );
			}
			else {
				return std::make_unique<MR>();
			}
		}

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
			if ( size_ == 0 ) {
				return;
			}

			if constexpr ( MR::is_host_accessible ) {
				std::fill( data_, data_ + size_, value );
			}
			else if constexpr ( MR::is_device_accessible && !MR::is_host_accessible ) {
				std::vector<T> temp( size_, value );
				Compute::cudaCheckStatus( cudaMemcpy( data_, temp.data(), size_ * sizeof( T ), cudaMemcpyHostToDevice ) );
			}
			// FUTURE: Fallback for any other possible memory resource types
			else {
				// This could be a custom memory resource or a future addition
				// Consider a more generic approach or a warning/error
				std::cerr << "Warning: fillBuffer not implemented for this memory resource type" << std::endl;
			}
		}
	};
}
