/**
 * @file TensorBuffer.ixx
 * @brief Device-agnostic memory management layer for tensor data using abstract data types
 *
 * This module provides a sophisticated memory management system for tensor data that operates
 * across heterogeneous compute environments (CPU, CUDA, Metal, OpenCL, Vulkan) using abstract
 * TensorDataType enumeration. The system handles device-specific alignment optimization and
 * automatic memory resource selection based on data type compatibility constraints.
 *
 * Key architectural features:
 * - Abstract data type system eliminates device-specific compilation dependencies
 * - Automatic memory alignment optimization for target hardware (SIMD, CUDA warps)
 * - Device-agnostic memory operations with compile-time dispatch
 * - Optional memory allocation tracking and profiling capabilities
 * - Exception-safe memory management with strong guarantees
 */

module;
#include <memory>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <cstring>

export module Dnn.TensorBuffer;

import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.MemoryResource;
import Compute.MemoryResourceTracker;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.DeviceContext;

namespace Mila::Dnn
{
	namespace Detail
	{
		/**
		 * @brief CUDA warp size alignment for optimal GPU memory access patterns
		 *
		 * Defines the alignment boundary required for optimal memory coalescing
		 * in CUDA warp-level operations, ensuring maximum memory throughput
		 * in GPU kernels and device functions.
		 */
		constexpr size_t CUDA_WARP_SIZE = 32;

		/**
		 * @brief AVX-512 alignment boundary for optimal CPU SIMD operations
		 *
		 * Defines the alignment requirement for maximum efficiency with
		 * AVX-512 vector instructions, enabling optimal vectorized operations
		 * on modern CPU architectures.
		 */
		constexpr size_t CPU_SIMD_ALIGN = 64;

		/**
		 * @brief Determines optimal memory alignment based on memory resource and data type
		 *
		 * Calculates the appropriate memory alignment boundary considering both the
		 * memory resource characteristics (CPU vs GPU) and data type requirements.
		 * This ensures optimal performance across different hardware architectures.
		 *
		 * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
		 * @tparam MR Memory resource type determining allocation strategy
		 * @return Optimal alignment boundary in bytes for the given configuration
		 */
		template<TensorDataType TDataType, typename MR>
		constexpr size_t get_alignment() {
			if constexpr (std::is_same_v<MR, Compute::CudaDeviceMemoryResource>) {
				return CUDA_WARP_SIZE * TensorDataTypeTraits<TDataType>::size_in_bytes;
			}
			else {
				return CPU_SIMD_ALIGN;
			}
		}

		/**
		 * @brief Calculates storage size in bytes for given logical element count
		 *
		 * Computes the required storage bytes for a given number of logical elements
		 * of the specified tensor data type, handling potential overflow conditions.
		 *
		 * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
		 * @param logical_size Number of logical elements
		 * @return Required storage size in bytes
		 * @throws std::overflow_error If calculation would overflow
		 */
		template<TensorDataType TDataType>
		constexpr size_t getStorageSize( size_t logical_size ) {
			constexpr size_t element_size = TensorDataTypeTraits<TDataType>::size_in_bytes;

			if (logical_size > std::numeric_limits<size_t>::max() / element_size) {
				throw std::overflow_error( "Storage size calculation would overflow." );
			}

			return logical_size * element_size;
		}
	}

	/**
	 * @brief Device-agnostic buffer for storing tensor data with abstract type system
	 *
	 * Advanced memory management container providing efficient storage for tensor data
	 * across heterogeneous compute environments using abstract TensorDataType enumeration.
	 * Supports device-specific alignment optimization and automatic compatibility validation.
	 *
	 * Core architectural principles:
	 * - Abstract data types prevent device-specific compilation issues
	 * - Automatic memory alignment optimization for target hardware
	 * - Support for precision formats including FP32, FP16, BF16, FP8, and integer types
	 * - Device-agnostic memory operations with compile-time dispatch optimization
	 * - Optional allocation tracking for memory profiling and debugging
	 * - Exception-safe design with strong safety guarantees
	 *
	 * The buffer supports both owned memory management and external memory wrapping,
	 * enabling integration with existing memory pools and external libraries while
	 * maintaining optimal performance characteristics.
	 *
	 * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
	 * @tparam TMemoryResource Memory resource type determining allocation strategy and device targeting
	 * @tparam TrackMemory When true, enables detailed memory allocation tracking and profiling
	 *
	 * @note Thread Safety: Buffer operations are not thread-safe; external synchronization required
	 * @note Exception Safety: Strong guarantee for most operations; basic guarantee for constructors
	 * @note Memory Layout: Automatic optimization for device-specific alignment
	 * @note Memory Transfers: Transfer operations belong at the Tensor level where device contexts are meaningful
	 *
	 * @see TensorDataType for supported abstract data type enumeration
	 * @see TensorDataTypeTraits for compile-time data type characteristics
	 * @see MemoryResource for device memory abstraction layer
	 *
	 * Example usage:
	 * @code
	 * // CPU buffer with FP32 data type
	 * TensorBuffer<TensorDataType::FP32, Compute::CpuMemoryResource> cpuBuffer(device_context, 100);
	 *
	 * // GPU buffer with FP16 data type
	 * TensorBuffer<TensorDataType::FP16, Compute::CudaDeviceMemoryResource> gpuBuffer(device_context, 100);
	 *
	 * // Buffer with memory tracking enabled
	 * TensorBuffer<TensorDataType::BF16, Compute::CudaDeviceMemoryResource, true> trackedBuffer(device_context, 100);
	 * @endcode
	 */
	export template <TensorDataType TDataType, typename TMemoryResource, bool TrackMemory = false>
		requires isValidTensor<TDataType, TMemoryResource>
	class TensorBuffer {
	public:

		using DataType = TensorDataType;                                           ///< Abstract data type enumeration
		using DataTypeTraits = TensorDataTypeTraits<TDataType>;                   ///< Compile-time data type characteristics

		static constexpr TensorDataType data_type = TDataType;                     ///< Compile-time data type constant
		static constexpr size_t element_size = DataTypeTraits::size_in_bytes;      ///< Storage size per element
		static constexpr size_t alignment = Detail::get_alignment<TDataType, TMemoryResource>(); ///< Optimal memory alignment
		static constexpr bool is_float_type = DataTypeTraits::is_float_type;       ///< Floating-point type classification
		static constexpr bool is_integer_type = DataTypeTraits::is_integer_type;   ///< Integer type classification
		static constexpr bool is_device_only = DataTypeTraits::is_device_only;     ///< Device-only type restriction

		/**
		 * @brief Constructs buffer with owned memory and zero initialization
		 *
		 * Allocates optimally aligned memory using the specified memory resource
		 * and initializes all memory to zero for deterministic behavior.
		 * Memory alignment is optimized based on data type and target hardware.
		 *
		 * @param device_context Device context for memory resource initialization
		 * @param logical_size Number of logical elements to store in the buffer
		 *
		 * @throws std::invalid_argument If device_context is null
		 * @throws std::overflow_error If size causes overflow in storage calculations
		 * @throws std::bad_alloc If memory allocation fails
		 * @throws std::runtime_error If memory resource operations fail
		 *
		 * @note Memory is aligned according to hardware optimization requirements
		 * @note Zero-sized buffers are handled efficiently without allocation
		 */
		explicit TensorBuffer( int device_id, size_t logical_size )
			: device_id_( device_id ), logical_size_( logical_size ), storage_bytes_( 0 ), mr_( createMemoryResource() ) {

			if (logical_size_ == 0) {
				data_ = nullptr;
				return;
			}

			// Calculate storage requirements
			storage_bytes_ = Detail::getStorageSize<TDataType>( logical_size_ );

			// Check for overflow in alignment calculations
			if (storage_bytes_ > (std::numeric_limits<size_t>::max() - alignment + 1)) {
				throw std::overflow_error( "Storage size too large, causing overflow in alignment calculation." );
			}

			aligned_size_ = calculateAlignedSize( storage_bytes_ );

			data_ = static_cast<std::byte*>(mr_->allocate( aligned_size_, alignment ));

			if constexpr (TrackMemory) {
				logAllocation();
			}

			// Initialize memory to zero for deterministic behavior
			// mr_->memset( data_, 0, storage_bytes_ );
		}


		/**
		 * @brief Destructor with automatic memory cleanup via RAII
		 *
		 * Automatically deallocates owned memory through the memory resource.
		 * External memory is not deallocated, maintaining safe resource management.
		 * Provides optional deallocation tracking for memory profiling.
		 */
		~TensorBuffer() {
			if (mr_ && data_) {
				if constexpr (TrackMemory) {
					logDeallocation();
				}
				mr_->deallocate( data_, aligned_size_, alignment );
			}
		}

		/**
		 * @brief Copy operations explicitly deleted for performance safety
		 *
		 * Prevents accidental expensive copy operations involving large memory
		 * transfers across different memory spaces and devices.
		 */
		TensorBuffer( const TensorBuffer& ) = delete;
		TensorBuffer& operator=( const TensorBuffer& ) = delete;

		/**
		 * @brief Move constructor for efficient ownership transfer
		 *
		 * Transfers all resources from source buffer without memory copying,
		 * leaving source in valid but empty state.
		 */
		TensorBuffer( TensorBuffer&& other ) noexcept
			: device_id_( other.device_id_ ),
			logical_size_( other.logical_size_ ),
			storage_bytes_( other.storage_bytes_ ),
			aligned_size_( other.aligned_size_ ),
			data_( other.data_ ),
			mr_( std::move( other.mr_ ) ) {

			other.device_id_ = 0;
			other.logical_size_ = 0;
			other.storage_bytes_ = 0;
			other.aligned_size_ = 0;
			other.data_ = nullptr;
		}

		/**
		 * @brief Move assignment operator for efficient ownership transfer
		 *
		 * Safely transfers ownership while properly cleaning up existing resources.
		 */
		TensorBuffer& operator=( TensorBuffer&& other ) noexcept {
			if (this != &other) {
				// Clean up existing resources
				if (mr_ && data_) {
					mr_->deallocate( data_, aligned_size_, alignment );
				}

				// Transfer resources
				device_id_ = other.device_id_;
				logical_size_ = other.logical_size_;
				storage_bytes_ = other.storage_bytes_;
				aligned_size_ = other.aligned_size_;
				data_ = other.data_;
				mr_ = std::move( other.mr_ );

				// Reset source
				other.device_id_ = -1;
				other.logical_size_ = 0;
				other.storage_bytes_ = 0;
				other.aligned_size_ = 0;
				other.data_ = nullptr;
			}
			return *this;
		}

		/**
		 * @brief Returns the number of logical elements in the buffer
		 *
		 * @return Number of logical elements stored in the buffer
		 */
		size_t size() const noexcept {
			return logical_size_;
		}

		/**
		 * @brief Returns the storage size in bytes
		 *
		 * Provides the actual memory storage used.
		 *
		 * @return Storage size in bytes
		 */
		size_t storageBytes() const noexcept {
			return storage_bytes_;
		}

		/**
		 * @brief Returns the aligned memory allocation size in bytes
		 *
		 * This represents the actual memory allocated, which includes
		 * alignment padding for optimal hardware performance.
		 *
		 * @return Total allocated memory size including alignment padding
		 */
		size_t alignedSize() const noexcept {
			return aligned_size_;
		}

		/**
		 * @brief Checks if buffer memory is properly aligned
		 *
		 * Verifies that the buffer's data pointer meets the alignment
		 * requirements for optimal performance on the target hardware.
		 *
		 * @return true if properly aligned, false otherwise
		 */
		bool isAligned() const noexcept {
			return reinterpret_cast<std::uintptr_t>(data_) % alignment == 0;
		}

		/**
		 * @brief Checks if the buffer is empty
		 *
		 * @return true if buffer contains no logical elements, false otherwise
		 */
		bool empty() const noexcept {
			return logical_size_ == 0;
		}

		/**
		 * @brief Returns raw pointer to buffer data
		 *
		 * Provides direct access to the underlying memory buffer for
		 * performance-critical operations and device kernel interfacing.
		 *
		 * @return Raw pointer to buffer memory
		 *
		 * @warning No type safety or bounds checking
		 * @note Use with appropriate casting based on data type requirements
		 */
		void* rawData() noexcept {
			return data_;
		}

		/**
		 * @brief Returns const raw pointer to buffer data
		 *
		 * Provides read-only access to the underlying memory buffer.
		 *
		 * @return Const raw pointer to buffer memory
		 *
		 * @warning No type safety or bounds checking
		 */
		const void* rawData() const noexcept {
			return data_;
		}

		/**
		 * @brief Resizes buffer preserving existing data when possible
		 *
		 * Allocates new optimally aligned memory, copies existing data up to
		 * the minimum of old and new sizes. Provides strong exception safety.
		 *
		 * @param new_logical_size New number of logical elements
		 *
		 * @throws std::runtime_error If buffer uses external memory
		 * @throws std::overflow_error If new size causes storage overflow
		 * @throws std::bad_alloc If memory allocation fails
		 *
		 * @note Preserves existing data up to minimum of old/new sizes
		 * @note New elements beyond old size are zero-initialized
		 */
		void resize( size_t new_logical_size ) {
			if (!mr_) {
				throw std::runtime_error( "Cannot resize buffer with external memory management." );
			}

			if (logical_size_ == new_logical_size) {
				return;
			}

			// Handle resize to zero
			if (new_logical_size == 0) {
				if (data_) {
					mr_->deallocate( data_, aligned_size_, alignment );
					data_ = nullptr;
				}
				logical_size_ = 0;
				storage_bytes_ = 0;
				aligned_size_ = 0;
				return;
			}

			// Calculate new storage requirements
			size_t new_storage_bytes = Detail::getStorageSize<TDataType>( new_logical_size );

			if (new_storage_bytes > (std::numeric_limits<size_t>::max() - alignment + 1)) {
				throw std::overflow_error( "New storage size too large for alignment calculation." );
			}

			size_t new_aligned_size = calculateAlignedSize( new_storage_bytes );

			std::byte* new_data = nullptr;

			try {
				new_data = static_cast<std::byte*>(mr_->allocate( new_aligned_size, alignment ));

				// Copy existing data if present
				// FIXME:
				//if (data_ && storage_bytes_ > 0) {
				//	size_t copy_bytes = std::min( storage_bytes_, new_storage_bytes );
				//	mr_->memcpy( new_data, data_, copy_bytes );
				//}

				//// Zero-initialize new memory beyond copied data
				//if (new_storage_bytes > storage_bytes_) {
				//	size_t zero_bytes = new_storage_bytes - storage_bytes_;
				//	mr_->memset( new_data + storage_bytes_, 0, zero_bytes );
				//}

				// Clean up old memory after successful allocation and copy
				if (data_) {
					mr_->deallocate( data_, aligned_size_, alignment );
				}

				// Update buffer state
				data_ = new_data;
				logical_size_ = new_logical_size;
				storage_bytes_ = new_storage_bytes;
				aligned_size_ = new_aligned_size;

			}
			catch (...) {
				if (new_data) {
					mr_->deallocate( new_data, new_aligned_size, alignment );
				}
				throw;
			}
		}

		/**
		 * @brief Returns pointer to the memory resource managing this buffer's storage
		 *
		 * Provides access to the memory resource for efficient dispatch optimization,
		 * zero-copy operations when memory resources are compatible, and type-safe
		 * downcasting to specific memory resource types.
		 *
		 * @return Pointer to memory resource (nullptr if using external memory)
		 *
		 * @note Returns nullptr for buffers constructed with external memory
		 * @note For owned memory, returns pointer to the underlying memory resource
		 * @note Can be safely cast to TMemoryResource* based on template parameter
		 * @note Enables efficient memory resource compatibility checks
		 *
		 * Example:
		 * @code
		 * TensorBuffer<TensorDataType::FP32, CudaMemoryResource> buffer(ctx, 100);
		 * auto* mr = buffer.getMemoryResource();
		 *
		 * // Type-safe downcast check
		 * if (auto* cuda_mr = dynamic_cast<CudaMemoryResource*>(mr)) {
		 *     // Use CUDA-specific operations
		 * }
		 * @endcode
		 */
		Compute::MemoryResource* getMemoryResource() const noexcept {
			return mr_.get();
		}

	private:
		int device_id_{ -1 }; ///< Device Id for memory resource operations
		size_t logical_size_{ 0 };                                      ///< Number of logical elements in buffer
		size_t storage_bytes_{ 0 };                                     ///< Actual storage bytes 
		size_t aligned_size_{ 0 };                                      ///< Total allocated bytes including alignment
		std::byte* data_{ nullptr };                                    ///< Pointer to allocated memory buffer
		std::unique_ptr<Compute::MemoryResource> mr_{ nullptr };        ///< Memory resource for allocation (null for external)

		/**
		 * @brief Calculates aligned size for given storage requirements
		 *
		 * Ensures memory allocations meet hardware alignment requirements
		 * for optimal performance on target devices.
		 */
		size_t calculateAlignedSize( size_t storage_bytes ) const noexcept {
			return (storage_bytes + alignment - 1) & ~(alignment - 1);
		}

		/**
		 * @brief Creates appropriate memory resource with device context
		 *
		 * Instantiates memory resource with device context and optional allocation tracking
		 * wrapper for debugging and profiling purposes.
		 */
		std::unique_ptr<Compute::MemoryResource> createMemoryResource() {
			if constexpr (TrackMemory) {
				auto resource = std::make_unique<TMemoryResource>( device_id_ );
				return std::make_unique<Compute::TrackedMemoryResource>( resource.release() );
			}
			else {
				return std::make_unique<TMemoryResource>( device_id_ );
			}
		}

		/**
		 * @brief Logs memory allocation for tracking and profiling
		 */
		void logAllocation() const {
			if constexpr (TrackMemory) {
				std::cout << "TensorBuffer allocated: " << logical_size_ << " elements ("
					<< storage_bytes_ << " storage bytes, " << aligned_size_ << " aligned)"
					<< " DataType: " << DataTypeTraits::type_name
					<< " Pointer: 0x" << std::hex << reinterpret_cast<uintptr_t>(data_) << std::dec
					<< std::endl;
			}
		}

		/**
		 * @brief Logs memory deallocation for tracking and profiling
		 */
		void logDeallocation() const {
			if constexpr (TrackMemory) {
				std::cout << "TensorBuffer deallocated: " << logical_size_ << " elements ("
					<< storage_bytes_ << " storage bytes, " << aligned_size_ << " aligned)"
					<< " DataType: " << DataTypeTraits::type_name
					<< " Pointer: 0x" << std::hex << reinterpret_cast<uintptr_t>(data_) << std::dec
					<< std::endl;
			}
		}
	};
}