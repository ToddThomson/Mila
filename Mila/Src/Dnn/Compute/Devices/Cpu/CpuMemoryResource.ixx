module;
#include <memory_resource>
#include <exception>

#ifdef _WIN32
#include <malloc.h>  // For _aligned_malloc and _aligned_free
#else
#include <cstdlib>   // For std::aligned_alloc and std::free
#endif


export module Compute.CpuMemoryResource;

import Compute.MemoryResource;
import Compute.MemoryResourceProperties;

namespace Mila::Dnn::Compute
{
	/**
	* @brief A memory resource for CPU memory allocation.
	*/
	export class CpuMemoryResource : public MemoryResource {
	public:
		/**
		* @brief Indicates if the memory resource is accessible by the CPU.
		*/
		static constexpr bool is_host_accessible = HostAccessible::is_host_accessible;

		/**
		* @brief Indicates if the memory resource is accessible by CUDA.
		*/
		static constexpr bool is_device_accessible = false;

	protected:
		/**
		* @brief Allocates memory with the specified size and alignment.
		* 
		* @param n The size of the memory to allocate.
		* @param alignment The alignment of the memory to allocate.
		* @return A pointer to the allocated memory.
		* @throws std::bad_alloc if the memory allocation fails.
		*/
		void* do_allocate( std::size_t n, std::size_t alignment ) override {
			if ( n == 0 ) {
				return nullptr;
			}

		#ifdef _WIN32
			void* ptr = _aligned_malloc( n, alignment );
		#else
			void* ptr = std::aligned_alloc( alignment, n );
		#endif

			if ( !ptr ) {
				throw std::bad_alloc();
			}

			return ptr;
		}

		/**
		* @brief Deallocates the memory pointed to by ptr.
		* 
		* @param ptr The pointer to the memory to deallocate.
		* @param n The size of the memory to deallocate.
		* @param alignment The alignment of the memory to deallocate.
		*/
		void do_deallocate( void* ptr, std::size_t, std::size_t ) override {
		#ifdef _WIN32
			_aligned_free( ptr );
		#else
			std::free( ptr ); // aligned_alloc can be freed with regular free
		#endif
		}
		
		/**
		* @brief Checks if this memory resource is equal to another memory resource.
		* 
		* @param other The other memory resource to compare with.
		* @return true if the memory resources are equal, false otherwise.
		*/
		bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override {
			return this == &other;
		}
	};

	/**
	* @brief Alias for CpuMemoryResource that represents host-accessible memory.
	*
	* This alias provides a semantic name that describes the memory's accessibility
	* characteristics rather than its implementation details. Use HostMemoryResource
	* when you need memory that can be directly accessed from host (CPU) code.
	*
	* @see CpuMemoryResource
	*/
	export using HostMemoryResource = CpuMemoryResource;
}