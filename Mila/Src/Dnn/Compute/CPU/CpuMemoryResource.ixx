module;
#include <memory_resource>
#include <cstddef>
#include <stdexcept>

export module Compute.CpuMemoryResource;

import Compute.MemoryResource;
import Compute.MemoryResourceProperties;

namespace Mila::Dnn::Compute
{
	/**
	* @brief A memory resource for CPU memory allocation.
	*/
	export class CpuMemoryResource : public MemoryResource {
	protected:
		/**
		* @brief Allocates memory with the specified size and alignment.
		* 
		* @param n The size of the memory to allocate.
		* @param alignment The alignment of the memory to allocate.
		* @return void* A pointer to the allocated memory.
		* @throws std::bad_alloc if the memory allocation fails.
		*/
		void* do_allocate( std::size_t n, std::size_t alignment ) override {
			if ( n == 0 ) {
				return nullptr;
			}

			// TODO: Implement alignment
			void* ptr = std::malloc( /*alignment,*/ n );
			
			if ( !ptr ) {
				throw std::bad_alloc();
			}
			
			return ptr;
		}

		/**
		* @brief Deallocates the memory at the specified pointer.
		* 
		* @param ptr A pointer to the memory to deallocate.
		* @param n The size of the memory to deallocate.
		* @param alignment The alignment of the memory to deallocate.
		*/
		void do_deallocate( void* ptr, std::size_t, std::size_t ) override {
			std::free( ptr );
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

		friend void get_property( CpuMemoryResource const&, Compute::DeviceAccessible ) noexcept {}
	};
}