module;
#include <cstdio>  
#include <cstdlib>  
#include <iostream>  

export module Helpers.Memory;

namespace Mila::Helpers
{
	/**
	 * @brief Allocates memory with error checking.
	 *
	 * @param size The size of the memory to be allocated.
	 * @param file The source file name where the function is called.
	 * @param line The line number in the source file where the function is called.
	 * @return void* Pointer to the allocated memory.
	 */
	extern inline void* malloc_check( size_t size, const char* file, int line ) {
		void* ptr = malloc( size );
		if ( ptr == nullptr ) {
			std::cerr << "Error: Memory allocation failed at " << file << ":" << line << "\n";
			std::cerr << "Error details:\n";
			std::cerr << "  File: " << file << "\n";
			std::cerr << "  Line: " << line << "\n";
			std::cerr << "  Size: " << size << " bytes\n";
			std::exit( EXIT_FAILURE );
		}
		return ptr;
	}
	/**
	 * @brief Allocates memory with error checking.
	 *
	 * @param size The size of the memory to be allocated.
	 * @return void* Pointer to the allocated memory.
	 */
	export void* mallocCheck( size_t size ) {
		return malloc_check( size, __FILE__, __LINE__ );
	}
} // namespace Mila::Helpers