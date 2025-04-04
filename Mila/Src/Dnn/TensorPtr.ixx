/**
 * @file TensorPtr.ixx
 * @brief Memory-type-aware pointer wrappers for tensor data access.
 */

module;
#include <stdexcept>

export module Dnn.TensorPtr;

namespace Mila::Dnn
{
    /**
     * @brief Base tensor pointer class that wraps a raw pointer with memory-type safety.
     *
     * @tparam T Element type
     * @tparam IsHostAccessible Whether the pointer points to host-accessible memory
     */
    export template <typename T, bool IsHostAccessible>
        class TensorPtr {
        private:
            T* ptr_;

        public:
            /**
             * @brief Constructs a new TensorPtr
             *
             * @param ptr Raw pointer to wrap
             */
            explicit TensorPtr( T* ptr ) noexcept : ptr_( ptr ) {}

            /**
             * @brief Gets the raw pointer (explicit conversion)
             *
             * @return T* Raw pointer
             */
            T* get() const noexcept {
                return ptr_;
            }

            /**
             * @brief Host-only: Dereference operator
             *
             * @return Reference to the element
             * @throws std::runtime_error if memory is not host-accessible
             */
            T& operator*() const {
                if constexpr ( IsHostAccessible ) {
                    return *ptr_;
                }
                else {
                    throw std::runtime_error( "Cannot dereference device memory pointer from host code" );
                }
            }

            /**
             * @brief Host-only: Subscript operator
             *
             * @param index Element index
             * @return Reference to the element
             * @throws std::runtime_error if memory is not host-accessible
             */
            T& operator[]( size_t index ) const {
                if constexpr ( IsHostAccessible ) {
                    return ptr_[ index ];
                }
                else {
                    throw std::runtime_error( "Cannot access device memory from host code" );
                }
            }

            /**
             * @brief Host-only: Pointer addition
             *
             * @param offset Offset to add
             * @return TensorPtr at the offset position
             * @throws std::runtime_error if memory is not host-accessible
             */
            TensorPtr operator+( size_t offset ) const {
                if constexpr ( IsHostAccessible ) {
                    return TensorPtr( ptr_ + offset );
                }
                else {
                    throw std::runtime_error( "Cannot perform pointer arithmetic on device memory from host code" );
                }
            }

            /**
             * @brief Explicit conversion to raw pointer for CUDA kernels
             *
             * @return Raw pointer for use in CUDA kernels
             */
            operator T* () const {
                return ptr_;
            }
    };

    /**
     * @brief Type alias for host memory pointers
     */
    export template <typename T>
        using HostPtr = TensorPtr<T, true>;

    /**
     * @brief Type alias for device memory pointers
     */
    export template <typename T>
        using DevicePtr = TensorPtr<T, false>;

    /**
     * @brief Gets a raw pointer from a TensorPtr (similar to thrust::raw_pointer_cast)
     *
     * @tparam T Element type
     * @tparam IsHostAccessible Whether the pointer is host-accessible
     * @param ptr TensorPtr to convert
     * @return T* Raw pointer
     */
    export template <typename T, bool IsHostAccessible>
        T* raw_pointer_cast( const TensorPtr<T, IsHostAccessible>& ptr ) noexcept {
        return ptr.get();
    }
}
