/**
 * @file TensorPtr.ixx
 * @brief Lightweight pointer wrapper for type-safe tensor data access
 *
 * Provides a simple pointer wrapper for tensor data with support for
 * pointer arithmetic and array indexing. Memory accessibility is enforced
 * at the Tensor level via requires clauses, not at the pointer level.
 */

module;
#include <cstddef>

export module Dnn.TensorPtr;

namespace Mila::Dnn
{
    /**
     * @brief Lightweight pointer wrapper for tensor data
     *
     * Simple wrapper around a raw pointer that provides array indexing,
     * pointer arithmetic, and dereferencing. No runtime memory accessibility
     * checks - safety is enforced at compile-time via Tensor's requires clauses.
     *
     * @tparam T Element type (may be const-qualified)
     *
     * @note Memory accessibility is guaranteed by Tensor.data() requires clause
     * @note Designed for zero-overhead abstraction
     * @note Supports both const and non-const element types
     */
    export template <typename T>
    class TensorPtr {
    private:
        T* ptr_;

    public:
        /**
         * @brief Constructs a TensorPtr from a raw pointer
         *
         * @param ptr Raw pointer to wrap (may be nullptr)
         */
        explicit constexpr TensorPtr(T* ptr) noexcept : ptr_(ptr) {}

        /**
         * @brief Gets the raw pointer
         *
         * @return Raw pointer
         */
        constexpr T* get() const noexcept {
            return ptr_;
        }

        /**
         * @brief Dereference operator
         *
         * @return Reference to the element
         *
         * @note Behavior is undefined if ptr_ is nullptr
         * @note No runtime checks - caller must ensure validity
         */
        constexpr T& operator*() const noexcept {
            return *ptr_;
        }

        /**
         * @brief Array subscript operator
         *
         * @param index Element index
         * @return Reference to the element at the given index
         *
         * @note Behavior is undefined if index is out of bounds
         * @note No runtime checks - caller must ensure validity
         */
        constexpr T& operator[](size_t index) const noexcept {
            return ptr_[index];
        }

        /**
         * @brief Pointer arithmetic - addition
         *
         * @param offset Offset to add
         * @return TensorPtr at the offset position
         */
        constexpr TensorPtr operator+(size_t offset) const noexcept {
            return TensorPtr(ptr_ + offset);
        }

        /**
         * @brief Pointer arithmetic - subtraction
         *
         * @param offset Offset to subtract
         * @return TensorPtr at the offset position
         */
        constexpr TensorPtr operator-(size_t offset) const noexcept {
            return TensorPtr(ptr_ - offset);
        }

        /**
         * @brief Pre-increment operator
         *
         * @return Reference to this TensorPtr after increment
         */
        constexpr TensorPtr& operator++() noexcept {
            ++ptr_;
            return *this;
        }

        /**
         * @brief Post-increment operator
         *
         * @return TensorPtr before increment
         */
        constexpr TensorPtr operator++(int) noexcept {
            TensorPtr tmp(*this);
            ++ptr_;
            return tmp;
        }

        /**
         * @brief Implicit conversion to raw pointer for C API interop
         *
         * @return Raw pointer
         *
         * @note Enables seamless integration with C APIs and CUDA kernels
         */
        constexpr operator T*() const noexcept {
            return ptr_;
        }

        /**
         * @brief Equality comparison
         *
         * @param other TensorPtr to compare with
         * @return true if pointers are equal
         */
        constexpr bool operator==(const TensorPtr& other) const noexcept {
            return ptr_ == other.ptr_;
        }

        /**
         * @brief Inequality comparison
         *
         * @param other TensorPtr to compare with
         * @return true if pointers are not equal
         */
        constexpr bool operator!=(const TensorPtr& other) const noexcept {
            return ptr_ != other.ptr_;
        }

        /**
         * @brief Boolean conversion for nullptr checks
         *
         * @return true if pointer is non-null
         */
        explicit constexpr operator bool() const noexcept {
            return ptr_ != nullptr;
        }
    };

    /**
     * @brief Gets a raw pointer from a TensorPtr
     *
     * @tparam T Element type
     * @param ptr TensorPtr to convert
     * @return Raw pointer
     */
    export template <typename T>
    constexpr T* raw_pointer_cast(const TensorPtr<T>& ptr) noexcept {
        return ptr.get();
    }

    /**
     * @brief Deduction guide for TensorPtr
     *
     * Allows TensorPtr construction without explicit template arguments
     */
    export template<typename T>
    TensorPtr(T*) -> TensorPtr<T>;
}
