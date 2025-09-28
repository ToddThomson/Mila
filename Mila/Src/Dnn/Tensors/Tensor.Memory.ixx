module;
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

export module Dnn.Tensor:Memory;

import :Interface;

import Dnn.TensorTraits;
import Dnn.TensorPtr;
import Compute.MemoryResource;

//namespace Mila::Dnn
//{
//    template <typename MR>
//    static constexpr bool is_host_accessible() {
//        //if constexpr ( std::is_same_v<MR, Compute::DynamicMemoryResource> ) {
//        //    // For DynamicMemoryResource, we need to create an instance to check
//        //    // Since we don't have access to a runtime instance here, we'll make a conservative choice
//        //    // using std::is_same to avoid compilation errors
//        //    return true;  // This is a conservative choice - assuming it might be host accessible
//        //}
//        //else {
//        //    // For other memory resources, use the static constexpr member
//            return MR::is_host_accessible;
//        //}
//    }
//
//    template <typename MR>
//    static constexpr bool is_device_accessible() {
//        //if constexpr ( std::is_same_v<MR, Compute::DynamicMemoryResource> ) {
//        //    // Same conservative approach for device accessibility
//        //    return true;  // This is a conservative choice - assuming it might be device accessible
//        //}
//        //else {
//        //    // For other memory resources, use the static constexpr member
//            return MR::is_device_accessible;
//        //}
//    }
//
//    /**
//     * @brief Gets a mutable pointer to the tensor data.
//     * @return Mutable pointer to the tensor data
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    TElementType* Tensor<TElementType, TMemoryResource>::data() {
//        return buffer_->data();
//    }
//
//    /**
//     * @brief Gets a const pointer to the tensor data.
//     * @return Const pointer to the tensor data
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    const TElementType* Tensor<TElementType, TMemoryResource>::data() const {
//        return buffer_->data();
//    }
//
//    /**
//    * @brief Gets a pointer to the tensor data with memory type safety.
//    *
//    * Returns a memory-type-aware pointer that prevents unsafe host access
//    * to device memory at compile time.
//    *
//    * @return A TensorPtr wrapper that enforces memory access safety
//    */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    auto Tensor<TElementType, TMemoryResource>::dataPtr() {
//        if constexpr ( is_host_accessible<TMemoryResource>() ) {
//            return HostPtr<TElementType>( buffer_->data() );
//        }
//        else {
//            return DevicePtr<TElementType>( buffer_->data() );
//        }
//    }
//
//    /**
//     * @brief Gets a const pointer to the tensor data with memory type safety.
//     *
//     * Returns a memory-type-aware pointer that prevents unsafe host access
//     * to device memory at compile time.
//     *
//     * @return A const TensorPtr wrapper that enforces memory access safety
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    const auto Tensor<TElementType, TMemoryResource>::dataPtr() const {
//        if constexpr ( is_host_accessible<TMemoryResource>() ) {
//            return HostPtr<const TElementType>( buffer_->data() );
//        }
//        else {
//            return DevicePtr<const TElementType>( buffer_->data() );
//        }
//    }
//
//    /**
//     * @brief Gets a raw pointer to the tensor data for use in CUDA kernels.
//     *
//     * This method is intended for internal use by CUDA operation implementations.
//     * Use with caution as it bypasses memory type safety.
//     *
//     * @return Raw pointer to the tensor data
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    void* Tensor<TElementType, TMemoryResource>::rawData() {
//        return buffer_->data();
//    }
//
//    /**
//     * @brief Gets a const raw pointer to the tensor data for use in CUDA kernels.
//     *
//     * This method is intended for internal use by CUDA operation implementations.
//     * Use with caution as it bypasses memory type safety.
//     *
//     * @return Const raw pointer to the tensor data
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    const void* Tensor<TElementType, TMemoryResource>::rawData() const {
//        return buffer_->data();
//    }
//
//    /**
//    * @brief Copies data from another tensor into this tensor.
//    *
//    * This method copies the contents of the source tensor to this tensor. Both tensors must have
//    * the same shape. The data is copied using the appropriate memory transfer mechanism based on
//    * the source and destination memory resource types.
//    *
//    * @tparam SrcMemoryResource The memory resource type of the source tensor.
//    * @param src The source tensor to copy data from.
//    * @throws std::runtime_error If the shapes don't match or if a CUDA memory transfer operation fails.
//    */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    template<typename SrcMemoryResource>
//    void Tensor<TElementType, TMemoryResource>::copyFrom( const Tensor<TElementType, SrcMemoryResource>& src ) {
//        if ( shape_ != src.shape() ) {
//            throw std::runtime_error( "Cannot copy from tensor with different shape." );
//        }
//
//        if ( size_ == 0 ) {
//            return;
//        }
//
//        // Determine the appropriate copy method based on memory resource types
//        if constexpr ( is_device_accessible<TMemoryResource>() && is_device_accessible<SrcMemoryResource>() ) {
//            // Device to device transfer - use raw pointers for CUDA operations
//            cudaError_t status = cudaMemcpy( rawData(), src.rawData(),
//                size_ * sizeof( TElementType ),
//                cudaMemcpyDeviceToDevice );
//            if ( status != cudaSuccess ) {
//                throw std::runtime_error( "CUDA memory transfer failed: " +
//                    std::string( cudaGetErrorString( status ) ) );
//            }
//        }
//        else if constexpr ( is_host_accessible<TMemoryResource>() && is_device_accessible<SrcMemoryResource>() ) {
//            // Host destination, device source - need to bring to host first
//            auto host_src = src.template to<Compute::HostMemoryResource>();
//            std::copy( host_src.data(), host_src.data() + size_, data() );
//        }
//        else if constexpr ( is_device_accessible<TMemoryResource>() && is_host_accessible<SrcMemoryResource>() ) {
//            // Device destination, host source
//            cudaError_t status = cudaMemcpy( rawData(), src.rawData(),
//                size_ * sizeof( TElementType ),
//                cudaMemcpyHostToDevice );
//            if ( status != cudaSuccess ) {
//                throw std::runtime_error( "CUDA memory transfer failed: " +
//                    std::string( cudaGetErrorString( status ) ) );
//            }
//        }
//        else {
//            // Host to host transfer (use standard copy)
//            std::copy( src.data(), src.data() + size_, data() );
//        }
//
//        // Copy name if source has one and this tensor doesn't
//        if ( name_.empty() && !src.getName().empty() ) {
//            setName( src.getName() );
//        }
//    }
//
//    /**
//    * @brief Converts this tensor to use a different memory resource.
//    *
//    * This method creates a new tensor with the same shape and data but using a different memory resource.
//    * The data is copied from the current tensor to the new tensor using the appropriate memory transfer
//    * mechanism based on the source and destination memory resource types.
//    *
//    * @tparam TNewMR The target memory resource type.
//    * @return Tensor<TElementType, TNewMR> A new tensor with the specified memory resource type.
//    * @throws std::runtime_error If a CUDA memory transfer operation fails.
//    */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    template<typename TOtherMR>
//        requires std::is_base_of_v<Compute::MemoryResource, TOtherMR>
//    Tensor<TElementType, TOtherMR> Tensor<TElementType, TMemoryResource>::to() const {
//        Tensor<TElementType, TOtherMR> new_tensor( shape_ );
//
//        if ( !name_.empty() ) {
//            new_tensor.setName( name_ );
//        }
//
//        if ( size_ > 0 ) {
//            std::unique_ptr<Compute::MemoryResource> destResource = std::make_unique<TOtherMR>();
//
//            destResource->memcpy( new_tensor.data(), this->data(), size_ * sizeof( TElementType ) );
//        }
//
//        return new_tensor;
//    }
//
//    /**
//    * @brief Converts the tensor to a host-accessible memory resource.
//    *
//    * This is a convenience method that creates a new tensor with host-accessible memory.
//    * If the current tensor is already host-accessible, it creates a shallow copy before
//    * converting to the specified host memory resource type.
//    *
//    * @tparam HostAccessibleMR The target host memory resource type (defaults to HostMemoryResource)
//    * @return A new tensor with host-accessible memory
//    */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    template<typename HostAccessibleMR>
//    Tensor<TElementType, HostAccessibleMR> Tensor<TElementType, TMemoryResource>::toHostAccessible() const {
//        if constexpr ( std::is_same_v<TMemoryResource, Compute::HostMemoryResource> ||
//            std::is_same_v<TMemoryResource, Compute::CudaPinnedMemoryResource> ||
//            std::is_same_v<TMemoryResource, Compute::CudaManagedMemoryResource> ) {
//            // Create a shallow copy if the memory is already host-accessible
//            Tensor<TElementType, TMemoryResource> result( *this );
//            return result.template to<HostAccessibleMR>();
//        }
//        else {
//            // Create a new host-accessible tensor and copy the data
//            return this->template to<HostAccessibleMR>();
//        }
//    }
//
//    /**
//    * @brief Creates a deep copy of this tensor.
//    *
//    * Unlike the copy constructor which shares the underlying buffer,
//    * this method creates a completely independent copy with its own data buffer.
//    *
//    * @return Tensor<TElementType, TMemoryResource> A deep copy of this tensor
//    */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    Tensor<TElementType, TMemoryResource> Tensor<TElementType, TMemoryResource>::clone() const {
//        // Create a new tensor with the same shape
//        Tensor<TElementType, TMemoryResource> result( shape_ );
//
//        // Copy data from the current tensor to the new tensor
//        if ( size_ > 0 ) {
//            if constexpr ( is_host_accessible<TMemoryResource>() ) {
//                std::copy( data(), data() + size_, result.data() );
//            }
//            else {
//                cudaMemcpy( result.data(), data(), size_ * sizeof( TElementType ), cudaMemcpyDeviceToDevice );
//            }
//        }
//
//        if ( !name_.empty() ) {
//            result.setName( name_ );
//        }
//
//        return result;
//    }
//
//    /**
//    * @brief Fill the tensor with a uniform value.
//    *
//    * Sets all elements in the tensor to the specified value, handling
//    * both host and device memory appropriately.
//    *
//    * @param value The value to fill the tensor with
//    */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    void Tensor<TElementType, TMemoryResource>::fill( const TElementType& value ) {
//        if constexpr ( is_host_accessible<TMemoryResource>() ) {
//            std::fill( buffer_->data(), buffer_->data() + size_, value );
//        }
//        else {
//            // Create a temporary host tensor, fill it, then copy back
//            auto host_tensor = to<Compute::CpuMemoryResource>();
//            host_tensor.fill( value );
//            *this = host_tensor.template to<TMemoryResource>();
//        }
//    }
//
//    /**
//    * @brief Compares this tensor with another tensor for element-wise equivalence within a tolerance.
//    *
//    * Performs an element-by-element comparison between this tensor and another tensor,
//    * handling different memory resource types and element types appropriately:
//    * - For floating-point types: Compares values within the specified tolerance
//    * - For integer types: Performs exact comparison
//    * - For device tensors: Automatically transfers data to host for comparison
//    *
//    * @tparam TOtherMR Memory resource type of the other tensor
//    * @param other The tensor to compare with
//    * @param tolerance Maximum allowed absolute difference between elements (for floating-point types)
//    * @param relative_tolerance Maximum allowed relative difference (for values far from zero)
//    * @return true if tensors are equivalent, false otherwise
//    */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    template<typename TOtherMR>
//        requires std::is_base_of_v<Compute::MemoryResource, TOtherMR>
//    bool Tensor<TElementType, TMemoryResource>::isEquivalentTo(
//        const Tensor<TElementType, TOtherMR>& other,
//        float tolerance,
//        float relative_tolerance ) const {
//
//        if ( shape_ != other.shape() ) {
//            return false;
//        }
//
//        // Empty tensors with same shape are equivalent
//        if ( size_ == 0 ) {
//            return true; 
//        }
//
//        // For device tensors, bring data to host for comparison
//        if constexpr ( !is_host_accessible<TMemoryResource>() || !is_host_accessible<TOtherMR>() ) {
//            auto this_host = this->template toHostAccessible<Compute::HostMemoryResource>();
//            auto other_host = other.template toHostAccessible<Compute::HostMemoryResource>();
//            return this_host.isEquivalentTo( other_host, tolerance, relative_tolerance );
//        }
//        else {
//            // Both tensors are host-accessible, can compare directly
//            if constexpr ( std::is_floating_point_v<TElementType> ||
//                std::is_same_v<TElementType, half> ||
//                std::is_same_v<TElementType, nv_bfloat16> ||
//                std::is_same_v<TElementType, __nv_fp8_e4m3> ||
//                std::is_same_v<TElementType, __nv_fp8_e5m2> ) {
//
//                // For floating point types - use both absolute and relative tolerance
//                for ( size_t i = 0; i < size_; ++i ) {
//                    float a, b;
//
//                    if constexpr ( std::is_same_v<TElementType, half> ) {
//                        a = __half2float( buffer_->data()[ i ] );
//                        b = __half2float( other.data()[ i ] );
//                    }
//                    else if constexpr ( std::is_same_v<TElementType, nv_bfloat16> ) {
//                        a = static_cast<float>( buffer_->data()[ i ] );
//                        b = static_cast<float>( other.data()[ i ] );
//                    }
//                    else if constexpr ( std::is_same_v<TElementType, __nv_fp8_e4m3> ||
//                        std::is_same_v<TElementType, __nv_fp8_e5m2> ) {
//                        a = static_cast<float>(buffer_->data()[ i ]);
//                        b = static_cast<float>(other.data()[ i ]);
//                    }
//                    else {
//                        a = static_cast<float>(buffer_->data()[ i ]);
//                        b = static_cast<float>(other.data()[ i ]);
//                    }
//
//                    // Check absolute difference for values near zero
//                    float abs_diff = std::abs( a - b );
//                    if ( abs_diff > tolerance ) {
//                        // For larger values, also check relative difference
//                        float abs_a = std::abs( a );
//                        float abs_b = std::abs( b );
//                        float largest = std::max( abs_a, abs_b );
//
//                        // Avoid division by zero or very small numbers
//                        if ( largest > 1e-5f && abs_diff / largest > relative_tolerance ) {
//                            return false;
//                        }
//                    }
//                }
//                return true;
//            }
//            else {
//                // For integer types - exact comparison
//                for ( size_t i = 0; i < size_; ++i ) {
//                    if ( buffer_->data()[ i ] != other.rawData()[ i ] ) {
//                        return false;
//                    }
//                }
//                return true;
//            }
//        }
//    }
//
//    //export template class Tensor<float, Compute::CpuMemoryResource>;
//	//export template class Tensor<float, Compute::CudaMemoryResource>;
//}