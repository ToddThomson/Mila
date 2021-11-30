#ifndef MILA_DNN_CUDNN_OPAQUE_HANDLE_H_
#define MILA_DNN_CUDNN_OPAQUE_HANDLE_H_

#include <memory>
#include <cudnn.h>

//#include "Error.h"

namespace Mila::Dnn::CuDNN
{
    // Forward Declaration
    class OpaqueCudnnHandle;

    /// <summary>
    /// A shared_ptr wrapper on top of the OpaqueHandle
    /// </summary>
    using ManagedCudnnHandle = std::shared_ptr<OpaqueCudnnHandle>;

    /// <summary>
    /// A wrapper on top of the std::make_shared for the OpaqueHandle
    /// </summary>
    /// <returns></returns>
    static ManagedCudnnHandle MakeManagedCudnnHandle()
    {
        return std::make_shared<OpaqueCudnnHandle>();
    };

    class OpaqueCudnnHandle
    {
    public:

        /// <summary>
        /// Delete the copy constructor to prevent copies.
        /// </summary>
        OpaqueCudnnHandle( const OpaqueCudnnHandle& ) = delete;
        OpaqueCudnnHandle& operator=( const OpaqueCudnnHandle& ) = delete;

        OpaqueCudnnHandle( OpaqueCudnnHandle&& ) = default;

        /// <summary>
        /// Constructor
        /// </summary>
        OpaqueCudnnHandle()
        {
            std::cout << "OpaqueCudnnHandle() Creating cudnn handle\n";

            status_ = cudnnCreate( &handle_ );

            if ( status_ != CUDNN_STATUS_SUCCESS )
            {
                throw std::runtime_error( "Failed to create CuDNN handle." );
                    //cudnnException(
                    //std::string( "Failed to create CuDNN handle. Status: " ).c_str(),
                    //status_ );
            }
        }

        /// <summary>
        /// Gets the const reference to raw underlying CuDNN context handle.
        /// </summary>
        /// <returns></returns>
        cudnnHandle_t GetOpaqueHandle() const
        {
            return handle_;
        }

        /// <summary>
        /// Gets the status of the cudnnCreate call. 
        /// </summary>
        cudnnStatus_t GetStatus() const
        {
            return status_;
        }

        /// <summary>
        /// Returns true if the handle creation was completed successfully.
        /// </summary>
        bool IsStatusSuccess() const
        {
            return status_ == CUDNN_STATUS_SUCCESS;
        }

        /**
         * OpaqueHandle destructor.
         * Calls the cudnnDestroy to release resources.
         */
        ~OpaqueCudnnHandle() noexcept(false)
        {
            std::cout << "~OpaqueCudnnHandle() calling cudnnDestroy().. ";

            if ( handle_ != nullptr )
            {
                auto status = cudnnDestroy( handle_ );

                if ( status != CUDNN_STATUS_SUCCESS )
                {
                    throw std::runtime_error( "Failed to destroy cudnn handle." );// , status );
                }
            }

            std::cout << "Done. Completed successfully." << std::endl;
        }

    private:

        /// <summary>
        /// CuDNN handle returned from cudnnCreate() call.
        /// </summary>
        cudnnHandle_t handle_ = nullptr;

        /// <summary>
        /// Status of handle creation.
        /// </summary>
        cudnnStatus_t status_ = CUDNN_STATUS_SUCCESS;
    };
}
#endif