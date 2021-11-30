/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef MILA_DNN_CUDNN_OPAQUE_DESCRIPTOR_H_
#define MILA_DNN_CUDNN_OPAQUE_DESCRIPTOR_H_

#include <cudnn.h>

#include "OpaqueDescriptorType.h"

namespace Mila::Dnn::CuDNN
{
    /// <summary>
    /// A generic CuDNN descriptor pointer type.
    /// </summary>
    typedef void* opaqueDescriptor_t;

    /// <summary>
    /// Creates a descriptor instance of the specified type.
    /// </summary>
    /// <param name="descriptorType">The descriptor type to create.</param>
    /// <param name="descriptor">A pointer to the created descriptor</param>
    /// <returns>CuDNN status</returns>
    inline cudnnStatus_t CreateOpaqueDescriptor(
        opaqueDescriptorType_t descriptorType,
        opaqueDescriptor_t descriptor )
    {
        std::cout << "Creating opaque Descriptor\n";

        switch ( descriptorType )
        {
        case CUDNN_RNN_DESCRIPTOR:
            return cudnnCreateRNNDescriptor( static_cast<cudnnRNNDescriptor_t*>(descriptor) );

        case CUDNN_RNNDATA_DESCRIPTOR:
            return cudnnCreateRNNDataDescriptor( static_cast<cudnnRNNDataDescriptor_t*>(descriptor) );

        case CUDNN_TENSOR_DESCRIPTOR:
            return cudnnCreateTensorDescriptor( static_cast<cudnnTensorDescriptor_t*>(descriptor) );

        case CUDNN_DROPOUT_DESCRIPTOR:
            return cudnnCreateDropoutDescriptor( static_cast<cudnnDropoutDescriptor_t*>(descriptor) );

        default:
            throw std::invalid_argument( "Invalid descriptor type." );
        }
    }

    inline cudnnStatus_t DestroyOpaqueDescriptor(
        opaqueDescriptorType_t descriptorType,
        opaqueDescriptor_t descriptor )
    {
        std::cout << "Destroying opaque Descriptor..";

        switch ( descriptorType )
        {
        case CUDNN_RNN_DESCRIPTOR:
            return cudnnDestroyRNNDescriptor( static_cast<cudnnRNNDescriptor_t>(descriptor) );

        case CUDNN_RNNDATA_DESCRIPTOR:
            return cudnnDestroyRNNDataDescriptor( static_cast<cudnnRNNDataDescriptor_t>(descriptor) );

        case CUDNN_TENSOR_DESCRIPTOR:
            return cudnnDestroyTensorDescriptor( static_cast<cudnnTensorDescriptor_t>(descriptor) );

        case CUDNN_DROPOUT_DESCRIPTOR:
            return cudnnDestroyDropoutDescriptor( static_cast<cudnnDropoutDescriptor_t>(descriptor) );

        default:
            throw std::invalid_argument( "Invalid descriptor type." );
        }
    }
    
    /// <summary>
    /// Creates and stores an opaque structure pointer to a CuDNN descriptor type.
    /// </summary>
    class OpaqueDescriptor
    {
    public:

        OpaqueDescriptor( OpaqueDescriptor&& ) = default;
        OpaqueDescriptor& operator= ( OpaqueDescriptor&& ) = default;

        OpaqueDescriptor( const OpaqueDescriptor& ) = delete;
        OpaqueDescriptor& operator=( const OpaqueDescriptor& ) = delete;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="type">Descriptor type to create</param>
        OpaqueDescriptor( opaqueDescriptorType_t type )
        {
            std::cout << "OpaqueDescriptor() constructor\n";
            descriptorType_ = type;
            status_ = CreateOpaqueDescriptor( type, &descriptor_ );
        }

         /// <summary>
         /// Destroys the opaque descriptor.
         /// </summary>
        ~OpaqueDescriptor() noexcept(false)
        {
            std::cout << "~OpaqueDescriptor()\n";
            
            if ( descriptor_ )
            {
                auto status = DestroyOpaqueDescriptor( descriptorType_, descriptor_ );

                if ( status != CUDNN_STATUS_SUCCESS )
                {
                    throw std::runtime_error( "Failed to destroy opaque descriptor" );// status );
                }

                std::cout << "Descriptor destroyed successfully.\n";
            }
        }

        /// <summary>
        /// Gets the const reference to opaque descriptor. 
        /// </summary>
        /// <returns>The opaque descriptor</returns>
        opaqueDescriptor_t const& get_descriptor() const
        {
            return descriptor_;
        }

        /// <summary>
        /// Gets the opaque description creation status.
        /// </summary>
        /// <returns></returns>
        cudnnStatus_t get_status() const
        {
            return status_;
        }

        /// <summary>
        /// Flag returning opaque descriptor creation status.
        /// </summary>
        /// <returns>True if the opaque descriptor was created successfully. False otherwise.</returns>
        bool IsSuccessful() const
        {
            return status_ == CUDNN_STATUS_SUCCESS;
        }

    private:

        /// <summary>
        /// The pointer to the opaque descriptor object.
        /// </summary>
        opaqueDescriptor_t descriptor_ = nullptr;

        /// <summary>
        /// The opaque descriptor type
        /// </summary>
        opaqueDescriptorType_t descriptorType_;

        //!< status of creation of the Descriptor

        /// <summary>
        /// The status of opaque descriptor creation.
        /// </summary>
        cudnnStatus_t status_ = CUDNN_STATUS_SUCCESS;
    };
}
#endif