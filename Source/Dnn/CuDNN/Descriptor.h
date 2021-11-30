/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef MILA_DNN_CUDNN_DESCRIPTOR_BASE_H_
#define MILA_DNN_CUDNN_DESCRIPTOR_BASE_H_

#include <memory>
#include <sstream>

#include <cudnn.h>

#include "OpaqueDescriptor.h"
#include "CudnnContext.h"
#include "Utils.h"
#include "Error.h"

namespace Mila::Dnn::CuDNN
{
    /// <summary>
    /// A shared_ptr wrapper for an opaque descriptor pointer.
    /// </summary>
    using ManagedDescriptor = std::unique_ptr<OpaqueDescriptor>;

    //using ManagedCudnnContext = std::shared_ptr<CudnnContext>;

    /// <summary>
    /// Makes a managed shared pointer to an opaque descriptor type.
    /// </summary>
    /// <param name="type"></param>
    /// <returns></returns>
    static ManagedDescriptor MakeManagedDescriptor( opaqueDescriptorType_t type )
    {
        return std::make_unique<OpaqueDescriptor>( type );
    };

    /// <summary>
    /// Stores a managed pointer to a OpaqueDescriptorPointer class object.
    ///
    /// Contains the status and error message if set after any operation.
    /// If exception is disabled the user must query the status after
    /// build operation in order to check if the cudnn construct was built
    /// correctly.
    /// </summary>
    /// 
    
    class Descriptor
    {
    public:

        Descriptor( const ManagedCudnnHandle& cudnnHandle, opaqueDescriptorType_t descriptorType )
        {
            std::cout << ">>> Descriptor( handle )\n";
            cudnn_handle_ = cudnnHandle;

            auto status = CreateManagedDescriptor( descriptorType );

            if ( status != CUDNN_STATUS_SUCCESS )
            {
                throw std::runtime_error(
                    std::string( std::string( "Failed to create the managed descriptor." ) ).c_str() );
            }
        }

        /// <summary>
        /// Describes the descriptor parameters.
        /// </summary>
        /// <returns>a string describing the descriptor</returns>
        virtual std::string ToString() const = 0;

        /// <summary>
        /// Get a copy of the raw descriptor pointer. Ownership is retained and
        /// gets deleted when out of scope.
        /// </summary>
        /// <returns>the opaque descriptor pointer</returns>
        opaqueDescriptor_t GetOpaqueDescriptor() const
        {
            return managedDescriptor_->get_descriptor();
        }

        /// <summary>
        /// Gets the current status of the descriptor
        /// </summary>
        /// <returns></returns>
        cudnnStatus_t get_status() const
        {
            return status_;
        }

        /// <summary>
        /// Set status of the descriptor
        /// </summary>
        /// <param name="status"></param>
        void set_status( cudnnStatus_t const status ) const
        {
            status_ = status;
        }

        //! Set Diagonistic error message.
        void set_error( const char* message ) const
        {
            err_msg_ = message;
        }

        //! Diagonistic error message if any
        const char* get_error() const
        {
            return err_msg_.c_str();
        }
        
        /// <summary>
        /// Gets a copy of the managed descriptor
        /// </summary>
        /// <returns></returns>
        const ManagedDescriptor& get_desc()
        {
            return managedDescriptor_;
        }

        /// <summary>
        /// Creates a managed descriptor object of the specified type. 
        /// </summary>
        /// <param name="type">Descriptor type to create.</param>
        /// <returns>Create status</returns>
        cudnnStatus_t CreateManagedDescriptor( opaqueDescriptorType_t type )
        {
            managedDescriptor_ = MakeManagedDescriptor( type );

            return managedDescriptor_->get_status();
        }

        virtual cudnnStatus_t Finalize() = 0;

        bool IsFinalized()
        {
            return isFinalized_;
        }

        cudnnStatus_t SetFinalized()
        {
            isFinalized_ = true;

            return CUDNN_STATUS_SUCCESS;
        }

    protected:

        Descriptor() = default;

        /// <summary>
        /// Copy constructor is deleted. The Descriptor base class manages a non-copyable opaque descriptor.
        /// </summary>
        Descriptor( const Descriptor& ) = delete;

        /// <summary>
        /// Copy assignment operator is deleted. The Descriptor base class manages a non-copyable opaque descriptor.
        /// </summary>
        Descriptor& operator=( const Descriptor& ) = delete;

        /// <summary>
        ///  Move Constructor.
        /// </summary>
        Descriptor( Descriptor&& other ) 
            : managedDescriptor_( std::move(other.managedDescriptor_) ),
            cudnn_handle_( other.cudnn_handle_ ),
            isFinalized_( other.isFinalized_),
            status_( other.status_ ), err_msg_( other.err_msg_ )
        {
            std::cout << "Descriptor move constructor" << std::endl;
            other.cudnn_handle_ = nullptr;
            other.managedDescriptor_ = nullptr;
        }

        /// <summary>
        /// Descriptor move assignment operator.
        /// </summary>
        /// <param name="other">RHS Descriptor being moved.</param>
        /// <returns>This pointer</returns>
        Descriptor& operator=( Descriptor&& other )
        {
            // TJT: Do we need to deep copy here?
            std::cout << "Descriptor:: move assignment" << std::endl;
            // Transfer ownership
            cudnn_handle_ = std::move( other.cudnn_handle_ );
            managedDescriptor_ = std::move( other.managedDescriptor_ );

            return *this;
        }

        virtual ~Descriptor() = default;

        void CheckFinalizedThrow()
        {
            if ( isFinalized_ )
            {
                throw std::runtime_error( "Descriptor property cannot be set after Finalize is called." );
            }
        }

        ManagedDescriptor managedDescriptor_ = nullptr;
        ManagedCudnnHandle cudnn_handle_ = nullptr;

        mutable bool isFinalized_ = false;
        mutable cudnnStatus_t status_ = CUDNN_STATUS_SUCCESS;
        mutable std::string err_msg_ =""; // Empty string
    };
}
#endif