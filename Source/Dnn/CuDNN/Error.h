
#ifndef MILA_DNN_CUDNN_ERROR_H_
#define MILA_DNN_CUDNN_ERROR_H_

#include <exception>
#include <functional>
#include <string>

#include <cudnn.h>

#include "../Cuda/CudaError.h"

#include "Descriptor.h"

namespace Mila::Dnn::CuDNN 
{
#ifndef NV_CUDNN_DISABLE_EXCEPTION
    class cudnnException : public std::runtime_error
    {
    public:
        cudnnException( const char* message, cudnnStatus_t status_ ) 
            throw() : std::runtime_error( message )
        {
            error_status = status_;
        }
        
        virtual const char* what() const throw()
        {
            return std::runtime_error::what();
        }

        cudnnStatus_t GetStatus()
        {
            return error_status;
        }

        cudnnStatus_t error_status;
    };
#endif

    static inline void throw_if( std::function<bool()> expr, const char* message, cudnnStatus_t status )
    {
        if ( expr() )
        {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            throw cudnnException( message, status );
#endif
        }
    };

    static inline void throw_if( bool expr, const char* message, cudnnStatus_t status )
    {
        if ( expr )
        {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            throw cudnnException( message, status );
#endif
        }
    };

    static inline void SetErrorAndThrow(
        Descriptor *desc,
        cudnnStatus_t status,
        const char* message )
    {
        if ( desc != nullptr )
        {
            desc->set_status( status );
            desc->set_error( message );
        }

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        throw cudnnException(
            std::string( std::string( message ) + std::string( " cudnn_status: " ) + to_string( status ) ).c_str(), status );
#endif
    };

    static inline void cudnnStatusCheck( cudnnStatus_t status )
    {
        if ( status != CUDNN_STATUS_SUCCESS )
        {
            throw cudnnException(
                std::string( "CuDNN Error: " + std::string(" cudnn_status: ") + to_string(status)).c_str(),
                status );
        }
    }
}
#endif