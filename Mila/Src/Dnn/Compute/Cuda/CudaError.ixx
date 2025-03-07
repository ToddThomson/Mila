module;
#include <stdexcept>
#include <string>
#include <ostream>
#include <sstream>
#include <cuda_runtime.h>

export module Cuda.Error;

namespace Mila::Dnn::Compute::Cuda {

    export class CudaError : public std::runtime_error
    {
    public:

        CudaError( cudaError_t status )
            : std::runtime_error( get_message( status ) ), cuda_error_( status )
        {
        }

        cudaError_t Error() const noexcept
        {
            return cuda_error_;
        }

    private:

        cudaError_t cuda_error_ = cudaSuccess;

        static std::string get_message( cudaError_t status )
        {
            const char* name = cudaGetErrorName( status );
            const char* desc = cudaGetErrorString( status );

            if ( !name )
                name = "<unknown error>";

            std::ostringstream ss;
            ss << "CUDA runtime API error "
                << name << " (" << static_cast<unsigned>(status) << ")";
            
            if ( desc && *desc )
                ss << ":\n" << desc;

            return ss.str();
        }
    };
}
