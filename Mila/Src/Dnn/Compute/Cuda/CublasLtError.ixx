module;
#include <cublasLt.h>
#include <stdexcept>
#include <string>
#include <ostream>
#include <sstream>
#include <source_location>

export module CublasLt.Error;

namespace Mila::Dnn::Compute
{
    export class CublasLtError : public std::runtime_error
    {
    public:
        CublasLtError( cublasStatus_t status, const std::source_location& location = std::source_location::current() )
            : std::runtime_error( getMessage( status, location ) ),
            cublas_error_( status ),
            file_( location.file_name() ),
            line_( location.line() ),
            function_( location.function_name() )
        {}

        cublasStatus_t getError() const noexcept
        {
            return cublas_error_;
        }

        const char* getFile() const noexcept
        {
            return file_;
        }

        uint32_t getLine() const noexcept
        {
            return line_;
        }

        const char* getFunction() const noexcept
        {
            return function_;
        }

    private:
        cublasStatus_t cublas_error_; ///< The cuBLAS error code
        const char* file_;            ///< Source file where the error occurred
        uint32_t line_;               ///< Line number where the error occurred
        const char* function_;        ///< Function name where the error occurred

        static std::string getMessage(
            cublasStatus_t status,
            const std::source_location& location
        )
        {
            // Convert cublasStatus_t to string representation
            const char* name = nullptr;
            switch ( status ) {
                case CUBLAS_STATUS_SUCCESS:          name = "CUBLAS_STATUS_SUCCESS"; break;
                case CUBLAS_STATUS_NOT_INITIALIZED:  name = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
                case CUBLAS_STATUS_ALLOC_FAILED:     name = "CUBLAS_STATUS_ALLOC_FAILED"; break;
                case CUBLAS_STATUS_INVALID_VALUE:    name = "CUBLAS_STATUS_INVALID_VALUE"; break;
                case CUBLAS_STATUS_ARCH_MISMATCH:    name = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
                case CUBLAS_STATUS_MAPPING_ERROR:    name = "CUBLAS_STATUS_MAPPING_ERROR"; break;
                case CUBLAS_STATUS_EXECUTION_FAILED: name = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
                case CUBLAS_STATUS_INTERNAL_ERROR:   name = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
                case CUBLAS_STATUS_NOT_SUPPORTED:    name = "CUBLAS_STATUS_NOT_SUPPORTED"; break;
                case CUBLAS_STATUS_LICENSE_ERROR:    name = "CUBLAS_STATUS_LICENSE_ERROR"; break;
                default: name = "<unknown cublasLt error>"; break;
            }

            // Get error description
            const char* desc = nullptr;
            switch ( status ) {
                case CUBLAS_STATUS_SUCCESS:
                    desc = "The operation completed successfully"; break;
                case CUBLAS_STATUS_NOT_INITIALIZED:
                    desc = "The cuBLAS library was not initialized"; break;
                case CUBLAS_STATUS_ALLOC_FAILED:
                    desc = "Resource allocation failed"; break;
                case CUBLAS_STATUS_INVALID_VALUE:
                    desc = "An invalid value was used as an argument"; break;
                case CUBLAS_STATUS_ARCH_MISMATCH:
                    desc = "The operation requires features not available on the device"; break;
                case CUBLAS_STATUS_MAPPING_ERROR:
                    desc = "An access to GPU memory space failed"; break;
                case CUBLAS_STATUS_EXECUTION_FAILED:
                    desc = "The GPU program failed to execute"; break;
                case CUBLAS_STATUS_INTERNAL_ERROR:
                    desc = "An internal cuBLAS operation failed"; break;
                case CUBLAS_STATUS_NOT_SUPPORTED:
                    desc = "The functionality requested is not supported"; break;
                case CUBLAS_STATUS_LICENSE_ERROR:
                    desc = "The functionality requested requires licensing"; break;
                default:
                    desc = "Unknown error"; break;
            }

            std::ostringstream ss;
            ss << "cuBLASLt API error "
                << name << " (" << static_cast<unsigned>(status) << ")"
                << " at " << location.file_name() << ":" << location.line()
                << " in " << location.function_name();

            if ( desc )
                ss << ":\n" << desc;

            return ss.str();
        }
    };

    /**
     * @brief Checks the status of a cuBLASLt operation and throws if an error occurred.
     *
     * @param status The cuBLASLt error status code to check.
     * @param location Source location information (automatically populated by default).
     * @throws CublasLtError if the status is not CUBLAS_STATUS_SUCCESS.
     */
    export inline void cublasLtCheckStatus( cublasStatus_t status, const std::source_location& location = std::source_location::current() ) {
        if ( status != CUBLAS_STATUS_SUCCESS ) {
            throw CublasLtError( status, location );
        }
    }
}