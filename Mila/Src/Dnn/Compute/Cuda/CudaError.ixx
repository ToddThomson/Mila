/**
 * @file CudaError.ixx
 * @brief CUDA error handling utilities and exception class.
 *
 * This module provides a comprehensive error handling system for CUDA operations,
 * including a custom exception class and utility functions for checking CUDA
 * operation status. It simplifies error detection, provides detailed error
 * information, and maintains source location context for debugging.
 */

module;
#include <stdexcept>
#include <string>
#include <ostream>
#include <sstream>
#include <cuda_runtime.h>
#include <source_location>

export module Cuda.Error;

namespace Mila::Dnn::Compute
{

    /**
     * @brief Exception class for CUDA runtime errors.
     *
     * This class wraps CUDA runtime API errors with additional context information
     * such as the file, line, and function where the error occurred. It inherits
     * from std::runtime_error and provides methods to access error details.
     */
    export class CudaError : public std::runtime_error
    {
    public:
        /**
         * @brief Constructs a new CudaError exception.
         *
         * @param status The CUDA error status code.
         * @param location Source location information (automatically populated by default).
         */
        CudaError( cudaError_t status, const std::source_location& location = std::source_location::current() )
            : std::runtime_error( getMessage( status, location ) ),
            cuda_error_( status ),
            file_( location.file_name() ),
            line_( location.line() ),
            function_( location.function_name() )
        {}

        /**
         * @brief Gets the CUDA error code.
         *
         * @return cudaError_t The original CUDA error status.
         */
        cudaError_t getError() const noexcept
        {
            return cuda_error_;
        }

        /**
         * @brief Gets the filename where the error occurred.
         *
         * @return const char* Pointer to the filename string.
         */
        const char* getFile() const noexcept
        {
            return file_;
        }

        /**
         * @brief Gets the line number where the error occurred.
         *
         * @return uint32_t The line number in the source file.
         */
        uint32_t getLine() const noexcept
        {
            return line_;
        }

        /**
         * @brief Gets the function name where the error occurred.
         *
         * @return const char* Pointer to the function name string.
         */
        const char* getFunction() const noexcept
        {
            return function_;
        }

    private:
        cudaError_t cuda_error_ = cudaSuccess; ///< The CUDA error code
        const char* file_;                     ///< Source file where the error occurred
        uint32_t line_;                        ///< Line number where the error occurred
        const char* function_;                 ///< Function name where the error occurred

        /**
         * @brief Generates a detailed error message from a CUDA error.
         *
         * @param status The CUDA error status code.
         * @param location Source location information.
         * @return std::string A formatted error message with context information.
         */
        static std::string getMessage(
            cudaError_t status,
            const std::source_location& location
        )
        {
            const char* name = cudaGetErrorName( status );
            const char* desc = cudaGetErrorString( status );

            if ( !name )
                name = "<unknown error>";

            std::ostringstream ss;
            ss << "CUDA runtime API error "
                << name << " (" << static_cast<unsigned>(status) << ")"
                << " at " << location.file_name() << ":" << location.line()
                << " in " << location.function_name();

            if ( desc && *desc )
                ss << ":\n" << desc;

            return ss.str();
        }
    };


    /**
     * @brief Checks the status of a CUDA operation and throws if an error occurred.
     *
     * @param status The CUDA error status code to check.
     * @param location Source location information (automatically populated by default).
     * @throws CudaError if the status is not cudaSuccess.
     */
    export inline void cudaCheckStatus( cudaError_t status, const std::source_location& location = std::source_location::current() ) {
        if ( status != cudaSuccess ) {
            throw CudaError( status, location );
        }
    }

    /**
     * @brief Checks the last CUDA error and throws if an error occurred.
     *
     * @param location Source location information (automatically populated by default).
     * @throws CudaError if the last error is not cudaSuccess.
     */
    export inline void cudaCheckLastError(
        const std::source_location& location = std::source_location::current()
    ) {
        cudaError_t error = cudaGetLastError();
        if ( error != cudaSuccess ) {
            throw CudaError( error, location );
        }
    }
}