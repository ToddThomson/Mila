#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <source_location>

// Convenience function for calculating grid/block dimensions for kernels
constexpr int ceil_div(int M, int N) {
    return (M + N - 1) / N;
}

class CudaException : public std::runtime_error {
public:
    CudaException( cudaError_t error, const std::string& file, int line )
        : std::runtime_error( buildMessage( error, file, line ) ) , error_( error ) {}

    cudaError_t getError() const { return error_; }

private:
    static std::string buildMessage( cudaError_t error, const std::string& file, int line ) {
        return "[CUDA ERROR] at file " + file + ":" + std::to_string( line ) +
            ":\n" + cudaGetErrorString( error );
    }

    cudaError_t error_;
};

inline void cudaCheck( cudaError_t error, const std::source_location& loc = std::source_location::current() ) {
    if ( error != cudaSuccess ) {
        throw CudaException( error, loc.file_name(), loc.line() );
    }
}