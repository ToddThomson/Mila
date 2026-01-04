/**
 * @file CublasLtHelpers.ixx
 * @brief Shared cuBLASLt helpers for building and executing matmul plans (RAII + builders).
 *
 * Provides templated utilities to build cuBLASLt matmul plans (including strided-batched)
 * and execute them. Designed to be reused by CUDA Linear and Attention operations.
 */

module;
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>

export module CublasLtHelpers;

import CublasLt.Error;
import Utils.Logger;

export namespace Mila::Dnn::Compute::Cuda::Common::CublasLtHelpers
{
    using namespace Mila::Dnn;

    /**
     * @brief Simple RAII wrapper that owns cuBLASLt descriptors and selected algorithm.
     *
     * Holds:
     *  - matmul_desc
     *  - matrix layouts for A, B and C
     *  - preference object
     *  - algorithm selection (if found)
     *
     * The destructor releases any created descriptors.
     */
    template <typename TNative>
    struct CublasLtMatMulPlan
    {
        mutable cublasLtMatmulDesc_t matmul_desc{ nullptr };
        cublasLtMatrixLayout_t layoutA{ nullptr };
        cublasLtMatrixLayout_t layoutB{ nullptr };
        cublasLtMatrixLayout_t layoutC{ nullptr };
        cublasLtMatmulPreference_t preference{ nullptr };
        cublasLtMatmulAlgo_t algorithm{};
        bool has_algorithm{ false };
        bool has_bias_epilogue{ false };

        CublasLtMatMulPlan() = default;

        ~CublasLtMatMulPlan()
        {
            if ( matmul_desc ) cublasLtMatmulDescDestroy( matmul_desc );
            if ( layoutA ) cublasLtMatrixLayoutDestroy( layoutA );
            if ( layoutB ) cublasLtMatrixLayoutDestroy( layoutB );
            if ( layoutC ) cublasLtMatrixLayoutDestroy( layoutC );
            if ( preference ) cublasLtMatmulPreferenceDestroy( preference );
        }

        // Non-copyable
        CublasLtMatMulPlan( const CublasLtMatMulPlan& ) = delete;
        CublasLtMatMulPlan& operator=( const CublasLtMatMulPlan& ) = delete;

        // Move semantics
        CublasLtMatMulPlan( CublasLtMatMulPlan&& other ) noexcept
            : matmul_desc( other.matmul_desc )
            , layoutA( other.layoutA )
            , layoutB( other.layoutB )
            , layoutC( other.layoutC )
            , preference( other.preference )
            , algorithm( other.algorithm )
            , has_algorithm( other.has_algorithm )
            , has_bias_epilogue( other.has_bias_epilogue )
        {
            other.matmul_desc = nullptr;
            other.layoutA = nullptr;
            other.layoutB = nullptr;
            other.layoutC = nullptr;
            other.preference = nullptr;
            other.has_algorithm = false;
            other.has_bias_epilogue = false;
        }

        CublasLtMatMulPlan& operator=( CublasLtMatMulPlan&& other ) noexcept
        {
            if ( this != &other )
            {
                if ( matmul_desc ) cublasLtMatmulDescDestroy( matmul_desc );
                if ( layoutA ) cublasLtMatrixLayoutDestroy( layoutA );
                if ( layoutB ) cublasLtMatrixLayoutDestroy( layoutB );
                if ( layoutC ) cublasLtMatrixLayoutDestroy( layoutC );
                if ( preference ) cublasLtMatmulPreferenceDestroy( preference );

                matmul_desc = other.matmul_desc;
                layoutA = other.layoutA;
                layoutB = other.layoutB;
                layoutC = other.layoutC;
                preference = other.preference;
                algorithm = other.algorithm;
                has_algorithm = other.has_algorithm;
                has_bias_epilogue = other.has_bias_epilogue;

                other.matmul_desc = nullptr;
                other.layoutA = nullptr;
                other.layoutB = nullptr;
                other.layoutC = nullptr;
                other.preference = nullptr;
                other.has_algorithm = false;
                other.has_bias_epilogue = false;
            }
            return *this;
        }

        bool isValid() const
        {
            return matmul_desc != nullptr;
        }
    };

    /**
     * @brief Build a cuBLASLt matmul plan for strided-batched matmuls.
     *
     * This builder accepts explicit layout dimensions (rows/cols) and element-stride
     * values (in elements) for A, B and C.
     *
     * Parameters:
     *  - handle: cuBLASLt handle (must be valid)
     *  - A_rows, A_cols, ldA, strideA_elems: layout describing A in memory (elements)
     *  - B_rows, B_cols, ldB, strideB_elems: layout describing B in memory (elements)
     *  - C_rows, C_cols, ldC, strideC_elems: layout describing C in memory (elements)
     *  - opA, opB: cublasOperation_t specifying transpose behavior requested by caller
     *  - compute_type, scale_type, cuda_data_type: cuBLASLt types
     *  - batch_count: number of batched matmuls to perform
     *  - has_bias: enable bias epilogue if true
     *
     * Returns a plan with descriptors created and an algorithm heuristically selected (if available).
     *
     * @note CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET expects stride in elements, not bytes.
     */
    template <typename TNative>
    CublasLtMatMulPlan<TNative> build_strided_plan(
        cublasLtHandle_t handle,
        int A_rows, int A_cols, int ldA, long long strideA_elems,
        int B_rows, int B_cols, int ldB, long long strideB_elems,
        int C_rows, int C_cols, int ldC, long long strideC_elems,
        cublasOperation_t opA, cublasOperation_t opB,
        cublasComputeType_t compute_type,
        cudaDataType_t scale_type,
        cudaDataType_t cuda_data_type,
        int batch_count,
        bool has_bias = false )
    {
        CublasLtMatMulPlan<TNative> plan;
        plan.has_bias_epilogue = has_bias;

        cublasStatus_t status = cublasLtMatmulDescCreate( &plan.matmul_desc, compute_type, scale_type );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "cublasLtMatmulDescCreate failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatmulDescSetAttribute(
            plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof( cublasOperation_t ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set TRANSA failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatmulDescSetAttribute(
            plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof( cublasOperation_t ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set TRANSB failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        if ( has_bias )
        {
            const int epilogue = CUBLASLT_EPILOGUE_BIAS;
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof( epilogue ) ) );
        }

        status = cublasLtMatrixLayoutCreate(
            &plan.layoutA, cuda_data_type, A_rows, A_cols, ldA );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "cublasLtMatrixLayoutCreate for A failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatrixLayoutCreate(
            &plan.layoutB, cuda_data_type, B_rows, B_cols, ldB );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "cublasLtMatrixLayoutCreate for B failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatrixLayoutCreate(
            &plan.layoutC, cuda_data_type, C_rows, C_cols, ldC );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "cublasLtMatrixLayoutCreate for C failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        // CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET expects stride in ELEMENTS, not bytes
        status = cublasLtMatrixLayoutSetAttribute(
            plan.layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA_elems, sizeof( strideA_elems ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set stride A failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatrixLayoutSetAttribute(
            plan.layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB_elems, sizeof( strideB_elems ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set stride B failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatrixLayoutSetAttribute(
            plan.layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC_elems, sizeof( strideC_elems ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set stride C failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatrixLayoutSetAttribute(
            plan.layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof( batch_count ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set batch count A failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatrixLayoutSetAttribute(
            plan.layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof( batch_count ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set batch count B failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatrixLayoutSetAttribute(
            plan.layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof( batch_count ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set batch count C failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        cublasLtCheckStatus( cublasLtMatmulPreferenceCreate( &plan.preference ) );

        cublasLtMatmulHeuristicResult_t heuristic_result{};
        int returned_algo_count = 0;

        status = cublasLtMatmulAlgoGetHeuristic(
            handle, plan.matmul_desc,
            plan.layoutA, plan.layoutB, plan.layoutC, plan.layoutC,
            plan.preference, 1, &heuristic_result, &returned_algo_count );

        if ( status == CUBLAS_STATUS_SUCCESS && returned_algo_count > 0 )
        {
            plan.algorithm = heuristic_result.algo;
            plan.has_algorithm = true;
        }
        else if ( status == CUBLAS_STATUS_SUCCESS && returned_algo_count == 0 )
        {
            Utils::Logger::warning( "cuBLASLt strided heuristic found no algorithms, will use default at execution" );
            plan.algorithm = {};
            plan.has_algorithm = false;
        }
        else
        {
            Utils::Logger::error( "cuBLASLt strided heuristic failed with error status" );
            cublasLtCheckStatus( status );
        }

        return plan;
    }

    /**
     * @brief Execute a previously-built cuBLASLt plan.
     *
     * The caller is responsible for ensuring the pointers and plan layouts match.
     * If the plan was built with a bias epilogue, the caller must pass a non-null bias pointer.
     */
    template <typename TNative>
    void execute_plan(
        cublasLtHandle_t handle,
        const CublasLtMatMulPlan<TNative>& plan,
        const void* alpha,
        const TNative* A,
        const TNative* B,
        const void* beta,
        TNative* C,
        const TNative* bias,
        cudaStream_t stream,
        void* workspace = nullptr,
        size_t workspaceSize = 0 )
    {
        if ( !plan.isValid() )
        {
            throw std::invalid_argument( "execute_plan - provided cuBLASLt plan is not valid" );
        }

        if ( plan.has_bias_epilogue && bias != nullptr )
        {
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof( bias ) ) );
        }

        const cublasLtMatmulAlgo_t* algo_ptr = plan.has_algorithm ? &plan.algorithm : nullptr;

        cublasStatus_t status = cublasLtMatmul(
            handle,
            plan.matmul_desc,
            alpha,
            A, plan.layoutA,
            B, plan.layoutB,
            beta,
            C, plan.layoutC,
            C, plan.layoutC,
            algo_ptr,
            workspace, workspaceSize,
            stream );

        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            throw CublasLtError( status );
        }
    }

    // Add to CublasLtHelpers.ixx

#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cuda_runtime.h>

/**
 * @brief Helper to dump a single 2D column-major matrix (host memory)
 */
    template<typename T>
    void dump_2d_colmajor_host(
        std::ostringstream& oss,
        const T* host_data,
        int rows,
        int cols,
        int max_display,
        int indent = 0 )
    {
        std::string indent_str( indent, ' ' );

        int rows_display = std::min( rows, max_display );
        int cols_display = std::min( cols, max_display );

        for ( int r = 0; r < rows_display; ++r ) {
            oss << indent_str << "[ ";
            for ( int c = 0; c < cols_display; ++c ) {
                // Column-major indexing: element (row=r, col=c) is at r + c*rows
                T value = host_data[ r + c * rows ];
                oss << std::setw( 10 ) << static_cast<float>( value );
                if ( c < cols_display - 1 ) oss << " ";
            }
            if ( cols_display < cols ) {
                oss << " ... (" << (cols - cols_display) << " more)";
            }
            oss << " ]\n";
        }

        if ( rows_display < rows ) {
            oss << indent_str << "... (" << (rows - rows_display) << " more rows)\n";
        }
    }

    /**
     * @brief Debug utility to dump column-major tensor from device memory
     *
     * This utility copies data from device to host, then properly interprets
     * the column-major layout for display.
     *
     * @tparam T Data type (float, __half, etc.)
     * @param device_data Device pointer to column-major data
     * @param shape Shape vector [B, NH, rows, cols] or similar
     * @param name Tensor name for display
     * @param max_display_size Maximum elements to display per dimension
     * @param stream CUDA stream for async copy (nullptr for default stream)
     * @return String representation of the tensor
     */
    template<typename T = float>
    std::string dump_colmajor_tensor(
        const T* device_data,
        const std::vector<int>& shape,
        const std::string& name = "tensor",
        int max_display_size = 8,
        cudaStream_t stream = nullptr )
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision( 6 );

        // Validate input
        if ( shape.empty() || device_data == nullptr ) {
            return name + ": <invalid tensor>\n";
        }

        // Calculate total size
        size_t total_size = 1;
        for ( int dim : shape ) {
            if ( dim <= 0 ) {
                return name + ": <invalid shape>\n";
            }
            total_size *= dim;
        }

        // Allocate host memory
        std::vector<T> host_data( total_size );

        // Copy from device to host
        cudaError_t err;
        if ( stream != nullptr ) {
            err = cudaMemcpyAsync( host_data.data(), device_data,
                total_size * sizeof( T ),
                cudaMemcpyDeviceToHost,
                stream );
            if ( err == cudaSuccess ) {
                err = cudaStreamSynchronize( stream );
            }
        }
        else {
            err = cudaMemcpy( host_data.data(), device_data,
                total_size * sizeof( T ),
                cudaMemcpyDeviceToHost );
        }

        if ( err != cudaSuccess ) {
            return name + ": <cudaMemcpy failed: " +
                std::string( cudaGetErrorString( err ) ) + ">\n";
        }

        // Header
        oss << "\n=== Column-Major Tensor: " << name << " ===\n";
        oss << "Shape: [";
        for ( size_t i = 0; i < shape.size(); ++i ) {
            oss << shape[ i ];
            if ( i < shape.size() - 1 ) oss << ", ";
        }
        oss << "] (column-major for last 2 dims)\n";
        oss << "Total elements: " << total_size << "\n\n";

        const T* read_ptr = host_data.data();

        // Determine what to display based on dimensionality
        if ( shape.size() == 2 ) {
            // Simple 2D matrix [rows, cols] in column-major
            dump_2d_colmajor_host( oss, read_ptr, shape[ 0 ], shape[ 1 ], max_display_size, 0 );
        }
        else if ( shape.size() == 4 ) {
            // Common case: [B, NH, rows, cols] - attention matrices
            int B = shape[ 0 ];
            int NH = shape[ 1 ];
            int rows = shape[ 2 ];
            int cols = shape[ 3 ];

            int b_display = std::min( B, max_display_size / 2 );  // Show fewer batches
            int nh_display = std::min( NH, max_display_size / 2 );

            for ( int b = 0; b < b_display; ++b ) {
                oss << "Batch " << b << ":\n";
                for ( int nh = 0; nh < nh_display; ++nh ) {
                    oss << "  Head " << nh << ":\n";

                    // Pointer to this (b, nh) matrix
                    size_t matrix_offset = (b * NH * rows * cols) + (nh * rows * cols);
                    dump_2d_colmajor_host( oss, read_ptr + matrix_offset, rows, cols, max_display_size, 4 );
                    oss << "\n";
                }
                if ( nh_display < NH ) {
                    oss << "  ... (" << (NH - nh_display) << " more heads)\n";
                }
            }
            if ( b_display < B ) {
                oss << "... (" << (B - b_display) << " more batches)\n";
            }
        }
        else if ( shape.size() == 3 ) {
            // [B, rows, cols] case
            int B = shape[ 0 ];
            int rows = shape[ 1 ];
            int cols = shape[ 2 ];

            int b_display = std::min( B, max_display_size / 2 );

            for ( int b = 0; b < b_display; ++b ) {
                oss << "Batch " << b << ":\n";
                size_t matrix_offset = b * rows * cols;
                dump_2d_colmajor_host( oss, read_ptr + matrix_offset, rows, cols, max_display_size, 2 );
                oss << "\n";
            }
            if ( b_display < B ) {
                oss << "... (" << (B - b_display) << " more batches)\n";
            }
        }
        else {
            oss << "Unsupported shape dimensionality (" << shape.size()
                << ") for column-major display\n";
            oss << "Supported: 2D [rows, cols], 3D [B, rows, cols], 4D [B, NH, rows, cols]\n";
        }

        return oss.str();
    }

    /**
     * @brief Verify column-major attention matrix properties (causal + row sums)
     *
     * Copies from device and checks:
     * 1. Causal masking: att[t, t2] = 0 for t2 > t
     * 2. Row normalization: sum(att[t, :]) = 1.0 for each row t
     *
     * @param device_att Device pointer to attention matrix [B, NH, T, T] in column-major
     * @param B Batch size
     * @param NH Number of heads
     * @param T Sequence length
     * @param tolerance Tolerance for floating point comparisons
     * @param stream CUDA stream (nullptr for default)
     * @return true if all checks pass
     */
    inline bool verify_colmajor_attention(
        const float* device_att,
        int B, int NH, int T,
        float tolerance = 1e-4,
        cudaStream_t stream = nullptr )
    {
        // Copy to host
        size_t total_size = static_cast<size_t>(B) * NH * T * T;
        std::vector<float> host_att( total_size );

        cudaError_t err;
        if ( stream != nullptr ) {
            err = cudaMemcpyAsync( host_att.data(), device_att,
                total_size * sizeof( float ),
                cudaMemcpyDeviceToHost,
                stream );
            if ( err == cudaSuccess ) {
                err = cudaStreamSynchronize( stream );
            }
        }
        else {
            err = cudaMemcpy( host_att.data(), device_att,
                total_size * sizeof( float ),
                cudaMemcpyDeviceToHost );
        }

        if ( err != cudaSuccess ) {
            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString( err ) << "\n";
            return false;
        }

        bool all_passed = true;
        int causal_violations = 0;
        int sum_violations = 0;

        for ( int b = 0; b < B; ++b ) {
            for ( int nh = 0; nh < NH; ++nh ) {
                size_t matrix_offset = (static_cast<size_t>( b ) * NH * T * T) +
                    (static_cast<size_t>( nh ) * T * T);

                for ( int t = 0; t < T; ++t ) {
                    float row_sum = 0.0f;

                    for ( int t2 = 0; t2 < T; ++t2 ) {
                        // Column-major: element (row=t, col=t2) at t + t2*T
                        float value = host_att[ matrix_offset + t + t2 * T ];

                        // Check causal masking
                        if ( t2 > t ) {
                            if ( std::abs( value ) > tolerance ) {
                                if ( causal_violations < 5 ) {  // Limit output
                                    std::cerr << "Causal mask violation at [b=" << b
                                        << ", nh=" << nh << ", t=" << t
                                        << ", t2=" << t2 << "]: value=" << value << "\n";
                                }
                                causal_violations++;
                                all_passed = false;
                            }
                        }
                        else {
                            row_sum += value;
                        }
                    }

                    // Check row sum (should be 1.0)
                    if ( std::abs( row_sum - 1.0f ) > tolerance ) {
                        if ( sum_violations < 5 ) {  // Limit output
                            std::cerr << "Row sum violation at [b=" << b
                                << ", nh=" << nh << ", t=" << t
                                << "]: sum=" << row_sum << " (expected 1.0)\n";
                        }
                        sum_violations++;
                        all_passed = false;
                    }
                }
            }
        }

        if ( !all_passed ) {
            std::cerr << "\n=== Attention Verification Failed ===\n";
            if ( causal_violations > 0 ) {
                std::cerr << "Total causal violations: " << causal_violations << "\n";
            }
            if ( sum_violations > 0 ) {
                std::cerr << "Total row sum violations: " << sum_violations << "\n";
            }
        }

        return all_passed;
    }

    /**
     * @brief Compare two column-major tensors element-wise (both on device)
     *
     * Useful for validating kernel outputs against reference implementations.
     *
     * @param device_a First tensor on device
     * @param device_b Second tensor on device
     * @param size Number of elements
     * @param name Name for error reporting
     * @param tolerance Absolute difference tolerance
     * @param stream CUDA stream (nullptr for default)
     * @return true if tensors match within tolerance
     */
    template<typename T = float>
    bool compare_colmajor_tensors(
        const T* device_a,
        const T* device_b,
        size_t size,
        const std::string& name = "tensors",
        float tolerance = 1e-5,
        cudaStream_t stream = nullptr )
    {
        // Copy both to host
        std::vector<T> host_a( size );
        std::vector<T> host_b( size );

        cudaError_t err_a, err_b;
        if ( stream != nullptr ) {
            err_a = cudaMemcpyAsync( host_a.data(), device_a,
                size * sizeof( T ),
                cudaMemcpyDeviceToHost,
                stream );
            err_b = cudaMemcpyAsync( host_b.data(), device_b,
                size * sizeof( T ),
                cudaMemcpyDeviceToHost,
                stream );
            if ( err_a == cudaSuccess && err_b == cudaSuccess ) {
                cudaStreamSynchronize( stream );
            }
        }
        else {
            err_a = cudaMemcpy( host_a.data(), device_a,
                size * sizeof( T ),
                cudaMemcpyDeviceToHost );
            err_b = cudaMemcpy( host_b.data(), device_b,
                size * sizeof( T ),
                cudaMemcpyDeviceToHost );
        }

        if ( err_a != cudaSuccess || err_b != cudaSuccess ) {
            std::cerr << "cudaMemcpy failed during comparison\n";
            return false;
        }

        bool all_match = true;
        int mismatch_count = 0;
        float max_diff = 0.0f;
        size_t max_diff_idx = 0;

        for ( size_t i = 0; i < size; ++i ) {
            float val_a = static_cast<float>( host_a[ i ] );
            float val_b = static_cast<float>( host_b[ i ] );
            float diff = std::abs( val_a - val_b );

            if ( diff > tolerance ) {
                all_match = false;
                mismatch_count++;

                if ( diff > max_diff ) {
                    max_diff = diff;
                    max_diff_idx = i;
                }

                // Report first few mismatches
                if ( mismatch_count <= 10 ) {
                    std::cerr << "Mismatch at index " << i
                        << ": " << val_a << " vs " << val_b
                        << " (diff=" << diff << ")\n";
                }
            }
        }

        if ( !all_match ) {
            std::cerr << "\n=== Comparison Failed for " << name << " ===\n";
            std::cerr << "Total mismatches: " << mismatch_count << "/" << size
                << " (" << (100.0 * mismatch_count / size) << "%)\n";
            std::cerr << "Max difference: " << max_diff
                << " at index " << max_diff_idx << "\n";
        }
        else {
            std::cout << "=== Comparison Passed for " << name << " ===\n";
            std::cout << "All " << size << " elements match within tolerance "
                << tolerance << "\n";
        }

        return all_match;
    }

    // Usage examples:
    /*
    // Example 1: Dump attention matrix after softmax (device memory)
    std::string att_str = dump_colmajor_tensor_device(
        att_,                           // Device pointer
        {batch_size_, num_heads_, seq_length_, seq_length_},
        "attention_weights",
        4,                              // Show 4x4 matrices max
        stream_                         // Your CUDA stream
    );
    std::cout << att_str;

    // Example 2: Verify attention properties (device memory)
    bool valid = verify_colmajor_attention_device(
        att_,
        batch_size_,
        num_heads_,
        seq_length_,
        1e-4f,
        stream_
    );
    if (!valid) {
        std::cerr << "Attention matrix verification failed!\n";
    }

    // Example 3: Compare two device tensors
    bool match = compare_colmajor_tensors_device(
        att_gpu,           // Your kernel output (device)
        att_reference,     // Reference implementation (device)
        B * NH * T * T,
        "attention_output",
        1e-5f,
        stream_
    );
    */
}