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

import Dnn.TensorTypes;
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
        int batch_count,
        bool has_bias = false,
        cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F,
        cudaDataType_t cuda_data_type = CUDA_R_32F,
        cudaDataType_t scale_type = CUDA_R_32F,
        cublasLtOrder_t order = CUBLASLT_ORDER_ROW )
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

        status = cublasLtMatrixLayoutSetAttribute(
            plan.layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof( order ) );
        
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set order for A failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatrixLayoutCreate(
            &plan.layoutB, cuda_data_type, B_rows, B_cols, ldB );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "cublasLtMatrixLayoutCreate for B failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatrixLayoutSetAttribute(
            plan.layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof( order ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set order for B failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatrixLayoutCreate(
            &plan.layoutC, cuda_data_type, C_rows, C_cols, ldC );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "cublasLtMatrixLayoutCreate for C failed with status: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatrixLayoutSetAttribute(
            plan.layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof( order ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set order for C failed with status: " + std::to_string( status ) );
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

    /**
     * @brief Helper to dump a single 2D row-major matrix (host memory)
     *
     * Indexing: element (row=r, col=c) -> host_data[r * cols + c]
     */
    template<typename T>
    void dump_2d_rowmajor_host(
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
                T value = host_data[ r * cols + c ];
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
     * @brief Debug utility to dump row-major tensor from device memory
     *
     * This utility copies data from device to host, then properly interprets
     * the row-major layout for display.
     *
     * @tparam T Data type (float, __half, etc.)
     * @param device_data Device pointer to row-major data
     * @param shape Shape vector (shape_t)
     * @param name Tensor name for display
     * @param max_display_size Maximum elements to display per dimension
     * @param stream CUDA stream for async copy (nullptr for default stream)
     * @return String representation of the tensor
     */
    template<typename T = float>
    std::string dump_tensor(
        const T* device_data,
        const shape_t& shape,
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
        for ( auto dim : shape ) {
            if ( dim <= 0 ) {
                return name + ": <invalid shape>\n";
            }
            total_size *= static_cast<size_t>( dim );
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
        oss << "\n=== Row-Major Tensor: " << name << " ===\n";
        oss << "Shape: [";
        for ( size_t i = 0; i < shape.size(); ++i ) {
            oss << shape[ i ];
            if ( i < shape.size() - 1 ) oss << ", ";
        }
        oss << "] (row-major for last 2 dims)\n";
        oss << "Total elements: " << total_size << "\n\n";

        const T* read_ptr = host_data.data();

        // Determine what to display based on dimensionality
        if ( shape.size() == 2 ) {
            // Simple 2D matrix [rows, cols] in row-major
            int rows = static_cast<int>( shape[0] );
            int cols = static_cast<int>( shape[1] );
            dump_2d_rowmajor_host( oss, read_ptr, rows, cols, max_display_size, 0 );
        }
        else if ( shape.size() == 4 ) {
            // Common case: [B, NH, rows, cols]
            int B = static_cast<int>( shape[0] );
            int NH = static_cast<int>( shape[1] );
            int rows = static_cast<int>( shape[2] );
            int cols = static_cast<int>( shape[3] );

            int b_display = std::min( B, max_display_size / 2 );
            int nh_display = std::min( NH, max_display_size / 2 );

            for ( int b = 0; b < b_display; ++b ) {
                oss << "Batch " << b << ":\n";
                for ( int nh = 0; nh < nh_display; ++nh ) {
                    oss << "  Head " << nh << ":\n";

                    // Pointer to this (b, nh) matrix (row-major block)
                    size_t matrix_offset = ( static_cast<size_t>(b) * NH + static_cast<size_t>(nh) ) * rows * cols;
                    dump_2d_rowmajor_host( oss, read_ptr + matrix_offset, rows, cols, max_display_size, 4 );
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
            // [B, rows, cols] case (row-major)
            int B = static_cast<int>( shape[0] );
            int rows = static_cast<int>( shape[1] );
            int cols = static_cast<int>( shape[2] );

            int b_display = std::min( B, max_display_size / 2 );

            for ( int b = 0; b < b_display; ++b ) {
                oss << "Batch " << b << ":\n";
                size_t matrix_offset = static_cast<size_t>(b) * rows * cols;
                dump_2d_rowmajor_host( oss, read_ptr + matrix_offset, rows, cols, max_display_size, 2 );
                oss << "\n";
            }
            if ( b_display < B ) {
                oss << "... (" << (B - b_display) << " more batches)\n";
            }
        }
        else {
            oss << "Unsupported shape dimensionality (" << shape.size()
                << ") for row-major display\n";
            oss << "Supported: 2D [rows, cols], 3D [B, rows, cols], 4D [B, NH, rows, cols]\n";
        }

        return oss.str();
    }
}