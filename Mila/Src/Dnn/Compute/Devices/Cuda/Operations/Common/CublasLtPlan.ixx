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

export module Compute.CublasLtPlan;

import Dnn.TensorTypes;
import CublasLt.Error;
import Utils.Logger;

export namespace Mila::Dnn::Compute::Cuda
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

    template <typename TNative>
    CublasLtMatMulPlan<TNative> build_plan(
        cublasLtHandle_t handle,
        int batch_size,
        int in_features,
        int out_features,
        bool has_bias,
        cudaDataType_t data_type,
        cublasComputeType_t compute_type,
        cudaDataType_t scale_type )
    {
        const long long strideA = 0LL;
        const long long strideB = 0LL;
        const long long strideC = 0LL;

        return build_strided_plan<TNative>(
            handle,
            /*A_rows=*/ batch_size,   /*A_cols=*/ in_features,  /*ldA=*/ in_features,  /*strideA=*/ strideA,
            /*B_rows=*/ out_features, /*B_cols=*/ in_features,  /*ldB=*/ in_features,  /*strideB=*/ strideB,
            /*C_rows=*/ batch_size,   /*C_cols=*/ out_features, /*ldC=*/ out_features, /*strideC=*/ strideC,
            /*opA=*/ CUBLAS_OP_N, /*opB=*/ CUBLAS_OP_T,
            /*batch_count=*/ 1,
            has_bias,
            compute_type,
            data_type,
            scale_type );
    }

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

    
}