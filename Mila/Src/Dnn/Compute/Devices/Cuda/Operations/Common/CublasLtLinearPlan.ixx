/**
 * @file CublasLtLinearPlan.ixx
 * @brief cuBLASLt matmul plan builder for CudaLinearOp.
 *
 * Provides a mixed-precision-aware plan struct and build/execute functions
 * for the Linear op only. For now, GQA and MHA continue to use the existing
 * build_plan / build_strided_plan / execute_plan path untouched.
 *
 * Template parameters follow Mila conventions:
 *   TComputePrecision   - activation and output element type (e.g. BF16)
 *   TParameterPrecision - weight element type (e.g. FP8_E4M3); defaults to TComputePrecision
 *   TAccumPrecision     - accumulator and scale type; always float
 *
 * When TParameterPrecision == TComputePrecision (non-quantized):
 *   - layoutA, layoutB, layoutC all use the same cudaDataType_t
 *   - has_per_channel_scale is false
 *   - execute_linear_plan ignores the per_channel_scale pointer
 *
 * When TParameterPrecision != TComputePrecision (quantized, e.g. BF16 x FP8):
 *   - layoutA and layoutC use cuda_data_type_v<TComputePrecision>
 *   - layoutB uses cuda_data_type_v<TParameterPrecision>
 *   - has_per_channel_scale is true
 *   - execute_linear_plan sets CUBLASLT_MATMUL_DESC_B_SCALE_POINTER at execution time
 *
 * Caller contract:
 *   compute_type and scale_type are supplied by CudaLinearOp::getComputeTypes()
 *   and must be consistent with TComputePrecision and the active ComputePrecision::Policy.
 *   The cudaDataType_t values for A, B, C are derived at compile time from the
 *   template parameters via cuda_data_type_v and are not caller-supplied.
 */

module;
#include <stdexcept>
#include <string>
#include <cublasLt.h>

export module Compute.Cuda.CublasLtLinearPlan;

import Dnn.TensorDataType;
import Compute.CudaTensorDataType;
import Cuda.Error;
import Utils.Logger;

namespace Mila::Dnn::Compute::Cuda
{
    /**
     * @brief RAII wrapper owning cuBLASLt descriptors for a Linear matmul.
     *
     * Owns:
     *   matmul_desc              - operation descriptor (transpose flags, epilogue, bias/scale pointers)
     *   layoutA, layoutB, layoutC - matrix memory layouts (B may differ from A/C when quantized)
     *   preference               - algorithm preference used during heuristic search
     *   algorithm                - selected heuristic algorithm
     *   has_algorithm            - true when heuristic returned a valid algorithm
     *   has_bias_epilogue        - true when CUBLASLT_EPILOGUE_BIAS is active
     *   has_per_channel_scale    - true when TParameterPrecision != TComputePrecision (FP8 path)
     *
     * Non-copyable; move-only.
     */
    export template<TensorDataType TComputePrecision, TensorDataType TParameterPrecision = TComputePrecision>
    struct CublasLtLinearPlan
    {
        using ActivationType = typename TensorDataTypeMap<TComputePrecision>::device_type;
        using ParameterType = typename TensorDataTypeMap<TParameterPrecision>::device_type;
        using TAccumPrecision = float;

        static constexpr bool kIsQuantized = (TParameterPrecision != TComputePrecision);

        cublasLtMatmulDesc_t       matmul_desc{ nullptr };
        cublasLtMatrixLayout_t     layoutA{ nullptr };
        cublasLtMatrixLayout_t     layoutB{ nullptr };
        cublasLtMatrixLayout_t     layoutC{ nullptr };
        cublasLtMatmulPreference_t preference{ nullptr };
        cublasLtMatmulAlgo_t       algorithm{};
        bool has_algorithm{ false };
        bool has_bias_epilogue{ false };
        bool has_per_channel_scale{ kIsQuantized };

        CublasLtLinearPlan() = default;

        ~CublasLtLinearPlan()
        {
            if ( matmul_desc ) cublasLtMatmulDescDestroy( matmul_desc );
            if ( layoutA )     cublasLtMatrixLayoutDestroy( layoutA );
            if ( layoutB )     cublasLtMatrixLayoutDestroy( layoutB );
            if ( layoutC )     cublasLtMatrixLayoutDestroy( layoutC );
            if ( preference )  cublasLtMatmulPreferenceDestroy( preference );
        }

        CublasLtLinearPlan( const CublasLtLinearPlan& ) = delete;
        CublasLtLinearPlan& operator=( const CublasLtLinearPlan& ) = delete;

        CublasLtLinearPlan( CublasLtLinearPlan&& other ) noexcept
            : matmul_desc( other.matmul_desc )
            , layoutA( other.layoutA )
            , layoutB( other.layoutB )
            , layoutC( other.layoutC )
            , preference( other.preference )
            , algorithm( other.algorithm )
            , has_algorithm( other.has_algorithm )
            , has_bias_epilogue( other.has_bias_epilogue )
            , has_per_channel_scale( other.has_per_channel_scale )
        {
            other.matmul_desc = nullptr;
            other.layoutA = nullptr;
            other.layoutB = nullptr;
            other.layoutC = nullptr;
            other.preference = nullptr;
            other.has_algorithm = false;
            other.has_bias_epilogue = false;
            other.has_per_channel_scale = false;
        }

        CublasLtLinearPlan& operator=( CublasLtLinearPlan&& other ) noexcept
        {
            if ( this != &other )
            {
                if ( matmul_desc ) cublasLtMatmulDescDestroy( matmul_desc );
                if ( layoutA )     cublasLtMatrixLayoutDestroy( layoutA );
                if ( layoutB )     cublasLtMatrixLayoutDestroy( layoutB );
                if ( layoutC )     cublasLtMatrixLayoutDestroy( layoutC );
                if ( preference )  cublasLtMatmulPreferenceDestroy( preference );

                matmul_desc = other.matmul_desc;
                layoutA = other.layoutA;
                layoutB = other.layoutB;
                layoutC = other.layoutC;
                preference = other.preference;
                algorithm = other.algorithm;
                has_algorithm = other.has_algorithm;
                has_bias_epilogue = other.has_bias_epilogue;
                has_per_channel_scale = other.has_per_channel_scale;

                other.matmul_desc = nullptr;
                other.layoutA = nullptr;
                other.layoutB = nullptr;
                other.layoutC = nullptr;
                other.preference = nullptr;
                other.has_algorithm = false;
                other.has_bias_epilogue = false;
                other.has_per_channel_scale = false;
            }

            return *this;
        }

        bool isValid() const
        {
            return matmul_desc != nullptr;
        }
    };

    /**
     * @brief Build a cuBLASLt plan for a Linear matmul.
     *
     * Computes:
     *   C[outer_size, out_features] = A[outer_size, in_features] @ B^T[in_features, out_features]
     *
     * Row-major layout, opA=N, opB=T, single matmul instance (no strided batch).
     *
     * cudaDataType_t values for A, B, C are derived at compile time from the
     * template parameters via cuda_data_type_v. compute_type and scale_type are
     * supplied by the caller via CudaLinearOp::getComputeTypes() so that the
     * active ComputePrecision::Policy is respected.
     *
     * @param outer_size    Row count for A and C (B * T for transformers).
     * @param in_features   Inner dimension (columns of A, columns of B).
     * @param out_features  Output dimension (rows of B, columns of C).
     * @param has_bias      When true, activates CUBLASLT_EPILOGUE_BIAS.
     * @param compute_type  Supplied by CudaLinearOp::getComputeTypes().
     * @param scale_type    Supplied by CudaLinearOp::getComputeTypes(); always CUDA_R_32F.
     */
    export template<TensorDataType TComputePrecision, TensorDataType TParameterPrecision = TComputePrecision>
        CublasLtLinearPlan<TComputePrecision, TParameterPrecision> build_linear_plan(
            cublasLtHandle_t handle,
            int outer_size,
            int in_features,
            int out_features,
            bool has_bias,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
    {
        constexpr cudaDataType_t data_type_A = cuda_data_type_v<TComputePrecision>;
        constexpr cudaDataType_t data_type_B = cuda_data_type_v<TParameterPrecision>;
        constexpr cudaDataType_t data_type_C = cuda_data_type_v<TComputePrecision>;

        CublasLtLinearPlan<TComputePrecision, TParameterPrecision> plan;
        plan.has_bias_epilogue = has_bias;

        // --- descriptor ---
        cublasStatus_t status = cublasLtMatmulDescCreate( &plan.matmul_desc, compute_type, scale_type );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "cublasLtMatmulDescCreate failed: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        const cublasOperation_t opA = CUBLAS_OP_N;
        const cublasOperation_t opB = CUBLAS_OP_T;

        status = cublasLtMatmulDescSetAttribute(
            plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof( opA ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set TRANSA failed: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatmulDescSetAttribute(
            plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof( opB ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set TRANSB failed: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        if ( has_bias )
        {
            const int epilogue = CUBLASLT_EPILOGUE_BIAS;
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof( epilogue ) ) );
        }

        // --- layouts ---
        // A: [outer_size, in_features] - activations, TComputePrecision
        status = cublasLtMatrixLayoutCreate( &plan.layoutA, data_type_A, outer_size, in_features, in_features );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "layoutA create failed: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        // B: [out_features, in_features] - weights, TParameterPrecision (may differ from A)
        status = cublasLtMatrixLayoutCreate( &plan.layoutB, data_type_B, out_features, in_features, in_features );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "layoutB create failed: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        // C: [outer_size, out_features] - output, TComputePrecision
        status = cublasLtMatrixLayoutCreate( &plan.layoutC, data_type_C, outer_size, out_features, out_features );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "layoutC create failed: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

        status = cublasLtMatrixLayoutSetAttribute( plan.layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof( order ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set order for A failed: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatrixLayoutSetAttribute( plan.layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof( order ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set order for B failed: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        status = cublasLtMatrixLayoutSetAttribute( plan.layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof( order ) );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Utils::Logger::error( "Set order for C failed: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        // --- heuristic ---
        cublasLtCheckStatus( cublasLtMatmulPreferenceCreate( &plan.preference ) );

        constexpr size_t kWorkspaceHint = 4ull * 1024 * 1024;
        cublasLtCheckStatus( cublasLtMatmulPreferenceSetAttribute(
            plan.preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kWorkspaceHint, sizeof( kWorkspaceHint ) ) );

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
            Utils::Logger::warning( "cuBLASLt heuristic found no algorithms, will use default at execution" );
            plan.algorithm = {};
            plan.has_algorithm = false;
        }
        else
        {
            Utils::Logger::error( "cuBLASLt heuristic failed with error status" );
            cublasLtCheckStatus( status );
        }

        return plan;
    }

    /**
     * @brief Execute a previously-built CublasLtLinearPlan.
     *
     * Computes: C = alpha * op(A) * op(B) + beta * C
     * with optional bias epilogue and optional per-channel weight scale.
     *
     * @param A                 Device pointer to activations (TComputePrecision).
     * @param B                 Device pointer to weights (TParameterPrecision).
     * @param C                 Device pointer to output (TComputePrecision).
     * @param bias              Device pointer to bias vector (TComputePrecision).
     *                          Applied only when plan.has_bias_epilogue is true.
     * @param per_channel_scale Device pointer to per-channel weight scales (float).
     *                          Applied only when plan.has_per_channel_scale is true.
     *                          Set via CUBLASLT_MATMUL_DESC_B_SCALE_POINTER.
     * @param workspace         Optional device scratch buffer.
     * @param workspace_size    Size of workspace in bytes.
     */
    export template<TensorDataType TComputePrecision, TensorDataType TParameterPrecision = TComputePrecision>
        void execute_linear_plan(
            cublasLtHandle_t handle,
            const CublasLtLinearPlan<TComputePrecision, TParameterPrecision>&plan,
            const float* alpha,
            const typename CublasLtLinearPlan<TComputePrecision, TParameterPrecision>::ActivationType * A,
            const typename CublasLtLinearPlan<TComputePrecision, TParameterPrecision>::ParameterType * B,
            const float* beta,
            typename CublasLtLinearPlan<TComputePrecision, TParameterPrecision>::ActivationType * C,
            const typename CublasLtLinearPlan<TComputePrecision, TParameterPrecision>::ActivationType * bias,
            const float* per_channel_scale,
            cudaStream_t stream,
            void* workspace = nullptr,
            size_t workspace_size = 0 )
    {
        if ( !plan.isValid() )
        {
            throw std::invalid_argument( "execute_linear_plan - plan is not valid" );
        }

        if ( plan.has_bias_epilogue && bias != nullptr )
        {
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof( bias ) ) );
        }

        if ( plan.has_per_channel_scale && per_channel_scale != nullptr )
        {
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &per_channel_scale, sizeof( per_channel_scale ) ) );
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
            workspace, workspace_size,
            stream );

        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            throw CublasLtError( status );
        }
    }
}