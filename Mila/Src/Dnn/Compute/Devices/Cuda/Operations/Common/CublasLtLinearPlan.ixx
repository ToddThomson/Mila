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
import CublasLt.Error;
import Logging.Logger;

namespace Mila::Dnn::Compute::Cuda
{
    /**
     * @brief RAII wrapper owning cuBLASLt descriptors for a Linear matmul.
     *
     * Owns:
     *   matmul_desc              - operation descriptor (transpose flags, epilogue, scale pointers)
     *   layoutA, layoutB, layoutC - matrix memory layouts
     *   preference               - algorithm preference used during heuristic search
     *   algorithm                - selected heuristic algorithm
     *   has_algorithm            - true when heuristic returned a valid algorithm
     *   has_bias_epilogue        - true when CUBLASLT_EPILOGUE_BIAS is active
     *   has_weight_scale         - true when TParameterPrecision != TComputePrecision (FP8 path)
     *
     * Layout convention (compile-time, driven by kIsQuantized):
     *
     *   Non-quantized (NT row-major):
     *     A = activations [outer_size x in_features],  opA = N
     *     B = weights     [out_features x in_features], opB = T
     *     C = output      [outer_size x out_features]
     *
     *   Quantized (TN column-major, Ada SM 8.9+):
     *     A = weights (FP8) [in_features x out_features], opA = T  -> op(A) = W[out_features, in_features]
     *     B = activations   [in_features x outer_size],   opB = N  -> op(B) = X^T[in_features, outer_size]
     *     C = output        [out_features x outer_size]             (col-major == row-major Y[outer_size, out_features])
     *     A_SCALE_POINTER = per-tensor weight scale
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
        bool has_weight_scale{ kIsQuantized };  ///< true when a weight scale pointer is needed (FP8 path)

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
            , has_weight_scale( other.has_weight_scale )
        {
            other.matmul_desc = nullptr;
            other.layoutA = nullptr;
            other.layoutB = nullptr;
            other.layoutC = nullptr;
            other.preference = nullptr;
            other.has_algorithm = false;
            other.has_bias_epilogue = false;
            other.has_weight_scale = false;
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
                has_weight_scale = other.has_weight_scale;

                other.matmul_desc = nullptr;
                other.layoutA = nullptr;
                other.layoutB = nullptr;
                other.layoutC = nullptr;
                other.preference = nullptr;
                other.has_algorithm = false;
                other.has_bias_epilogue = false;
                other.has_weight_scale = false;
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
     * Layout is selected at compile time based on kIsQuantized:
     *
     *   Non-quantized (TComputePrecision == TParameterPrecision):
     *     NT row-major -- A = activations, B = weights, opA=N, opB=T
     *     C[outer_size, out_features] = A[outer_size, in_features] x B^T[in_features, out_features]
     *
     *   Quantized (Ada SM 8.9+, TParameterPrecision = FP8_E4M3):
     *     TN column-major -- A = weights (FP8), B = activations (BF16), opA=T, opB=N
     *     Exploits the row-major / column-major duality:
     *       row-major W[N, K] == col-major W^T[K, N]   (same bytes, lda = K)
     *       row-major X[M, K] == col-major X^T[K, M]   (same bytes, ldb = K)
     *       op(A) = (W^T)^T = W[N, K],  op(B) = X^T[K, M]
     *       D = W x X^T = Y^T[N, M] col-major == row-major Y[M, N]  (ldc = N)
     *     A_SCALE_POINTER = per-tensor weight scale (weight_scales_[0]).
     *
     * @param outer_size    Token count (M = B * T for transformers).
     * @param in_features   Inner dimension K.
     * @param out_features  Output channels N.
     * @param has_bias      Activates CUBLASLT_EPILOGUE_BIAS (non-quantized path only).
     * @param compute_type  Supplied by CudaLinearOp::getComputeTypes().
     * @param scale_type    Always CUDA_R_32F.
     * @param weight_scale  Quantized path only: device pointer to the per-tensor weight scale
     *                      (weight_scales_[0]). Must be non-null when kIsQuantized because
     *                      CUBLASLT_MATMUL_DESC_A_SCALE_POINTER must be set in the descriptor
     *                      *before* cublasLtMatmulAlgoGetHeuristic for FP8 algorithms to be
     *                      returned. Ignored on the non-quantized path (default nullptr).
     * @param output_row_stride Row stride of C, in elements. Zero (the default) means
     *                      out_features, i.e. C is a standalone [outer_size, out_features]
     *                      matrix -- the only shape every caller but one needs. Passing the
     *                      stride of a WIDER matrix makes C a column slice of it, so a plan
     *                      covering out_features of that matrix's channels writes them in
     *                      place. That is what lets the codebook prefill path dequantize and
     *                      multiply one strip of output channels at a time instead of
     *                      expanding a whole weight matrix (Specifications/Qwen3.8.md
     *                      sections 5 and 8). Non-quantized path only; a strided C has no
     *                      meaning for the FP8 layout and is rejected.
     */
    export template<TensorDataType TComputePrecision, TensorDataType TParameterPrecision = TComputePrecision>
        CublasLtLinearPlan<TComputePrecision, TParameterPrecision> build_linear_plan(
            cublasLtHandle_t handle,
            int outer_size,
            int in_features,
            int out_features,
            bool has_bias,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type,
            const float* weight_scale = nullptr,
            int output_row_stride = 0 )
    {
        constexpr bool kIsQuantized = (TParameterPrecision != TComputePrecision);

        if ( output_row_stride != 0 && output_row_stride < out_features )
        {
            throw std::invalid_argument(
                "build_linear_plan - output_row_stride must be zero or at least out_features" );
        }

        if constexpr ( kIsQuantized )
        {
            if ( output_row_stride != 0 )
            {
                throw std::invalid_argument(
                    "build_linear_plan - output_row_stride is not supported on the quantized "
                    "FP8 path, whose C is column-major" );
            }
        }

        // Activation type (A non-quantized / B quantized), weight type (B non-quantized / A quantized)
        constexpr cudaDataType_t data_type_activation = cuda_data_type_v<TComputePrecision>;
        constexpr cudaDataType_t data_type_weight      = cuda_data_type_v<TParameterPrecision>;
        constexpr cudaDataType_t data_type_output      = cuda_data_type_v<TComputePrecision>;

        CublasLtLinearPlan<TComputePrecision, TParameterPrecision> plan;
        plan.has_bias_epilogue = has_bias && !kIsQuantized;  // bias is added manually on quantized path

        // --- matmul descriptor ---
        cublasStatus_t status = cublasLtMatmulDescCreate( &plan.matmul_desc, compute_type, scale_type );
        if ( status != CUBLAS_STATUS_SUCCESS )
        {
            Logging::Logger::error( "cublasLtMatmulDescCreate failed: " + std::to_string( status ) );
            cublasLtCheckStatus( status );
        }

        if constexpr ( kIsQuantized )
        {
            // TN column-major: A = weight (FP8, transposed), B = activation (BF16, no-transpose)
            const cublasOperation_t opA = CUBLAS_OP_T;
            const cublasOperation_t opB = CUBLAS_OP_N;

            status = cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof( opA ) );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "Set TRANSA failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }

            status = cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof( opB ) );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "Set TRANSB failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }

            // Column-major layouts (default order -- do not set CUBLASLT_ORDER_ROW).
            // A = weight:      col-major [K x N], lda = K   (row-major W[N, K] same bytes)
            // B = activation:  col-major [K x M], ldb = K   (row-major X[M, K] same bytes)
            // C = output:      col-major [N x M], ldc = N   (row-major Y[M, N] same bytes)
            status = cublasLtMatrixLayoutCreate(
                &plan.layoutA, data_type_weight,      in_features,  out_features, in_features );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "layoutA (weight) create failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }

            status = cublasLtMatrixLayoutCreate(
                &plan.layoutB, data_type_activation,  in_features,  outer_size,   in_features );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "layoutB (activation) create failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }

            status = cublasLtMatrixLayoutCreate(
                &plan.layoutC, data_type_output,      out_features, outer_size,   out_features );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "layoutC (output) create failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }

            // A_SCALE_POINTER must be set before cublasLtMatmulAlgoGetHeuristic so the
            // planner can enumerate FP8 tensor-core algorithms. The same pointer is also
            // re-set at execution time by execute_linear_plan.
            if ( weight_scale != nullptr )
            {
                status = cublasLtMatmulDescSetAttribute(
                    plan.matmul_desc,
                    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    &weight_scale, sizeof( weight_scale ) );
                if ( status != CUBLAS_STATUS_SUCCESS )
                {
                    Logging::Logger::error( "Set A_SCALE_POINTER failed: " + std::to_string( status ) );
                    cublasLtCheckStatus( status );
                }
            }

            // Fast accumulation enables the FP8 tensor-core algorithm path on Ada.
            // Without this the heuristic returns no valid algorithms.
            const int8_t fast_accum = 1;
            status = cublasLtMatmulDescSetAttribute(
                plan.matmul_desc,
                CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                &fast_accum, sizeof( fast_accum ) );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "Set FAST_ACCUM failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }
        }
        else
        {
            // NT row-major: A = activation (no-transpose), B = weight (transposed)
            const cublasOperation_t opA = CUBLAS_OP_N;
            const cublasOperation_t opB = CUBLAS_OP_T;

            status = cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof( opA ) );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "Set TRANSA failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }

            status = cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof( opB ) );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "Set TRANSB failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }

            if ( has_bias )
            {
                const int epilogue = CUBLASLT_EPILOGUE_BIAS;
                cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                    plan.matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof( epilogue ) ) );
            }

            // Row-major layouts.
            // A = activation:  [M x K], lda = K
            // B = weight:      [N x K], ldb = K
            // C = output:      [M x N], ldc = N, or the caller's wider stride so that C is a
            //                  column slice written in place.
            const int output_stride = ( output_row_stride != 0 ) ? output_row_stride : out_features;

            status = cublasLtMatrixLayoutCreate(
                &plan.layoutA, data_type_activation, outer_size,   in_features,  in_features );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "layoutA (activation) create failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }

            status = cublasLtMatrixLayoutCreate(
                &plan.layoutB, data_type_weight,     out_features, in_features,  in_features );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "layoutB (weight) create failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }

            status = cublasLtMatrixLayoutCreate(
                &plan.layoutC, data_type_output,     outer_size,   out_features, output_stride );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "layoutC (output) create failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }

            const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

            status = cublasLtMatrixLayoutSetAttribute(
                plan.layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof( order ) );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "Set row order for A failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }

            status = cublasLtMatrixLayoutSetAttribute(
                plan.layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof( order ) );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "Set row order for B failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }

            status = cublasLtMatrixLayoutSetAttribute(
                plan.layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof( order ) );
            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                Logging::Logger::error( "Set row order for C failed: " + std::to_string( status ) );
                cublasLtCheckStatus( status );
            }
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
            Logging::Logger::warning( "cuBLASLt heuristic found no algorithms, will use default at execution" );
            plan.algorithm = {};
            plan.has_algorithm = false;
        }
        else
        {
            Logging::Logger::error( "cuBLASLt heuristic failed with error status" );
            cublasLtCheckStatus( status );
        }

        return plan;
    }

    /**
     * @brief Execute a previously-built CublasLtLinearPlan.
     *
     * Computes: D = alpha * op(A) * op(B) + beta * C
     * with optional bias epilogue (non-quantized path) and optional weight scale (FP8 path).
     *
     * Pointer semantics match the layout built by build_linear_plan:
     *   Non-quantized (NT row-major): A = activations, B = weights
     *   Quantized (TN col-major):     A = weights (FP8), B = activations (BF16)
     *
     * Both A and B are passed as const void* to accommodate the pointer-swap between
     * the two layouts without template type conflicts. Type safety is enforced at the
     * plan-build level via the layout descriptors.
     *
     * @param A             Device pointer to matrix A (see layout note above).
     * @param B             Device pointer to matrix B (see layout note above).
     * @param C             Device pointer to output (TComputePrecision).
     * @param bias          Device pointer to bias vector; used only when has_bias_epilogue.
     * @param weight_scale  Device pointer to per-tensor weight scale (float);
     *                      applied as A_SCALE_POINTER when has_weight_scale is true.
     *                      For the TN path weight_scales_[0] is the global per-tensor scale.
     * @param workspace     Optional device scratch buffer.
     * @param workspace_size Size of workspace in bytes.
     */
    export template<TensorDataType TComputePrecision, TensorDataType TParameterPrecision = TComputePrecision>
        void execute_linear_plan(
            cublasLtHandle_t handle,
            const CublasLtLinearPlan<TComputePrecision, TParameterPrecision>& plan,
            const float* alpha,
            const void*  A,
            const void*  B,
            const float* beta,
            typename CublasLtLinearPlan<TComputePrecision, TParameterPrecision>::ActivationType* C,
            const typename CublasLtLinearPlan<TComputePrecision, TParameterPrecision>::ActivationType* bias,
            const float* weight_scale,
            cudaStream_t stream,
            void*   workspace      = nullptr,
            size_t  workspace_size = 0 )
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

        if ( plan.has_weight_scale && weight_scale != nullptr )
        {
            // Quantized TN path: A = weights (FP8), so scale goes to A_SCALE_POINTER.
            // weight_scale points to a device float holding the per-tensor scale value.
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weight_scale, sizeof( weight_scale ) ) );
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

    /**
     * @brief Build a cuBLASLt FP8xFP8 GEMM plan for the W4A8-FP8 prefill path.
     *
     * Both operands are FP8_E4M3; the output is TComputePrecision (BF16). This is the
     * true FP8 tensor-core path (~2x BF16 on Ada, measured), distinct from the FP8-weight
     * x BF16-activation mixed layout in build_linear_plan (which stages to BF16). No fast
     * accumulate -- the microbench shows regular FP32 accumulate already delivers the 2x,
     * so accumulation precision is kept.
     *
     * TN column-major (identical operand roles to the quantized build_linear_plan branch):
     *   A = weight (FP8)      col-major [K x N], lda = K    op(A) = W[N, K]
     *   B = activation (FP8)  col-major [K x M], ldb = K    op(B) = X^T[K, M]
     *   C = output            col-major [N x M], ldc = N    == row-major Y[M, N]
     *   A_SCALE_POINTER = weight scale (per-tensor, static),  B_SCALE_POINTER = constant 1.0f
     *
     * Ada cuBLASLt accepts only per-tensor scale pointers, so the activation B_SCALE is a
     * constant unit scalar; the true per-token activation scales are applied exactly by a
     * post-GEMM epilogue in the caller (cuda_fp8_apply_per_token_scales). Both scale
     * pointers are bound at build time -- they are stable device scalars, and they must be
     * present in the descriptor before cublasLtMatmulAlgoGetHeuristic so FP8 algorithms
     * are enumerated.
     *
     * @param outer_size        Token count M.
     * @param in_features       Inner dimension K.
     * @param out_features       Output channels N.
     * @param activation_scale  Device float scalar for the FP8 activation operand (B_SCALE, 1.0f).
     * @param weight_scale      Device float scalar for the FP8 weight operand (A_SCALE).
     */
    export template<TensorDataType TComputePrecision>
        CublasLtLinearPlan<TComputePrecision, TensorDataType::FP8_E4M3> build_fp8_prefill_plan(
            cublasLtHandle_t handle,
            int outer_size,
            int in_features,
            int out_features,
            const float* activation_scale,
            const float* weight_scale )
    {
        constexpr cudaDataType_t data_type_output = cuda_data_type_v<TComputePrecision>;

        CublasLtLinearPlan<TComputePrecision, TensorDataType::FP8_E4M3> plan;
        plan.has_bias_epilogue = false;  // bias is added post-GEMM by cuda_add_bias

        cublasLtCheckStatus( cublasLtMatmulDescCreate( &plan.matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F ) );

        // TN column-major: A = weight (transposed), B = activation (no-transpose).
        const cublasOperation_t opA = CUBLAS_OP_T;
        const cublasOperation_t opB = CUBLAS_OP_N;
        cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
            plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof( opA ) ) );
        cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
            plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof( opB ) ) );

        // Scale pointers must be set before the heuristic call so FP8 algorithms enumerate.
        cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
            plan.matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weight_scale, sizeof( weight_scale ) ) );
        cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
            plan.matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &activation_scale, sizeof( activation_scale ) ) );

        // Column-major layouts (default order -- do not set CUBLASLT_ORDER_ROW).
        cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
            &plan.layoutA, CUDA_R_8F_E4M3, in_features,  out_features, in_features ) );
        cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
            &plan.layoutB, CUDA_R_8F_E4M3, in_features,  outer_size,   in_features ) );
        cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
            &plan.layoutC, data_type_output, out_features, outer_size, out_features ) );

        cublasLtCheckStatus( cublasLtMatmulPreferenceCreate( &plan.preference ) );

        constexpr size_t kWorkspaceHint = 4ull * 1024 * 1024;
        cublasLtCheckStatus( cublasLtMatmulPreferenceSetAttribute(
            plan.preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kWorkspaceHint, sizeof( kWorkspaceHint ) ) );

        cublasLtMatmulHeuristicResult_t heuristic_result{};
        int returned_algo_count = 0;

        cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
            handle, plan.matmul_desc,
            plan.layoutA, plan.layoutB, plan.layoutC, plan.layoutC,
            plan.preference, 1, &heuristic_result, &returned_algo_count );

        if ( status == CUBLAS_STATUS_SUCCESS && returned_algo_count > 0 )
        {
            plan.algorithm = heuristic_result.algo;
            plan.has_algorithm = true;
        }
        else
        {
            Logging::Logger::warning( "cuBLASLt FP8 prefill heuristic found no algorithm, will use default at execution" );
            plan.has_algorithm = false;
        }

        return plan;
    }

    /**
     * @brief Execute a previously-built FP8xFP8 prefill plan.
     *
     * Computes D = alpha * (A_scale * op(A)) * (B_scale * op(B)) + beta * C, with the
     * scale pointers already bound in the descriptor by build_fp8_prefill_plan.
     *
     * @param weight_fp8      Device FP8_E4M3 weight staging buffer (matrix A).
     * @param activation_fp8  Device FP8_E4M3 activation buffer (matrix B).
     * @param C               Device output (TComputePrecision).
     */
    export template<TensorDataType TComputePrecision>
        void execute_fp8_prefill_plan(
            cublasLtHandle_t handle,
            const CublasLtLinearPlan<TComputePrecision, TensorDataType::FP8_E4M3>& plan,
            const float* alpha,
            const void*  weight_fp8,
            const void*  activation_fp8,
            const float* beta,
            typename CublasLtLinearPlan<TComputePrecision, TensorDataType::FP8_E4M3>::ActivationType* C,
            cudaStream_t stream,
            void*   workspace      = nullptr,
            size_t  workspace_size = 0 )
    {
        if ( !plan.isValid() )
        {
            throw std::invalid_argument( "execute_fp8_prefill_plan - plan is not valid" );
        }

        const cublasLtMatmulAlgo_t* algo_ptr = plan.has_algorithm ? &plan.algorithm : nullptr;

        cublasStatus_t status = cublasLtMatmul(
            handle,
            plan.matmul_desc,
            alpha,
            weight_fp8,     plan.layoutA,
            activation_fp8, plan.layoutB,
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