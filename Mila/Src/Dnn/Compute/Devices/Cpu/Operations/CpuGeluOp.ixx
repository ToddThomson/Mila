module;
#include <memory>
#include <vector>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#ifdef USE_OMP
#include <omp.h>
#endif
#include <iostream>
#include <stdexcept>

export module Compute.CpuGeluOp;

import Dnn.Modules.Gelu;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ModuleConfig;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.CpuExecutionContext;
import Compute.OperationType;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CpuTensorDataTypeTraits;
import Compute.CpuDevice;
import Compute.Precision;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Scaling factor for GELU tanh approximation: sqrt(2/pi)
     *
     * Used in the tanh approximation formula:
     * GELU(x) ~= 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
     */
    constexpr float GELU_SCALING_FACTOR = 0.7978845608f;  // sqrt(2/pi)

    /**
     * @brief CPU implementation of GELU activation operation using abstract TensorDataType
     *
     * Implements the Gaussian Error Linear Unit (GELU) activation function for CPU devices.
     * Supports multiple approximation methods as configured via GeluConfig:
     * - Exact: Uses error function (not yet implemented)
     * - Tanh: Fast approximation using tanh (default, implemented)
     * - Sigmoid: Fast approximation using sigmoid (not yet implemented)
     *
     * Key features:
     * - Uses abstract TensorDataType enumeration to represent compute precision
     * - Supports scalar tensor operations (rank 0)
     * - OpenMP parallelization for large tensors
     * - Element-wise activation preserving tensor shape
     *
     * @tparam TPrecision Abstract compute precision (TensorDataType enum)
     *
     * @note Currently only FP32 (float) is fully supported
     * @note Other precisions will require template specialization
     */
    export class CpuGeluOp : public UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using MR = CpuMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorType = Tensor<TensorDataType::FP32, MR>;
        using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;

        /**
         * @brief Constructs a new CpuGeluOp with a specific execution context.
         *
         * @param context The execution context to use for this operation.
         * @param config Configuration for GELU operation.
         * @throws std::runtime_error If the context is not for a CPU device.
         */
        CpuGeluOp(  std::shared_ptr<CpuExecutionContext> context, const GeluConfig& config )
            : context_( context ), config_( config )
        {
            if (!context_)
            {
                throw std::runtime_error( "CpuGeluOp requires a CPU execution context" );
            }
        }

		// ====================================================================
		// Parameters and Gradients
        // 
		// Note: GELU has no learnable parameters. 
		//       The OperationBase setParameters() add setParameterGradients() is used as a no-op.
		// ====================================================================
        
		// ====================================================================
		// Lifecycle
		// ====================================================================

        void build( const shape_t& input_shape ) override
        {
			// No shape-dependent setup required for this implementation.
			// The default OperationBase build() could be used instead.

			this->is_built_ = true;
		}

		// ====================================================================
		// Computation
		// ====================================================================

        /**
         * @brief Performs the forward pass of the GELU activation function.
         *
         * Implements the Gaussian Error Linear Unit (GELU) activation function using
         * the tanh approximation:
         * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
         *
         * Tensor shape handling:
         * - Scalar input (rank 0): Produces scalar output
         * - Vector/matrix input: Produces output with same shape
         * - Element-wise operation preserving input shape
         *
         * @param input The input tensor (may be scalar, rank 0).
         * @param parameters Parameter tensors (not used in GELU).
         * @param output The output tensor (resized to match input shape).
         * @param output_state Cache for intermediate results (not used in current implementation).
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            if (!this->is_built_)
            {
                throw std::runtime_error( "GeluOp not built - call build() first" );
            }

            // Obtain host element buffers (float) once before the loop.
            const float* input_data = static_cast<const float*>(input.rawData());
            float* output_data = static_cast<float*>(output.rawData());

            if (!input_data || !output_data)
            {
                throw std::runtime_error( "CpuGeluOp::forward - null tensor data pointer" );
            }

            const size_t N = input.size();

#pragma omp parallel for if(N > 1000)
            for (int i = 0; i < static_cast<int>( N ); i++)
            {
                float x = input_data[i];
                float cube = static_cast<float>( 0.044715f * x * x * x );
                output_data[i] = static_cast<float>( 0.5f * x * (1.0f + tanhf( GELU_SCALING_FACTOR * (x + cube) )) );
            }
        }

        /**
         * @brief Performs the backward pass of the GELU activation function.
         *
         * Computes the gradient of the GELU function with respect to its input.
         *
         * @param input Original input tensor from forward pass (may be scalar).
         * @param output_grad Gradient from next layer (dL/doutput, may be scalar).
         * @param parameters Parameter tensors (not used in GELU).
         * @param parameter_grads Parameter gradients (not used in GELU).
         * @param input_grad Gradient to propagate to previous layer (dL/dinput, may be scalar).
         * @param output_state Cached tensors from forward pass (not used currently).
         */

        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const {

            // Resize input_grad to match input shape if needed
            if (input_grad.shape() != input.shape())
            {
            }

            // General tensor case
            auto inp = static_cast<const float*>(input.rawData());
            auto dout = static_cast<const float*>(output_grad.rawData());
            auto dinp = static_cast<float*>(input_grad.rawData());

            if (!inp || !dout || !dinp)
            {
                throw std::runtime_error( "CpuGeluOp::backward - null tensor data pointer" );
            }

            const size_t N = input.size();

            // Use OpenMP for larger tensors
#pragma omp parallel for if(N > 1000)
            for (int i = 0; i < static_cast<int>( N ); i++)
            {
                float x = inp[i];
                float cube = static_cast<float>( 0.044715f * x * x * x );
                float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
                float tanh_out = tanhf( tanh_arg );
                float coshf_out = coshf( tanh_arg );
                float sech_out = 1.0f / (coshf_out * coshf_out);
                float local_grad = static_cast<float>(
                    0.5f * (1.0f + tanh_out) +
                    x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x)
                    );
                dinp[i] += local_grad * dout[i];
            }
        }

        OperationType getOperationType() const override {
            return OperationType::GeluOp;
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cpu::GeluOp").
         */
        std::string getName() const override {
            return "Cpu::GeluOp";
        }

    private:
        
        GeluConfig config_; ///< Configuration for the GELU operation (approximation method, etc.)
        std::shared_ptr<CpuExecutionContext> context_;
    };

    /**
     * @brief Class responsible for registering CPU GELU operations
     *
     * Registers CPU GELU operation implementations with the OperationRegistry
     * for different precisions. Currently supports FP32.
     *
     * The registrar uses static initialization to automatically register
     * operations when the module is loaded.
     */
    export class CpuGeluOpRegistrar {
    public:
        /**
         * @brief Registers CPU GELU operations with the OperationRegistry
         *
         * Registers factory functions for creating CPU GELU operations with
         * different data types. Currently registers:
         * - FP32 (TensorDataType::FP32)
         */
        static void registerOperations()
        {
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, TensorDataType::FP32, TensorDataType::FP32>(
                "GeluOp",
                []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
                    const ModuleConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32, TensorDataType::FP32>>
                {
                    const auto& geluConfig = static_cast<const GeluConfig&>(config);
                    
                    return std::make_shared<CpuGeluOp>( context, geluConfig );
                }
            );
        }

        /**
         * @brief Static initialization flag ensuring operations are registered
         *
         * This static member is initialized before main(), causing registerOperations()
         * to be called automatically when the module is loaded.
         */
        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}