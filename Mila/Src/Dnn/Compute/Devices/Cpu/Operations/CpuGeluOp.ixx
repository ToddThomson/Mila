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
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
import Dnn.ConfigurationBase;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
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
     * - Uses abstract TensorDataType enumeration for type safety
     * - Supports scalar tensor operations (rank 0)
     * - OpenMP parallelization for large tensors
     * - Element-wise activation preserving tensor shape
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
     *
     * @note Currently only FP32 (float) is fully supported
     * @note Other data types will require template specialization
     */
    export template<TensorDataType TDataType = TensorDataType::FP32>
        class CpuGeluOp : public UnaryOperation<DeviceType::Cpu, TDataType> {
        public:
            using MR = CpuMemoryResource;
            using UnaryOperationBase = UnaryOperation<DeviceType::Cpu, TDataType>;
            using TensorType = Tensor<TDataType, MR>;
            using HostType = typename TensorHostTypeMap<TDataType>::host_type;

            /**
             * @brief Constructs a new CpuGeluOp with the default device context.
             *
             * @param config Configuration for GELU operation (approximation method, etc.)
             */
            CpuGeluOp( const GeluConfig& config )
                : UnaryOperationBase( OperationType::GeluOp ), config_( config ) {
            }

            /**
             * @brief Constructs a new CpuGeluOp with a specific device context.
             *
             * @param context The device context to use for this operation.
             * @param config Configuration for GELU operation.
             * @throws std::runtime_error If the context is not for a CPU device.
             */
            CpuGeluOp( std::shared_ptr<DeviceContext> context, const GeluConfig& config )
                : UnaryOperationBase( OperationType::GeluOp, context ), config_( config ) {

                if (context->getDevice()->getDeviceType() != DeviceType::Cpu) {
                    throw std::runtime_error( "CpuGeluOp requires a CPU device context" );
                }
            }

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
             *
             * Example:
             * @code
             * // Regular tensor
             * Tensor<TensorDataType::FP32, CpuMemoryResource> input("CPU", {128, 768});
             * Tensor<TensorDataType::FP32, CpuMemoryResource> output("CPU", {});
             * op.forward(input, {}, output, {});
             * // output.shape() == {128, 768}
             *
             * // Scalar tensor
             * Tensor<TensorDataType::FP32, CpuMemoryResource> scalar("CPU", {});
             * scalar.item() = 1.5f;
             * Tensor<TensorDataType::FP32, CpuMemoryResource> scalar_out("CPU", {});
             * op.forward(scalar, {}, scalar_out, {});
             * // scalar_out.item() contains activated value
             * @endcode
             */
            void forward(
                const TensorType& input,
                const std::vector<std::shared_ptr<ITensor>>& parameters,
                TensorType& output,
                std::vector<std::shared_ptr<TensorType>>& output_state ) const override {

                // Resize output to match input shape
                output.reshape( input.shape() );

                // Handle scalar case specially
                if (input.isScalar()) {
                    HostType x = input.item();
                    HostType cube = 0.044715f * x * x * x;
                    output.item() = static_cast<HostType>(0.5f * x * (1.0f + tanhf( GELU_SCALING_FACTOR * (x + cube) )));
                    return;
                }

                // General tensor case
                auto X = input.data();
                auto Y = output.data();

                const size_t N = input.size();

                // Use OpenMP for larger tensors
#pragma omp parallel for if(N > 1000)
                for (int i = 0; i < static_cast<int>( N ); i++) {
                    HostType x = X[i];
                    HostType cube = static_cast<HostType>( 0.044715f * x * x * x );
                    Y[i] = static_cast<HostType>( 0.5f * x * (1.0f + tanhf( GELU_SCALING_FACTOR * (x + cube) )) );
                }
            }

            /**
             * @brief Performs the backward pass of the GELU activation function.
             *
             * Computes the gradient of the GELU function with respect to its input.
             * The derivative of GELU using tanh approximation is:
             *
             * d/dx GELU(x) = 0.5 * (1 + tanh(arg)) + x * 0.5 * sech^2(arg) * sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
             *
             * where arg = sqrt(2/pi) * (x + 0.044715 * x^3)
             *
             * Gradient computation:
             * - Scalar gradient: grad_out is scalar -> grad_in is scalar
             * - Tensor gradient: element-wise multiplication preserving shape
             * - Uses cached input values for efficient computation
             *
             * @param input Original input tensor from forward pass (may be scalar).
             * @param output_grad Gradient from next layer (dL/doutput, may be scalar).
             * @param parameters Parameter tensors (not used in GELU).
             * @param parameter_grads Parameter gradients (not used in GELU).
             * @param input_grad Gradient to propagate to previous layer (dL/dinput, may be scalar).
             * @param output_state Cached tensors from forward pass (not used currently).
             *
             * @note input_grad is accumulated (+=), not overwritten, to support gradient accumulation
             *
             * Example:
             * @code
             * // Regular tensor backward
             * Tensor<TensorDataType::FP32, CpuMemoryResource> input("CPU", {128, 768});
             * Tensor<TensorDataType::FP32, CpuMemoryResource> output_grad("CPU", {128, 768});
             * Tensor<TensorDataType::FP32, CpuMemoryResource> input_grad("CPU", {128, 768});
             * op.backward(input, output_grad, {}, {}, input_grad, {});
             *
             * // Scalar backward
             * Tensor<TensorDataType::FP32, CpuMemoryResource> scalar_in("CPU", {});
             * Tensor<TensorDataType::FP32, CpuMemoryResource> scalar_grad_out("CPU", {});
             * Tensor<TensorDataType::FP32, CpuMemoryResource> scalar_grad_in("CPU", {});
             * op.backward(scalar_in, scalar_grad_out, {}, {}, scalar_grad_in, {});
             * @endcode
             */
            void backward(
                const TensorType& input,
                const TensorType& output_grad,
                const std::vector<std::shared_ptr<ITensor>>& parameters,
                std::vector<std::shared_ptr<TensorType>>& parameter_grads,
                TensorType& input_grad,
                const std::vector<std::shared_ptr<TensorType>>& output_state ) const override {

                // Verify CPU device
                if (this->getDeviceContext()->getDevice()->getDeviceType() != DeviceType::Cpu) {
                    throw std::runtime_error( "CpuGeluOp::backward can only be executed on CPU memory" );
                }

                // Resize input_grad to match input shape if needed
                if (input_grad.shape() != input.shape()) {
                    input_grad.reshape( input.shape() );
                }

                // Handle scalar case specially
                if (input.isScalar()) {
                    HostType x = input.item();
                    HostType cube = static_cast<HostType>(0.044715f * x * x * x);
                    float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
                    float tanh_out = tanhf( tanh_arg );
                    float coshf_out = coshf( tanh_arg );
                    float sech_out = 1.0f / (coshf_out * coshf_out);
                    HostType local_grad = static_cast<HostType>(
                        0.5f * (1.0f + tanh_out) +
                        x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x)
                        );
                    input_grad.item() += local_grad * output_grad.item();
                    return;
                }

                // General tensor case
                auto inp = input.data();
                auto dout = output_grad.data();
                auto dinp = input_grad.data();

                const size_t N = input.size();

                // Use OpenMP for larger tensors
#pragma omp parallel for if(N > 1000)
                for (int i = 0; i < static_cast<int>( N ); i++) {
                    HostType x = inp[i];
                    HostType cube = static_cast<HostType>( 0.044715f * x * x * x );
                    float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
                    float tanh_out = tanhf( tanh_arg );
                    float coshf_out = coshf( tanh_arg );
                    float sech_out = 1.0f / (coshf_out * coshf_out);
                    HostType local_grad = static_cast<HostType>(
                        0.5f * (1.0f + tanh_out) +
                        x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x)
                        );
                    dinp[i] += local_grad * dout[i];
                }
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
    };

    /**
     * @brief Template specialization for FP32 (float) data type
     *
     * This is the primary implementation used by most CPU operations.
     */
    using CpuGeluOpFP32 = CpuGeluOp<TensorDataType::FP32>;

    /**
     * @brief Class responsible for registering CPU GELU operations
     *
     * Registers CPU GELU operation implementations with the OperationRegistry
     * for different data types. Currently supports FP32.
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
         *
         * Future data types (FP16, etc.) can be added here.
         */
        static void registerOperations() {
            // Register FP32 version
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>(
                "Cpu::GeluOp",
                []( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config )
                -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>> {
                    const auto& geluConfig = static_cast<const GeluConfig&>(config);
                    return context
                        ? std::make_shared<CpuGeluOpFP32>( context, geluConfig )
                        : std::make_shared<CpuGeluOpFP32>( geluConfig );
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