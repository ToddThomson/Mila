/**
 * @file Gelu.ixx
 * @brief Implementation of the Gaussian Error Linear Unit (GELU) activation function.
 *
 * This module implements the GELU activation function as described in:
 * "Gaussian Error Linear Units (GELUs)" by Hendrycks and Gimpel (2016).
 * https://arxiv.org/abs/1606.08415
 *
 * GELU activation has become a standard component in transformer architectures
 * like BERT, GPT, and their derivatives, often replacing traditional ReLU activations.
 *
 * Key features:
 * - Abstract data type system using TensorDataType enumeration
 * - Device-agnostic design (CPU, CUDA)
 * - Support for scalar tensor operations (element-wise activation)
 * - Multiple approximation methods (Exact, Tanh, Sigmoid)
 * - Zero trainable parameters (stateless activation)
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>

export module Dnn.Modules.Gelu;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Precision;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.ComputeDevice;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Gaussian Error Linear Unit (GELU) activation function module.
     *
     * GELU is defined mathematically as:
     * GELU(x) = x * Phi(x)
     *
     * Where Phi(x) is the cumulative distribution function of the standard normal distribution.
     *
     * Three approximation methods are supported (configured via GeluConfig):
     * 1. Exact: Uses the error function - most accurate but computationally expensive
     *    GELU(x) = 0.5x(1 + erf(x/sqrt(2)))
     * 2. Tanh: Fast approximation using tanh
     *    GELU(x) ~= 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
     * 3. Sigmoid: Fast approximation using sigmoid
     *    GELU(x) ~= x * sigmoid(1.702x)
     *
     * Scalar tensor support:
     * - Operates element-wise on tensors of any rank
     * - Supports scalar inputs (rank 0) for single-value activation
     * - Preserves input shape in output
     *
     * @tparam TDeviceType Computing device type (CPU or CUDA)
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
     *
     * Example usage:
     * @code
     * // FP32 CUDA GELU with tanh approximation
     * GeluConfig config("gelu_1", GeluConfig::ApproximationMethod::Tanh);
     * auto gelu = Gelu<DeviceType::Cuda, TensorDataType::FP32>("CUDA:0", config);
     *
     * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input("CUDA:0", {128, 512});
     * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> output("CUDA:0", {128, 512});
     * gelu.forward(input, output);
     *
     * // Scalar activation
     * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> scalar("CUDA:0", {});
     * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> scalar_out("CUDA:0", {});
     * gelu.forward(scalar, scalar_out); // Activates single value
     * @endcode
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda /*, TensorDataType TDataType = TensorDataType::FP32 */>
    class Gelu : public Module<TDeviceType /*, TDataType */> {
        public:
            /**
             * @brief Memory resource type determined based on device type.
             *
             * Automatically selects appropriate memory resource (CPU or CUDA) based on TDeviceType.
             */
            using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;

            /**
             * @brief Alias for base module type.
             */
            using ModuleBase = Module<TDeviceType /*, TDataType */>;

            /**
             * @brief Tensor type used by this module.
             */
            //using TensorType = Tensor<TDataType, MR>;

            /**
             * @brief Constructs a Gelu module using device name and configuration.
             *
             * Creates a new DeviceContext internally using the provided device name.
             * This constructor is useful for creating standalone modules without
             * pre-existing device contexts.
             *
             * @param device_name Device identifier string (e.g., "CPU", "CUDA:0")
             * @param config Configuration parameters for the GELU module
             * @throws std::invalid_argument If the device name is invalid or the configuration is invalid
             * @throws std::runtime_error If device type doesn't match template parameter TDeviceType
             *
             * Example:
             * @code
             * GeluConfig config("gelu_activation", GeluConfig::ApproximationMethod::Tanh);
             * auto gelu = Gelu<DeviceType::Cuda, TensorDataType::FP32>("CUDA:0", config);
             * @endcode
             */
            explicit Gelu( const std::string& device_name, const GeluConfig& config )
                : ModuleBase( device_name, config ), config_( config ) {

                config.validate();

                createOperation();
            }

            /**
             * @brief Constructs a Gelu module with an existing device context and configuration.
             *
             * Uses a pre-existing DeviceContext instance. This constructor is useful when integrating
             * the module into a larger network that shares device contexts across modules.
             *
             * @param device_context Shared pointer to an existing device context
             * @param config Configuration parameters for the GELU module
             * @throws std::invalid_argument If device_context is null or configuration is invalid
             * @throws std::runtime_error If device context type doesn't match template parameter TDeviceType
             *
             * Example:
             * @code
             * auto context = DeviceContext::create("CUDA:0");
             * GeluConfig config("gelu_activation", GeluConfig::ApproximationMethod::Tanh);
             * auto gelu = Gelu<DeviceType::Cuda, TensorDataType::FP32>(context, config);
             * @endcode
             */
            explicit Gelu( std::shared_ptr<DeviceContext> device_context, const GeluConfig& config )
                : ModuleBase( device_context, config ), config_( config ) {

                config.validate();

                createOperation();
            }

            /**
             * @brief Returns the number of trainable parameters in this module.
             *
             * GELU is a parameterless activation function with no trainable weights.
             *
             * @return Always returns 0
             */
            size_t parameterCount() const override {
                return 0;
            }

            /**
             * @brief Performs forward propagation through the GELU activation function.
             *
             * Applies the GELU transformation element-wise to each value in the input tensor.
             * The specific approximation method used is determined by the GeluConfig setting.
             *
             * Input/Output tensor requirements:
             * - Input may have any rank (0 for scalars, 1+ for higher dimensions)
             * - Output shape will match input shape exactly
             * - Scalar inputs produce scalar outputs
             *
             * @param input Input tensor to transform (may be scalar, rank 0).
             * @param output Tensor where results will be stored (resized to match input shape).
             *
             * Example:
             * @code
             * // Regular tensor activation
             * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input("CUDA:0", {128, 768});
             * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> output("CUDA:0", {});
             * gelu.forward(input, output);
             * // output.shape() == {128, 768}
             *
             * // Scalar activation
             * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> scalar_in("CUDA:0", {});
             * scalar_in.item() = 1.5f;
             * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> scalar_out("CUDA:0", {});
             * gelu.forward(scalar_in, scalar_out);
             * // scalar_out.item() contains activated value
             * @endcode
             */
            void forward( const ITensor& input, ITensor& output ) override {
                operation_->forward( input, parameters_, output, output_state_ );
            }

            /**
             * @brief Performs backward propagation, computing gradients for GELU activation.
             *
             * Computes the gradient of the GELU function with respect to its inputs,
             * which is needed for training via backpropagation.
             *
             * The GELU derivative depends on the approximation method:
             * - Exact: d/dx GELU(x) = Phi(x) + x * phi(x)
             *   where phi(x) is the PDF of the standard normal distribution
             * - Tanh: Derivative of the tanh approximation formula
             * - Sigmoid: Derivative of the sigmoid approximation formula
             *
             * Gradient computation:
             * - Element-wise gradient multiplication
             * - Preserves tensor shape (including scalars)
             * - Scalar gradients: grad_out is scalar -> grad_in is scalar
             *
             * @param input Original input tensor from the forward pass (may be scalar).
             * @param output_grad Gradient tensor from the next layer (dL/doutput, may be scalar).
             * @param input_grad Output tensor to store the computed gradients (dL/dinput, may be scalar).
             *
             * Example:
             * @code
             * // Regular tensor backward pass
             * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input("CUDA:0", {128, 768});
             * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> output_grad("CUDA:0", {128, 768});
             * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input_grad("CUDA:0", {});
             * gelu.backward(input, output_grad, input_grad);
             * // input_grad.shape() == {128, 768}
             *
             * // Scalar backward pass
             * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> scalar_in("CUDA:0", {});
             * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> scalar_grad_out("CUDA:0", {});
             * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> scalar_grad_in("CUDA:0", {});
             * gelu.backward(scalar_in, scalar_grad_out, scalar_grad_in);
             * // scalar_grad_in contains gradient for scalar input
             * @endcode
             */
            void backward(
                const ITensor& input,
                const ITensor& output_grad,
                ITensor& input_grad ) override {

                // GELU has no parameters, so parameter_gradients is empty
                std::vector<std::shared_ptr<ITensor>> parameter_gradients;

                operation_->backward(
                    input,                  // Input tensor from forward pass
                    output_grad,            // Gradient from next layer
                    parameters_,            // Empty for GELU (no parameters)
                    parameter_gradients,    // Empty (no parameter gradients)
                    input_grad,             // Gradient to propagate to previous layer
                    output_state_           // Cached tensors from forward pass
                );
            }

            /**
             * @brief Returns the current approximation method used by this GELU instance.
             *
             * @return Current approximation method from GeluConfig::ApproximationMethod enum
             */
            GeluConfig::ApproximationMethod getApproximationMethod() const {
                return config_.getApproximationMethod();
            }

            /**
             * @brief Serializes module state to a model archive.
             *
             * Implementation of the Module interface for serialization. Since GELU has no
             * learnable parameters, this is a no-op implementation.
             *
             * @param archive Model archive for serialization
             */
            void save( ModelArchive& archive ) const override {
                // No-op: GELU is a stateless activation function with no parameters to persist
            }

            /**
             * @brief Deserializes module state from a model archive.
             *
             * Implementation of the Module interface for deserialization. Since GELU has no
             * learnable parameters, this is a no-op implementation.
             *
             * @param archive Model archive for deserialization
             */
            void load( ModelArchive& archive ) override {
                // No-op: GELU is a stateless activation function with no parameters to load
            }

            /**
             * @brief Generates a string representation of this module's configuration.
             *
             * @return Formatted string with module name, device information, data type, and approximation method
             */
            std::string toString() const override {
                std::ostringstream oss;
                oss << "--------------------" << std::endl;
                oss << "Gelu: " << this->getDeviceName() << std::endl;
                oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
                oss << "Data Type: " << this->getDataTypeName() << std::endl;
                oss << "Approximation Method: " << approximationMethodToString( config_.getApproximationMethod() ) << std::endl;

                return oss.str();
            }

        private:
            /**
             * @brief Configuration for the GELU module.
             *
             * Stores the settings that define how the GELU function should be computed,
             * particularly which approximation method to use.
             */
            GeluConfig config_;

            /**
             * @brief Parameter tensors for the operation.
             *
             * Empty for GELU since it has no trainable parameters, but required by the
             * UnaryOperation interface.
             */
            std::vector<std::shared_ptr<ITensor>> parameters_;

            /**
             * @brief Output state cache for backward propagation.
             *
             * Stores intermediate results from the forward pass that may be needed
             * during backward propagation to efficiently compute gradients.
             * May include scalar tensors if caching normalized values.
             */
            std::vector<std::shared_ptr<ITensor>> output_state_;

            /**
             * @brief The underlying computational operation that implements GELU.
             *
             * This pointer is initialized based on the device type and configuration,
             * providing the device-specific implementation of the GELU function.
             * Uses abstract TensorDataType for type-safe operation dispatch.
             */
            std::shared_ptr<UnaryOperation<TDeviceType>> operation_{ nullptr };

            /**
             * @brief Converts approximation method enum to human-readable string.
             *
             * @param method The approximation method to convert
             * @return String representation of the approximation method
             */
            static std::string approximationMethodToString( GeluConfig::ApproximationMethod method ) {
                switch (method) {
                case GeluConfig::ApproximationMethod::Exact:
                    return "Exact";
                case GeluConfig::ApproximationMethod::Tanh:
                    return "Tanh";
                case GeluConfig::ApproximationMethod::Sigmoid:
                    return "Sigmoid";
                default:
                    return "Unknown";
                }
            }

            /**
             * @brief Initializes the appropriate GELU operation implementation.
             *
             * Creates the device-specific operation implementation based on the template
             * parameter TDeviceType and registers it with the operation registry.
             * Uses abstract TensorDataType for type-safe operation creation.
             *
             * The operation choice is determined at compile-time via constexpr branching.
             *
             * @throws std::runtime_error If operation creation fails or operation is not registered
             */
            void createOperation() {
                if constexpr (TDeviceType == DeviceType::Cpu) {
                    operation_ = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu>(
                        "Cpu::GeluOp",
                        this->getDeviceContext(),
                        config_
                    );
                }
                else {
                    operation_ = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda>(
                        "Cuda::GeluOp",
                        this->getDeviceContext(),
                        config_
                    );
                }
            }
    };

    /**
     * @brief Type alias for CPU-specific GELU module.
     *
     * Convenience type that pre-configures the Gelu template for CPU execution.
     *
     * @tparam TDataType Abstract tensor data type (default: FP32)
     *
     * Example:
     * @code
     * GeluConfig config("cpu_gelu", GeluConfig::ApproximationMethod::Tanh);
     * auto gelu = CpuGelu<TensorDataType::FP32>("CPU", config);
     * @endcode
     */
    /*export template<TensorDataType TDataType = TensorDataType::FP32>
        using CpuGelu = Gelu<DeviceType::Cpu, TDataType>;*/

    /**
     * @brief Type alias for CUDA-specific GELU module.
     *
     * Convenience type that pre-configures the Gelu template for CUDA GPU execution.
     *
     * @tparam TDataType Abstract tensor data type (default: FP32)
     *
     * Example:
     * @code
     * GeluConfig config("cuda_gelu", GeluConfig::ApproximationMethod::Tanh);
     * auto gelu = CudaGelu<TensorDataType::FP16>("CUDA:0", config);
     * @endcode
     */
    /*export template<TensorDataType TDataType = TensorDataType::FP32>
        using CudaGelu = Gelu<DeviceType::Cuda, TDataType>;*/
}