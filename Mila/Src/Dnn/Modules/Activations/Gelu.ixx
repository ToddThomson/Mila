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
import Dnn.TensorTraits;
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
import Compute.CudaMemoryResource;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
	using namespace Mila::Dnn::Serialization;

    /**
     * @brief Gaussian Error Linear Unit (GELU) activation function module.
     *
     * GELU is defined mathematically as:
     * GELU(x) = x * phi(x)
     *
     * Where phi(x) is the cumulative distribution function of the standard normal distribution.
     *
     * Three approximation methods are supported (configured via GeluConfig):
     * 1. Exact: Uses the error function - most accurate but computationally expensive
     * 2. Tanh: Fast approximation using tanh - GELU(x) ? 0.5x(1 + tanh(?(2/?)(x + 0.044715x³)))
     * 3. Sigmoid: Fast approximation using sigmoid - GELU(x) ? x * sigmoid(1.702x)
     *
     * Note: Currently only the Tanh approximation is fully supported in the implementation.
     *
     * @tparam TDeviceType Computing device type (CPU or CUDA)
     * @tparam TDataType Floating-point data type for computations (e.g., float, half )
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TDataType = float>
        requires ValidFloatTensorType<TDataType>
    class Gelu : public Module<TDeviceType, TDataType, TDataType> {
    public:
        /**
         * @brief Memory resource type determined based on device type.
         *
         * Automatically selects appropriate memory resource (CPU or CUDA) based on TDeviceType.
         */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;

        /**
         * @brief Alias for base module type.
         */
        using ModuleBase = Module<TDeviceType, TDataType, TDataType>;

        /**
         * @brief Constructs a Gelu module using device name and configuration.
         *
         * Creates a new DeviceContext internally using the provided device name.
         * This constructor is useful for creating standalone modules without
         * pre-existing device contexts.
         *
         * @param device_name Device identifier string (e.g., "cpu", "cuda:0")
         * @param config Configuration parameters for the GELU module
         * @throws std::invalid_argument If the device name is invalid or the configuration is invalid
         * @throws std::runtime_error If device type doesn't match template parameter TDeviceType
         */
        explicit Gelu( const std::string& device_name, const GeluConfig& config )
            : ModuleBase( std::make_shared<DeviceContext>( device_name ), config ), config_( config ) {

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
         * @param input Input tensor to transform
         * @param output Tensor where results will be stored (must be pre-allocated with matching dimensions)
         */
        void forward( const Tensor<TDataType, MR>& input, Tensor<TDataType, MR>& output ) {
            operation_->forward( input, parameters_, properties_, output, output_state_ );
        }

        /**
         * @brief Performs backward propagation, computing gradients for GELU activation.
         *
         * Computes the gradient of the GELU function with respect to its inputs,
         * which is needed for training via backpropagation.
         *
         * The GELU derivative is:
         * d/dx GELU(x) = ?(x) + x * ?'(x)
         *
         * Where ?'(x) is the derivative of the CDF (the PDF of the standard normal distribution).
         *
         * @param input Original input tensor from the forward pass
         * @param output_grad Gradient tensor from the next layer (?L/?output)
         * @param input_grad Output tensor to store the computed gradients (?L/?input)
         */
        void backward(
            const Tensor<TDataType, MR>& input,
            const Tensor<TDataType, MR>& output_grad,
            Tensor<TDataType, MR>& input_grad ) {
            operation_->backward(
                input,           // Input tensor
                output_grad,     // Gradient from next layer
                parameters_,     // Empty for GELU
                {},              // No parameter gradients for GELU
                input_grad,      // Gradient to propagate to previous layer
                properties_,     // Operation properties
                output_state_    // Cached tensors from forward pass
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
         * @brief Serializes module state to a ZIP archive.
         *
         * Implementation of the Module interface for serialization. Since GELU has no
         * learnable parameters, this is a no-op implementation.
         *
         * @param zip ZIP archive for serialization
         */
        void save( ModelArchive& zip ) const override {
            // No-op: GELU is a stateless activation function with no parameters to persist
        }

        /**
         * @brief Deserializes module state from a ZIP archive.
         *
         * Implementation of the Module interface for deserialization. Since GELU has no
         * learnable parameters, this is a no-op implementation.
         *
         * @param zip ZIP archive for deserialization
         */
        void load( ModelArchive& archive ) override {
            // No-op: GELU is a stateless activation function with no parameters to load
        }

        /**
         * @brief Generates a string representation of this module's configuration.
         *
         * @return Formatted string with module name, device information, and approximation method
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Gelu: " << this->getName() << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << "Approximation Method: " << approximationMethodToString( config_.getApproximationMethod() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;

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
        std::vector<std::shared_ptr<Tensor<TDataType, MR>>> parameters_;

        /**
         * @brief Output state cache for backward propagation.
         *
         * Stores intermediate results from the forward pass that may be needed
         * during backward propagation to efficiently compute gradients.
         */
        std::vector<std::shared_ptr<Tensor<TDataType, MR>>> output_state_;

        /**
         * @brief Additional attributes for operation customization.
         *
         * Holds configuration values that might be needed by specific implementations
         * of the GELU operation.
         */
		 // TODO: Remove as GeluConfig is now used for this purpose
        OperationAttributes properties_;

        /**
         * @brief The underlying computational operation that implements GELU.
         *
         * This pointer is initialized based on the device type and configuration,
         * providing the device-specific implementation of the GELU function.
         */
        std::shared_ptr<UnaryOperation<TDeviceType, TDataType, TDataType>> operation_{ nullptr };

        /**
         * @brief Converts approximation method enum to human-readable string.
         *
         * @param method The approximation method to convert
         * @return String representation of the approximation method
         */
        static std::string approximationMethodToString( GeluConfig::ApproximationMethod method ) {
            switch ( method ) {
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
         *
         * The operation choice is determined at compile-time via constexpr branching.
         */
        void createOperation() {
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TDataType, TDataType>(
                    "Cpu::GeluOp", this->getDeviceContext(), config_ );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TDataType, TDataType>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TDataType, TDataType>(
                    "Cuda::GeluOp", this->getDeviceContext(), config_ );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TDataType, TDataType>>(base_op);
            }
        }
    };

    /**
     * @brief Type alias for CPU-specific GELU module.
     *
     * Convenience type that pre-configures the Gelu template for CPU execution.
     *
     * @tparam TDataType Floating-point data type (default: float)
     */
    export template<typename TDataType = float>
        using CpuGelu = Gelu<DeviceType::Cpu, TDataType>;

    /**
     * @brief Type alias for CUDA-specific GELU module.
     *
     * Convenience type that pre-configures the Gelu template for CUDA GPU execution.
     *
     * @tparam TDataType Floating-point data type (default: float)
     */
    export template<typename TDataType = float>
        using CudaGelu = Gelu<DeviceType::Cuda, TDataType>;
}