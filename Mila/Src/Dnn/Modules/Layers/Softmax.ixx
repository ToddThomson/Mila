/**
 * @file Softmax.ixx
 * @brief Implementation of the Softmax activation function module.
 */

module;
#include <miniz.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <cstdint>

export module Dnn.Modules.Softmax;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;
import Compute.Precision;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.ComputeDevice;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationAttributes;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Softmax module for neural networks.
     *
     * This class implements the softmax function, which is often used in the final layer of a neural network
     * to convert raw scores into probabilities. The softmax operation normalizes the input values by applying:
     *
     * softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
     *
     * where the sum is computed over the specified axis. This normalization ensures all values sum to 1,
     * allowing them to be interpreted as probabilities for classification tasks.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TOutput The data type of the output tensor elements, defaults to TInput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidFloatTensorTypes<TInput, TOutput>
    class Softmax : public Module<TDeviceType, TInput, TOutput> {
    public:
        /**
         * @brief Memory resource type used for tensors, selected based on device type.
         */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;

        /**
         * @brief Alias for base module type.
         */
        using ModuleBase = Module<TDeviceType, TInput, TOutput>;

        /**
         * @brief Constructs a new Softmax module with a device name.
         *
         * Creates a new DeviceContext internally using the provided device name.
         * This constructor is useful for creating standalone modules without
         * pre-existing device contexts.
         *
         * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
         * @param config Configuration parameters for the Softmax module.
         * @throws std::invalid_argument If the device name is invalid or the configuration is invalid
         * @throws std::runtime_error If device type doesn't match template parameter TDeviceType
         */
        explicit Softmax( const std::string& device_name, const SoftmaxConfig& config )
            : ModuleBase( std::make_shared<DeviceContext>( device_name ), config ), config_( config ) {

            config.validate();

            createOperation();
        }

        /**
         * @brief Constructs a new Softmax module with a provided device context.
         *
         * Uses a pre-existing DeviceContext instance. This constructor is useful when integrating
         * the module into a larger network that shares device contexts across modules.
         *
         * @param device_context The device context to use for this module.
         * @param config Configuration parameters for the Softmax module.
         * @throws std::invalid_argument If device_context is null or configuration is invalid
         * @throws std::runtime_error If device context type doesn't match template parameter TDeviceType
         */
        explicit Softmax( std::shared_ptr<DeviceContext> device_context, const SoftmaxConfig& config )
            : ModuleBase( device_context, config ), config_( config ) {

            config.validate();

            createOperation();
        }

        /**
         * @brief Gets the number of trainable parameters in this module.
         *
         * The Softmax module has no trainable parameters as it's a fixed mathematical operation.
         *
         * @return size_t Always returns 0 as Softmax has no parameters.
         */
        size_t parameterCount() const override {
            return 0;
        }

        /**
         * @brief Performs the forward pass of the softmax operation.
         *
         * Computes the softmax of the input tensor along the specified axis and writes the result to the output tensor.
         * The operation exponentiates each element and then normalizes by the sum of all exponentiated values
         * along the specified axis.
         *
         * @param input The input tensor to apply softmax to.
         * @param output The tensor where softmax results will be stored.
         */
        void forward( const Tensor<TInput, MR>& input, Tensor<TOutput, MR>& output ) {
            operation_->forward( input, parameters_, attributes_, output, output_state_ );
        }

        /**
         * @brief Performs the backward pass of the Softmax operation.
         *
         * Computes the gradient of the softmax function with respect to its inputs.
         * The gradient of softmax is more complex than most activations because
         * each output depends on all inputs in the same dimension.
         *
         * @param input The input tensor from the forward pass.
         * @param output_grad The gradient of loss with respect to the output.
         * @param input_grad The tensor to store gradients with respect to input.
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TOutput, MR>& output_grad,
            Tensor<TInput, MR>& input_grad ) {

            operation_->backward(
                input,           // Input tensor
                output_grad,     // Gradient from next layer
                parameters_,     // Empty for Softmax
                {},              // No parameter gradients for Softmax
                input_grad,      // Gradient to propagate to previous layer
                attributes_,     // Operation properties
                output_state_    // Cached tensors from forward pass
            );
        }

        /**
         * @brief Gets the axis used for softmax computation.
         *
         * @return int64_t The axis along which softmax is applied.
         */
        int64_t getAxis() const {
            return config_.getAxis();
        }

        /**
         * @brief Serializes the module state to a ZIP archive.
         *
         * Implementation of the Module interface for serialization. Since Softmax has no
         * learnable parameters, this is a no-op implementation.
         *
         * @param zip ZIP archive for serialization
         */
        void save( mz_zip_archive& zip ) const override {
            // No-op: Softmax is a stateless activation function with no parameters to persist
        }

        /**
         * @brief Deserializes the module state from a ZIP archive.
         *
         * Implementation of the Module interface for deserialization. Since Softmax has no
         * learnable parameters, this is a no-op implementation.
         *
         * @param zip ZIP archive for deserialization
         */
        void load( mz_zip_archive& zip ) override {
            // No-op: Softmax is a stateless activation function with no parameters to load
        }

        /**
         * @brief Generates a string representation of this module's configuration.
         *
         * @return std::string A formatted string with module name, axis, device, and precision info
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Softmax: " << this->getName() << std::endl;
            oss << "Dimension: " << config_.getAxis() << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;

            return oss.str();
        }

    private:
        /**
         * @brief Configuration for the Softmax module.
         */
        SoftmaxConfig config_;

        /**
         * @brief Collection of parameters for this module (empty for Softmax).
         */
        std::vector<std::shared_ptr<Tensor<TInput, MR>>> parameters_;

        /**
         * @brief Collection of output state tensors for caching.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> output_state_;

        /**
         * @brief Operation attributes and configuration.
         */
        OperationAttributes attributes_;

        /**
         * @brief The operation that implements the softmax calculation.
         */
        std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput>> operation_{ nullptr };

        /**
         * @brief Creates the appropriate softmax operation for the current device.
         *
         * Instantiates either a CPU or CUDA softmax operation based on the device type.
         * Sets the axis attribute needed by the operation to properly apply softmax
         * along the specified dimension.
         */
        void createOperation() {
            // Set the axis attribute for the operation
            attributes_.set( "axis", config_.getAxis() );

            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TInput, TOutput>(
                    "Cpu::SoftmaxOp",
                    this->getDeviceContext(),
                    config_ );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TInput, TOutput>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TInput, TOutput>(
                    "Cuda::SoftmaxOp",
                    this->getDeviceContext(),
                    config_ );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TInput, TOutput>>(base_op);
            }
        }
    };

    /**
     * @brief Type alias for CPU-based softmax module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CpuSoftmax = Softmax<DeviceType::Cpu, TInput, TOutput>;

    /**
     * @brief Type alias for CUDA-based softmax module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CudaSoftmax = Softmax<DeviceType::Cuda, TInput, TOutput>;
}