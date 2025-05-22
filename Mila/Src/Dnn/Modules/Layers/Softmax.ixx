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
import Compute.CpuDevice;
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
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;
        using ModuleBase = Module<TDeviceType, TInput, TOutput>;

        /**
         * @brief Construct a new Softmax module from configuration.
         *
         * @param config The configuration for this module
         */
        explicit Softmax( const SoftmaxConfig& config )
            : ModuleBase(
                config.getContext() ? config.getContext() : std::make_shared<DeviceContext>( config.getDeviceName() ),
                TDeviceType == DeviceType::Cpu ? ComputePrecision::Policy::Disabled : config.getPrecision() ),
            axis_( config.getAxis() ) {

            config.validate();

            this->setName( config.getName() );
            this->setTraining( config.isTraining() );

            createOperation();
        }

        /**
         * @brief Construct a new Softmax module with the default device context.
         *
         * @param name The name identifier for the module.
         * @param device_name The name of the device to use for this module.
         * @param axis The dimension along which to apply the softmax operation. Default is -1 (last dimension).
         * @param is_training Whether the module is in training mode. Default is false.
         * @param precision The compute precision policy (CPU operations always use Disabled).
         */
        Softmax( std::string name, const std::string& device_name, int64_t axis = -1,
            bool is_training = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
            : ModuleBase( device_name, TDeviceType == DeviceType::Cpu ? ComputePrecision::Policy::Disabled : precision ),
            axis_( axis ) {
            this->setTraining( is_training );
            this->setName( name );

            createOperation();
        }

        /**
         * @brief Construct a new Softmax module with a specific device context.
         *
         * @param name The name identifier for the module.
         * @param context The device context to use for this module.
         * @param axis The dimension along which to apply the softmax operation. Default is -1 (last dimension).
         * @param is_training Whether the module is in training mode. Default is false.
         * @param precision The compute precision policy (CPU operations always use Disabled).
         */
        Softmax( std::string name, std::shared_ptr<DeviceContext> context, int64_t axis = -1,
            bool is_training = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
            : ModuleBase( context, TDeviceType == DeviceType::Cpu ? ComputePrecision::Policy::Disabled : precision ),
            axis_( axis ) {
            this->setTraining( is_training );
            this->setName( name );

            createOperation();
        }

        /**
         * @brief Get the number of parameters in the module.
         *
         * The Softmax module has no trainable parameters.
         *
         * @return size_t Always returns 0 as Softmax has no parameters.
         */
        size_t parameterCount() const override {
            return 0;
        }

        /**
         * @brief Perform the forward pass of the softmax operation.
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
         * Computes gradients for the input tensor based on the output gradients.
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
         * @brief Get the axis used for softmax computation.
         *
         * @return int64_t The axis along which softmax is applied
         */
        int64_t getAxis() const {
            return axis_;
        }

        /**
         * @brief Save the module's state to a zip archive.
         *
         * Serializes the module's state to the provided zip archive. Since Softmax has no trainable parameters,
         * this primarily preserves the module's configuration.
         *
         * @param zip The zip archive where the module state will be saved.
         */
        void save( mz_zip_archive& zip ) const override {
            // Softmax has no parameters to save
        }

        /**
         * @brief Load the module's state from a zip archive.
         *
         * Deserializes the module's state from the provided zip archive. Since Softmax has no trainable parameters,
         * this primarily restores the module's configuration.
         *
         * @param zip The zip archive from which to load the module state.
         */
        void load( mz_zip_archive& zip ) override {
            // Softmax has no parameters to load
        }

        /**
         * @brief Convert the module information to string.
         *
         * Creates a string representation of the Softmax module, including its name,
         * the dimension used for softmax computation, and the device.
         *
         * @return std::string A formatted string describing the module configuration.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Softmax: " << this->getName() << ", Dimension: " << axis_ << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;

            return oss.str();
        }

    private:
        int64_t axis_{ -1 };
        std::vector<std::shared_ptr<Tensor<TInput, MR>>> parameters_;
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> output_state_;
        OperationAttributes attributes_;
        std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput>> operation_{ nullptr };

        void createOperation() {
            attributes_.set( "axis", axis_ );

            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TInput, TOutput>(
                    "Cpu::SoftmaxOp",
                    this->getDeviceContext(),
                    ComputePrecision::Policy::Disabled ); // Always use Disabled for CPU

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TInput, TOutput>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TInput, TOutput>(
                    "Cuda::SoftmaxOp",
                    this->getDeviceContext(),
                    this->getComputePrecision().getPolicy() );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TInput, TOutput>>(base_op);
            }
        }
    };

    export template<typename TInput = float, typename TOutput = TInput>
        using CpuSoftmax = Softmax<DeviceType::Cpu, TInput, TOutput>;

    export template<typename TInput = float, typename TOutput = TInput>
        using CudaSoftmax = Softmax<DeviceType::Cuda, TInput, TOutput>;
}