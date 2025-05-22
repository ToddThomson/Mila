/**
 * @file Gelu.ixx
 * @brief Implementation of the Gaussian Error Linear Unit (GELU) activation function module.
 */

module;
#include <miniz.h>
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

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Gaussian Error Linear Unit (GELU) activation function module.
     *
     * GELU is an activation function defined as:
     * GELU(x) = x * phi(x)
     * where phi(x) is the standard Gaussian cumulative distribution function.
     *
     * This activation function is used in many state-of-the-art neural network
     * architectures, including transformers, as an alternative to ReLU.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which the module will operate.
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TOutput The data type of the output tensor elements, defaults to TInput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidFloatTensorTypes<TInput, TOutput>
    class Gelu : public Module<TDeviceType, TInput, TOutput> {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>; ///< Memory resource type based on device type
        using ModuleBase = Module<TDeviceType, TInput, TOutput>; ///< Base class type for the module

        /**
         * @brief Construct a new Gelu module from configuration.
         *
         * @param config The configuration for this module
         */
        explicit Gelu( const GeluConfig& config )
            : ModuleBase(
                config.getContext() ? config.getContext() : std::make_shared<DeviceContext>( config.getDeviceName() ),
                TDeviceType == DeviceType::Cpu ? ComputePrecision::Policy::Disabled : config.getPrecision() ),
            approximation_method_( config.getApproximationMethod() ) {

            config.validate();

            this->setName( config.getName() );
            this->setTraining( config.isTraining() );

            createOperation();
        }

        /**
         * @brief Gets the number of trainable parameters in this module.
         *
         * The GELU activation function has no trainable parameters.
         *
         * @return size_t Always returns 0 for GELU.
         */
        size_t parameterCount() const override {
            return 0;
        }

        /**
         * @brief Performs the forward pass of the GELU activation function.
         *
         * Applies the GELU activation function element-wise to the input tensor.
         *
         * @param input The input tensor to apply the activation function to.
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TInput, MR>& input, Tensor<TOutput, MR>& output ) {
            operation_->forward( input, parameters_, properties_, output, output_state_ );
        }

        /**
         * @brief Performs the backward pass of the GELU activation function.
         *
         * Computes the gradient of the GELU function with respect to its input.
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
                parameters_,     // Empty for GELU
                {},              // No parameter gradients for GELU
                input_grad,      // Gradient to propagate to previous layer
                properties_,     // Operation properties
                output_state_    // Cached tensors from forward pass
            );
        }

        /**
         * @brief Gets the approximation method used by this GELU module.
         *
         * @return GeluConfig::ApproximationMethod The method used to approximate the GELU function.
         */
        GeluConfig::ApproximationMethod getApproximationMethod() const {
            return approximation_method_;
        }

        /**
         * @brief Saves the module state to a ZIP archive.
         *
         * Since GELU has no trainable parameters, this function is mostly a placeholder
         * for the module interface.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( mz_zip_archive& zip ) const override {
            // GELU has no parameters to save
        }

        /**
         * @brief Loads the module state from a ZIP archive.
         *
         * Since GELU has no trainable parameters, this function is mostly a placeholder
         * for the module interface.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( mz_zip_archive& zip ) override {
            // GELU has no parameters to load
        }

        /**
         * @brief Converts the module information to a human-readable string.
         *
         * @return std::string A string representation of the module information.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Gelu: " << this->getName() << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << "Approximation Method: " << approximationMethodToString( approximation_method_ ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;

            return oss.str();
        }

    private:
        /**
         * @brief The approximation method used for computing GELU.
         */
        GeluConfig::ApproximationMethod approximation_method_ = GeluConfig::ApproximationMethod::Tanh;

        /**
         * @brief The parameters for the operation.
         *
         * The GELU activation has no parameters, so this is an empty vector.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameters_;

        /**
         * @brief The output cache.
         *
         * Storage for intermediate results that might be needed for the backward pass.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> output_state_;

        /**
         * @brief The operation properties.
         *
         * Additional attributes that might be needed for the operation.
         */
        OperationAttributes properties_;

        /**
         * @brief The underlying unary operation that implements the GELU function.
         *
         * This pointer will be updated based on the current device context.
         */
        std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput>> operation_{ nullptr };

        /**
         * @brief Converts the approximation method enum to a string representation.
         *
         * @param method The approximation method to convert
         * @return std::string String representation of the method
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
         * @brief Creates the appropriate GELU operation based on the current device context.
         *
         * This method initializes the operation_ member with the appropriate implementation
         * of the GELU operation for either CPU or CUDA, based on the current device context.
         * It also passes the compute precision policy to the operation.
         */
        void createOperation() {
            // Set approximation method in operation properties
            properties_[ "approximation_method" ] = static_cast<int>(approximation_method_);

            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TInput, TOutput>(
                    "Cpu::GeluOp", this->getDeviceContext(), ComputePrecision::Policy::Disabled );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TInput, TOutput>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TInput, TOutput>(
                    "Cuda::GeluOp", this->getDeviceContext(), this->getComputePrecision().getPolicy() );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TInput, TOutput>>(base_op);
            }
        }
    };

    // Convenient type aliases for common use cases
    export template<typename T = float>
        using CpuGelu = Gelu<DeviceType::Cpu, T, T>;

    export template<typename T = float>
        using CudaGelu = Gelu<DeviceType::Cuda, T, T>;
}