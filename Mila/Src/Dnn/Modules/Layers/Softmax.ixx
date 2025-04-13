/**
 * @file Softmax.ixx
 * @brief Implementation of the Softmax activation function module.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <cstdint>

export module Dnn.Modules.Softmax;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;
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

export namespace Mila::Dnn
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
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TPrecision The data type used for internal precision calculations, defaults to TInput.
     */
    export
        template<typename TInput, typename TPrecision = TInput>
        requires ValidTensorTypes<TInput, TPrecision>
    class Softmax final : public Module<TInput, TPrecision> {
    public:
        /**
         * @brief Construct a new Softmax module with the default device context.
         *
         * @param name The name identifier for the module.
         * @param axis The dimension along which to apply the softmax operation. Default is -1 (last dimension).
         * @param is_training Whether the module is in training mode. Default is false.
         */
        Softmax( std::string name, int64_t axis = -1, bool is_training = false )
            : Module<TInput, TPrecision>(),
            axis_{ axis } {
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
         */
        Softmax( std::string name, std::shared_ptr<DeviceContext> context, int64_t axis = -1, bool is_training = false )
            : Module<TInput, TPrecision>( context ),
            axis_{ axis } {
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
        template<typename TMR>
        void forward( const Tensor<TInput, TMR>& input, Tensor<TPrecision, TMR>& output ) {
            operation_->forward( input, parameters_, attributes_, output, output_state_ );
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
            oss << "Softmax: " << this->getName() << ", Dimension: " << axis_;
            oss << ", Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;

            return oss.str();
        }

    protected:
        /**
         * @brief Called when the device context changes.
         *
         * Recreates operations for the new device.
         */
        void onDeviceChanged() override {
            createOperation();
        }

    private:
        /**
         * @brief The dimension to perform the softmax operation on.
         *
         * Default is -1 for the last dimension. Negative values count backward from the end
         * (-1 means the last dimension, -2 means the second-to-last dimension, etc.).
         */
        int64_t axis_{ -1 };

        /**
         * @brief The parameters for the operation.
         *
         * The Softmax activation has no parameters, so this is an empty vector.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, typename Module<TInput, TPrecision>::MR>>> parameters_;

        /**
         * @brief The output state.
         *
         * Storage for intermediate results that might be needed for the backward pass.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, typename Module<TInput, TPrecision>::MR>>> output_state_;

        /**
         * @brief The operation attributes.
         *
         * Contains the configuration for the softmax operation, such as the axis along which to compute.
         */
        OperationAttributes attributes_;

        /**
         * @brief The underlying unary operation that implements the softmax function.
         */
        std::shared_ptr<UnaryOperation<TInput, TPrecision>> operation_{ nullptr };

        /**
         * @brief Create the appropriate softmax operation based on the current device context.
         *
         * This method initializes the operation_ member with the appropriate implementation
         * of the softmax operation for either CPU or CUDA, based on the current device context.
         */
        void createOperation() {
            // Set the axis in attributes
            attributes_.setInt64( "axis", axis_ );

            // Get the device type from the context
            auto device_type = this->getDeviceContext()->getDevice()->getDeviceType();

            if ( operation_ ) {
                operation_.reset(); // Clear existing operation
            }

            if ( device_type == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cpu>( "Cpu::SoftmaxOp" );
                operation_ = std::static_pointer_cast<UnaryOperation<TInput, TPrecision>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cuda>( "Cuda::SoftmaxOp" );
                operation_ = std::static_pointer_cast<UnaryOperation<TInput, TPrecision>>(base_op);
            }
        }
    };
}
