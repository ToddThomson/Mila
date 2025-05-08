/**
 * @file Residual.ixx
 * @brief Implementation of the Residual connection module for neural networks.
 */

module;
#include <sstream>
#include <vector>
#include <string>
#include <memory>

#include <type_traits>

export module Dnn.Modules.Residual;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationAttributes;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Residual connection module for neural networks.
     *
     * The Residual module implements a skip connection that adds the input tensor
     * to the output of some function, helping mitigate the vanishing gradient problem
     * in deep neural networks. This is a key component in many modern architectures
     * including ResNet and Transformers.
     *
     * The operation performed is: output = input_a + input_b
     *
     * @tparam TPrecision The data type of the input tensor elements.
     * @tparam TDataType The data type used for computation and output (defaults to the input type).
     */
    export
        template<typename TPrecision, DeviceType TDeviceType = DeviceType::Cuda>
        requires ValidFloatTensorType<TPrecision>
    class Residual : public Module<TPrecision, TPrecision, TDeviceType> {
    public:
		using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;
		using ModuleBase = Module<TPrecision, TPrecision, TDeviceType>;

        /**
         * @brief Constructs a new Residual connection module with the default device context.
         *
         * @param name The name of the module for identification purposes.
         * @param is_training Whether the module is being used in training mode (defaults to false).
         */
        Residual( std::string name, const std::string& device_name, bool is_training = false )
            : ModuleBase( device_name ) {
            this->setTraining( is_training );
            this->setName( name );
            createOperation();
        }

        /**
         * @brief Constructs a new Residual connection module with a specific device context.
         *
         * @param name The name of the module for identification purposes.
         * @param context The device context to use for this module.
         * @param is_training Whether the module is being used in training mode (defaults to false).
         */
        Residual( std::string name, std::shared_ptr<DeviceContext> context, bool is_training = false )
            : ModuleBase( context ) {
            this->setTraining( is_training );
            this->setName( name );
            createOperation();
        }

        /**
         * @brief Gets the number of trainable parameters in this module.
         *
         * The Residual connection has no trainable parameters.
         *
         * @return size_t Always returns 0 for Residual connections.
         */
        size_t parameterCount() const override {
            return 0;
        }

        /**
         * @brief Performs the forward pass of the Residual connection.
         *
         * Adds the two input tensors element-wise and stores the result in the output tensor.
         *
         * @param input_a The first input tensor (typically the skip connection).
         * @param input_b The second input tensor (typically the function output).
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TPrecision, MR>& input_a, const Tensor<TPrecision, MR>& input_b, Tensor<TPrecision, MR>& output ) {
            operation_->forward( input_a, input_b, parameters_, attributes_, output, output_state_ );
        }

        /**
         * @brief Saves the module state to a ZIP archive.
         *
         * Since Residual has no trainable parameters, this function is mostly a placeholder
         * for the module interface.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( mz_zip_archive& zip ) const override {
            // Residual has no parameters to save
        }

        /**
         * @brief Loads the module state from a ZIP archive.
         *
         * Since Residual has no trainable parameters, this function is mostly a placeholder
         * for the module interface.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( mz_zip_archive& zip ) override {
            // Residual has no parameters to load
        }

        /**
         * @brief Converts the module information to a human-readable string.
         *
         * @return std::string A string representation of the module information.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Residual: " << this->getName();
            oss << ", Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;

            return oss.str();
        }

    //protected:
    //    /**
    //     * @brief Called when the device context changes.
    //     *
    //     * Recreates operations for the new device.
    //     */
    //    void onDeviceChanged() override {
    //        createOperation();
    //    }

    private:
        /**
         * @brief The parameters for the operation.
         *
         * The Residual connection has no parameters, so this is an empty vector.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, MR>>> parameters_;

        /**
         * @brief The output cache.
         *
         * Storage for intermediate results that might be needed for the backward pass.
         * Not used in this module.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, MR>>> output_state_;

        /**
         * @brief The operation attributes.
         *
         * Additional attributes that might be needed for the operation.
         */
        OperationAttributes attributes_;

        /**
         * @brief The underlying binary operation that implements the Residual connection.
         *
         * For residual connections, this operation performs element-wise addition
         * between the two input tensors.
         */
        std::shared_ptr<BinaryOperation<TPrecision, TPrecision, TPrecision, TDeviceType>> operation_{ nullptr };

        /**
         * @brief Creates the appropriate Residual operation based on the current device context.
         *
         * This method initializes the operation_ member with the appropriate implementation
         * of the Residual operation for either CPU or CUDA, based on the current device context.
         *
         * The operation performs element-wise addition of the two input tensors.
         */
        void createOperation() {
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                operation_ = std::static_pointer_cast<BinaryOperation<TPrecision, TPrecision, TPrecision, TDeviceType>>(
                    OperationRegistry::instance().createBinaryOperation<TPrecision, TPrecision, TPrecision, DeviceType::Cpu>(
                        "Cpu::ResidualOp",
                        this->getDeviceContext() )
                );
            }
            else {
                operation_ = std::static_pointer_cast<BinaryOperation<TPrecision, TPrecision, TPrecision, TDeviceType>>(
                    OperationRegistry::instance().createBinaryOperation<TPrecision, TPrecision, TPrecision, DeviceType::Cuda>(
                        "Cuda::ResidualOp",
                        this->getDeviceContext() )
                );
            }
        }
    };
}
