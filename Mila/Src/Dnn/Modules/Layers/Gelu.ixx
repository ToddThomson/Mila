/**
 * @file Gelu.ixx
 * @brief Implementation of the Gaussian Error Linear Unit (GELU) activation function module.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>

export module Dnn.Modules.Gelu;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
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
     * @tparam TPrecision The data type of the input tensor elements.
     * @tparam TDataType The data type used for computation and output (defaults to the input type).
     */
    export template< typename TPrecision, DeviceType TDeviceType = DeviceType::Cuda>
        requires ValidFloatTensorType<TPrecision> 
    class Gelu : public Module<TPrecision, TPrecision, TDeviceType> {
    public:
		using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>; ///< Memory resource type based on device type
		using ModuleBase = Module<TPrecision, TPrecision, TDeviceType>; ///< Base class type for the module
        
        /**
         * @brief Constructs a new Gelu activation module with the default device context.
         *
         * @param name The name of the module for identification purposes.
         * @param is_training Whether the module is being used in training mode (defaults to false).
         */
        Gelu( std::string name, std::string device_name, bool is_training = false )
            : ModuleBase( device_name )
        {
            this->setTraining( is_training );
            this->setName( name );
            createOperation();
        }

        /**
         * @brief Constructs a new Gelu activation module with a specific device context.
         *
         * @param name The name of the module for identification purposes.
         * @param context The device context to use for this module.
         * @param is_training Whether the module is being used in training mode (defaults to false).
         */
        Gelu( std::string name, std::shared_ptr<DeviceContext> context, bool is_training = false )
            : ModuleBase( context )
        {
            this->setTraining( is_training );
            this->setName( name );
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
        void forward( const Tensor<TPrecision, MR>& input, Tensor<TPrecision, MR>& output ) {
            operation_->forward( input, parameters_, properties_, output, output_state_ );
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

            return oss.str();
        }

    //protected:
    //    /**
    //     * @brief Called when the device context changes.
    //     *
    //     * Recreates the operation for the new device.
    //     */
    //    void onDeviceChanged() override {
    //        createOperation();
    //    }

    private:
        /**
         * @brief The parameters for the operation.
         *
         * The GELU activation has no parameters, so this is an empty vector.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, MR>>> parameters_;

        /**
         * @brief The output cache.
         *
         * Storage for intermediate results that might be needed for the backward pass.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, MR>>> output_state_;

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
        std::shared_ptr<UnaryOperation<TPrecision, TPrecision, TDeviceType>> operation_{ nullptr };

        /**
         * @brief Creates the appropriate GELU operation based on the current device context.
         *
         * This method initializes the operation_ member with the appropriate implementation
         * of the GELU operation for either CPU or CUDA, based on the current device context.
         */
        void createOperation() {
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<TPrecision, TPrecision, DeviceType::Cpu>(
                    "Cpu::GeluOp",
                    this->getDeviceContext() );
                
                operation_ = std::static_pointer_cast<UnaryOperation<TPrecision, TPrecision, TDeviceType>>(base_op);
            }
            else 
            {
                auto base_op = OperationRegistry::instance().createUnaryOperation<TPrecision, TPrecision, DeviceType::Cuda>(
                    "Cuda::GeluOp",
                    this->getDeviceContext() );
                
                operation_ = std::static_pointer_cast<UnaryOperation<TPrecision, TPrecision, TDeviceType>>(base_op);
            }
        }
    };
}
