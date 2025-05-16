module;
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

export module Dnn.Blocks.MLP;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.CompositeModule;
import Compute.Precision;
import Compute.MemoryResource;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.CpuDevice;
import Compute.OperationRegistry;
import Dnn.Modules.Linear;
import Dnn.Modules.Gelu;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Multi-Layer Perceptron (MLP) block for neural networks.
     *
     * This module implements a two-layer MLP with a GELU activation in between:
     * input -> Linear -> GELU -> Linear -> output
     *
     * This is commonly used in transformer architectures as the feed-forward network.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TOutput The data type of the output tensor elements, defaults to TInput.
     * @tparam TPrecision The data type used for internal calculations, defaults to TOutput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda,
        typename TInput = float,
        typename TOutput = TInput,
        typename TPrecision = TOutput>
        requires ValidFloatTensorTypes<TInput, TOutput>&& ValidPrecisionType<TPrecision>
    class MLP : public CompositeModule<TDeviceType, TInput, TOutput, TPrecision> {
    public:
        using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::CudaMemoryResource, Compute::CpuMemoryResource>;
        using CompositeModuleBase = CompositeModule<TDeviceType, TInput, TOutput, TPrecision>; ///< Base class type for the module

        /**
         * @brief Constructs a new MLP module with the default device context.
         *
         * @param name The name of the module for identification purposes.
         * @param device_name The name of the device to use for this module.
         * @param input_shape The shape of the input tensor.
         * @param output_channels The number of output channels for the intermediate layer.
         * @param has_bias Whether to include bias terms in linear transformations. Default is true.
         * @param is_training Whether the module is initially in training mode. Default is false.
         */
        MLP( std::string name, const std::string& device_name, const std::vector<size_t>& input_shape, size_t output_channels,
            bool has_bias = true, bool is_training = false )
            : CompositeModuleBase( device_name ), input_shape_{ input_shape }, output_channels_{ output_channels } {
            this->setName( name );
            this->setTraining( is_training );

            // Infer the number of input channels from the input shape
            input_channels_ = input_shape.back();

            initializeModules( has_bias );
        }

        /**
         * @brief Constructs a new MLP module with a specific device context.
         *
         * @param name The name of the module for identification purposes.
         * @param context The device context to use for this module.
         * @param input_shape The shape of the input tensor.
         * @param output_channels The number of output channels for the intermediate layer.
         * @param has_bias Whether to include bias terms in linear transformations. Default is true.
         * @param is_training Whether the module is initially in training mode. Default is false.
         */
        MLP( std::string name, std::shared_ptr<DeviceContext> context, const std::vector<size_t>& input_shape, size_t output_channels,
            bool has_bias = true, bool is_training = false )
            : CompositeModuleBase( context ), input_shape_{ input_shape }, output_channels_{ output_channels } {
            this->setName( name );
            this->setTraining( is_training );

            // Infer the number of input channels from the input shape
            input_channels_ = input_shape.back();

            initializeModules( has_bias );
        }

        /**
         * @brief Performs the forward pass of the MLP block.
         *
         * Processes the input through the sequence: fc1 -> gelu -> fc_proj
         * May use fused operations for better performance during inference.
         *
         * @param input The input tensor to be processed.
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TPrecision, MR>& input, Tensor<TPrecision, MR>& output ) {
            // Check if we're in training mode where we need individual operations for backprop
            if ( this->isTraining() ) {
                // Training mode: use the individual operations
                fc_1_->forward( input, fc_1_output_ );
                gelu_->forward( fc_1_output_, gelu_output_ );
                fc_proj_->forward( gelu_output_, output );

                return;
            }

            // Inference mode: try to use fused operations
            // Note: This is a simplified version, as the fused operation lookup needs to be adapted
            // to work with the new device context approach

            // For now, just use the individual operations (can be optimized later)
            fc_1_->forward( input, fc_1_output_ );
            gelu_->forward( fc_1_output_, gelu_output_ );
            fc_proj_->forward( gelu_output_, output );
        }

        /**
         * @brief Performs the backward pass of the MLP block.
         *
         * Computes gradients for the input tensor based on the output gradients.
         *
         * @param input The input tensor to be processed.
         * @param output The output tensor where the results will be stored.
         */
        void backward( const Tensor<TPrecision, MR>& input, const Tensor<TPrecision, MR>& output ) {
            // Check if we're in training mode where we need individual operations for backprop
            if ( this->isTraining() ) {
                // Training mode: use the individual operations
                fc_proj_->backward( gelu_output_, output );
                gelu_->backward( fc_1_output_, gelu_output_ );
                fc_1_->backward( input, fc_1_output_ );
                return;
            }
            // Inference mode: try to use fused operations
            // Note: This is a simplified version, as the fused operation lookup needs to be adapted
            // to work with the new device context approach
            // For now, just use the individual operations (can be optimized later)
            fc_proj_->backward( gelu_output_, output );
            gelu_->backward( fc_1_output_, gelu_output_ );
            fc_1_->backward( input, fc_1_output_ );
        }

        /**
         * @brief Gets the number of trainable parameters in this module.
         *
         * Counts the total number of parameters in all sub-modules.
         *
         * @return size_t The total number of parameters.
         */
        size_t parameterCount() const override {
            size_t total_parameters = 0;
            for ( const auto& module : this->getModules() ) {
                total_parameters += module->parameterCount();
            }
            return total_parameters;
        }

        /**
         * @brief Saves the module state to a ZIP archive.
         *
         * Serializes all sub-module states to the provided archive.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( mz_zip_archive& zip ) const override {
            for ( const auto& module : this->getModules() ) {
                module->save( zip );
            }
        }

        /**
         * @brief Loads the module state from a ZIP archive.
         *
         * Deserializes all sub-module states from the provided archive.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( mz_zip_archive& zip ) override {
            for ( const auto& module : this->getModules() ) {
                module->load( zip );
            }
        }

        /**
         * @brief Converts the module information to a human-readable string.
         *
         * @return std::string A string representation of the module information.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "====================" << std::endl;
            oss << "MLP: " << this->getName();
            oss << ", Input shape: (";
            for ( size_t i = 0; i < input_shape_.size(); ++i ) {
                oss << input_shape_[ i ];
                if ( i != input_shape_.size() - 1 ) {
                    oss << ",";
                }
            }
            oss << ")";
            oss << ", Input channels: " << input_channels_;
            oss << ", Output channels: " << output_channels_;
            oss << ", Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;
            oss << "Sub-Modules..." << std::endl;
            for ( const auto& [name, module] : this->getNamedModules() ) {
                oss << *module;
            }

            return oss.str();
        }

    private:
        std::vector<size_t> input_shape_; ///< The input shape.
        size_t input_channels_; ///< The number of input channels
        size_t output_channels_; ///< The number of output channels

        std::shared_ptr<Linear<TDeviceType, TInput, TOutput, TPrecision>> fc_1_{ nullptr };
        std::shared_ptr<Gelu<TDeviceType, TInput, TOutput, TPrecision>> gelu_{ nullptr };
        std::shared_ptr<Linear<TDeviceType, TInput, TOutput, TPrecision>> fc_proj_{ nullptr };

        Tensor<TPrecision, MR> fc_1_output_;
        Tensor<TPrecision, MR> gelu_output_;

        /**
         * @brief Initializes the sub-modules and output tensors for the MLP block.
         *
         * @param has_bias Whether to include bias terms in linear transformations.
         */
        void initializeModules( bool has_bias ) {
            // Clear existing modules if any
            for ( const auto& [name, _] : this->getNamedModules() ) {
                this->removeModule( name );
            }

            // Create new modules with the current device context
            fc_1_ = std::make_shared<Linear<TDeviceType, TInput, TOutput, TPrecision>>(
                this->getName() + ".fc_1", this->getDeviceContext(), input_channels_, output_channels_, has_bias );

            gelu_ = std::make_shared<Gelu<TDeviceType, TInput, TOutput, TPrecision>>(
                this->getName() + ".gelu", this->getDeviceContext() );

            fc_proj_ = std::make_shared<Linear<TDeviceType, TInput, TOutput, TPrecision>>(
                this->getName() + ".fc_proj", this->getDeviceContext(), output_channels_, input_channels_, has_bias );

            // Add sub-modules to the MLP block
            this->addModule( "fc_1", fc_1_ );
            this->addModule( "gelu", gelu_ );
            this->addModule( "fc_proj", fc_proj_ );

            // Construct the output shape for the fc_1_ layer
            std::vector<size_t> fc_1_output_shape = input_shape_;
            fc_1_output_shape.back() = output_channels_;

            // Create output tensors for the intermediate steps
            fc_1_output_ = Tensor<TPrecision, MR>( fc_1_output_shape );
            gelu_output_ = Tensor<TPrecision, MR>( fc_1_output_shape );
        }
    };

    /**
     * @brief Type alias for CPU-based MLP module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     * @tparam TPrecision Data type used for internal calculations, defaults to TOutput.
     */
    export template<typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
        using CpuMLP = MLP<DeviceType::Cpu, TInput, TOutput, TPrecision>;

    /**
     * @brief Type alias for CUDA-based MLP module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     * @tparam TPrecision Data type used for internal calculations, defaults to TOutput.
     */
    export template<typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
        using CudaMLP = MLP<DeviceType::Cuda, TInput, TOutput, TPrecision>;
}