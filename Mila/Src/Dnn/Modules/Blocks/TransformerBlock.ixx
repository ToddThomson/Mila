module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream> 
#include <stdexcept>
#include <iosfwd>

export module Dnn.Blocks.TransformerBlock;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.CpuDevice;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Dnn.Module;
import Dnn.CompositeModule;
import Dnn.Modules.LayerNorm;
import Dnn.Modules.Linear;
import Dnn.Modules.Attention;
import Dnn.Modules.Residual;
import Dnn.Blocks.MLP;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief TransformerBlock implements a standard transformer encoder block.
     *
     * The transformer block consists of:
     * - Multi-head self-attention mechanism with residual connection
     * - Feed-forward network (MLP) with residual connection
     * - Layer normalization before each sub-block
     *
     * This is the fundamental building block of transformer architectures like BERT and GPT.
     * Since the transformer is a feed-forward network where layer outputs connect to the next
     * layer's inputs, a single data type is used throughout the network.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TDataType The data type used for tensor elements throughout the network.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TDataType = float>
        requires ValidFloatTensorType<TDataType>
    class TransformerBlock : public CompositeModule<TDeviceType, TDataType> {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;
        using CompositeModuleBase = CompositeModule<TDeviceType, TDataType>; ///< Base class type for the module

        /**
         * @brief Constructs a new TransformerBlock module with the default device context.
         *
         * @param name The name of the module for identification purposes.
         * @param device_name The name of the device to use for this module.
         * @param input_shape The shape of the input tensor, must be of rank 3 [batch_size, sequence_length, channels].
         * @param num_heads The number of attention heads to use.
         * @param is_training Whether the module is initially in training mode. Default is false.
         * @param precision The compute precision policy to use (defaults to Auto).
         * @throws std::invalid_argument If the input shape doesn't have rank 3.
         */
        TransformerBlock( std::string name, const std::string& device_name, const std::vector<size_t>& input_shape,
            const size_t num_heads, bool is_training = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
            : CompositeModuleBase( device_name, precision ), input_shape_{ validate_shape( input_shape ) }, num_heads_{ num_heads } {
            this->setName( name );
            this->setTraining( is_training );

            initializeModules();
        }

        /**
         * @brief Constructs a new TransformerBlock module with a specific device context.
         *
         * @param name The name of the module for identification purposes.
         * @param context The device context to use for this module.
         * @param input_shape The shape of the input tensor, must be of rank 3 [batch_size, sequence_length, channels].
         * @param num_heads The number of attention heads to use.
         * @param is_training Whether the module is initially in training mode. Default is false.
         * @param precision The compute precision policy to use (defaults to Auto).
         * @throws std::invalid_argument If the input shape doesn't have rank 3.
         */
        TransformerBlock( std::string name, std::shared_ptr<DeviceContext> context,
            const std::vector<size_t>& input_shape, const size_t num_heads, bool is_training = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
            : CompositeModuleBase( context, precision ), input_shape_{ validate_shape( input_shape ) }, num_heads_{ num_heads } {
            this->setName( name );
            this->setTraining( is_training );

            initializeModules();
        }

        /**
         * @brief Performs the forward pass of the TransformerBlock.
         *
         * The forward pass consists of the following sequence:
         * 1. Layer normalization 1
         * 2. QKV projection
         * 3. Multi-head attention
         * 4. Attention output projection
         * 5. Residual connection 1 (add attention output to input)
         * 6. Layer normalization 2
         * 7. MLP block
         * 8. Residual connection 2 (add MLP output to previous residual output)
         *
         * @param input The input tensor to be processed.
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TDataType, MR>& input, Tensor<TDataType, MR>& output ) {
            /*ln_1_->forward(input, ln_1_output_);
            fc_qkv_->forward(ln_1_output_, fc_qkv_output_);
            attn_->forward(fc_qkv_output_, attn_output_);
            fc_attn_proj_->forward(attn_output_, fc_attn_proj_output_);
            res_1_->forward(input, fc_attn_proj_output_, res_1_output_);
            ln_2_->forward(res_1_output_, ln_2_output_);
            mlp_->forward(ln_2_output_, mlp_output_);
            res_2_->forward(res_1_output_, mlp_output_, output);*/
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
            // Save the state of the child modules
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
            oss << "Transformer: " << this->getName();
            oss << ", Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;
            oss << "Sub-Modules..." << std::endl;

            for ( const auto& [name, module] : this->getNamedModules() ) {
                oss << *module;
            }

            return oss.str();
        }

    private:
        std::vector<size_t> input_shape_; ///< The input shape.
        size_t num_heads_; ///< The number of attention heads.

        // Sub-modules
        std::shared_ptr<LayerNorm<TDeviceType, TDataType>> ln_1_{ nullptr };
        std::shared_ptr<Linear<TDeviceType, TDataType>> fc_qkv_{ nullptr };
        std::shared_ptr<MultiHeadAttention<TDeviceType, TDataType>> attn_{ nullptr };
        std::shared_ptr<Linear<TDeviceType, TDataType>> fc_attn_proj_{ nullptr };
        std::shared_ptr<Residual<TDeviceType, TDataType>> res_1_{ nullptr };
        std::shared_ptr<LayerNorm<TDeviceType, TDataType>> ln_2_{ nullptr };
        std::shared_ptr<MLP<TDeviceType, TDataType>> mlp_{ nullptr };
        std::shared_ptr<Residual<TDeviceType, TDataType>> res_2_{ nullptr };

        // Intermediate tensors
        Tensor<TDataType, MR> ln_1_output_;
        Tensor<TDataType, MR> fc_qkv_output_;
        Tensor<TDataType, MR> attn_output_;
        Tensor<TDataType, MR> fc_attn_proj_output_;
        Tensor<TDataType, MR> res_1_output_;
        Tensor<TDataType, MR> ln_2_output_;
        Tensor<TDataType, MR> mlp_output_;
        Tensor<TDataType, MR> res_2_output_;

        /**
         * @brief Validates the input shape for the transformer block.
         *
         * @param shape The shape to validate.
         * @return The validated shape.
         * @throws std::invalid_argument If the shape doesn't have rank 3.
         */
        std::vector<size_t> validate_shape( const std::vector<size_t>& shape ) {
            if ( shape.size() != 3 ) {
                throw std::invalid_argument( "The input shape must have rank of 3." );
            }

            return shape;
        }

        /**
         * @brief Initializes the sub-modules and output tensors for the transformer block.
         */
        void initializeModules() {
            for ( const auto& [name, _] : this->getNamedModules() ) {
                this->removeModule( name );
            }

            auto B = input_shape_[ 0 ]; // Batch size
            auto T = input_shape_[ 1 ]; // Sequence length
            auto C = input_shape_[ 2 ]; // Number of channels

            auto precision = this->getComputePrecision().getPolicy();

            // Create new modules with the current device context and precision policy
            ln_1_ = std::make_shared<LayerNorm<TDeviceType, TDataType>>(
                this->getName() + ".ln_1", this->getDeviceContext(), input_shape_, precision );

            fc_qkv_ = std::make_shared<Linear<TDeviceType, TDataType>>(
                this->getName() + ".fc_qkv", this->getDeviceContext(), C, 3 * C,
                true, this->isTraining(), precision );

            attn_ = std::make_shared<MultiHeadAttention<TDeviceType, TDataType>>(
                this->getName() + ".attn", this->getDeviceContext(), input_shape_, num_heads_,
                this->isTraining(), precision );

            fc_attn_proj_ = std::make_shared<Linear<TDeviceType, TDataType>>(
                this->getName() + ".fc_attn_proj", this->getDeviceContext(), C, C,
                true, this->isTraining(), precision );

            res_1_ = std::make_shared<Residual<TDeviceType, TDataType>>(
                this->getName() + ".res_1", this->getDeviceContext(), precision );

            ln_2_ = std::make_shared<LayerNorm<TDeviceType, TDataType>>(
                this->getName() + ".ln_2", this->getDeviceContext(), input_shape_, precision );

            mlp_ = std::make_shared<MLP<TDeviceType, TDataType>>(
                this->getName() + ".mlp", this->getDeviceContext(), input_shape_, 4 * C,
                true, this->isTraining(), precision );

            res_2_ = std::make_shared<Residual<TDeviceType, TDataType>>(
                this->getName() + ".res_2", this->getDeviceContext(), precision );

            // Add sub-modules to the TransformerBlock
            this->addModule( "ln_1", ln_1_ );
            this->addModule( "fc_qkv", fc_qkv_ );
            this->addModule( "attn", attn_ );
            this->addModule( "fc_attn_proj", fc_attn_proj_ );
            this->addModule( "res_1", res_1_ );
            this->addModule( "ln_2", ln_2_ );
            this->addModule( "mlp", mlp_ );
            this->addModule( "res_2", res_2_ );

            // Create output tensors for the intermediate steps
            ln_1_output_ = Tensor<TDataType, MR>( input_shape_ );
            fc_qkv_output_ = Tensor<TDataType, MR>( { B, T, 3 * C } );
            attn_output_ = Tensor<TDataType, MR>( input_shape_ );
            fc_attn_proj_output_ = Tensor<TDataType, MR>( { B, T, C } );
            res_1_output_ = Tensor<TDataType, MR>( input_shape_ );
            ln_2_output_ = Tensor<TDataType, MR>( input_shape_ );
            mlp_output_ = Tensor<TDataType, MR>( input_shape_ );
            res_2_output_ = Tensor<TDataType, MR>( input_shape_ );
        }
    };

    /**
     * @brief Type alias for CPU-based transformer block with customizable tensor type.
     *
     * @tparam TDataType Data type used for tensor elements throughout the network.
     */
    export template<typename TDataType = float>
        using CpuTransformerBlock = TransformerBlock<DeviceType::Cpu, TDataType>;

    /**
     * @brief Type alias for CUDA-based transformer block with customizable tensor type.
     *
     * @tparam TDataType Data type used for tensor elements throughout the network.
     */
    export template<typename TDataType = float>
        using CudaTransformerBlock = TransformerBlock<DeviceType::Cuda, TDataType>;
}