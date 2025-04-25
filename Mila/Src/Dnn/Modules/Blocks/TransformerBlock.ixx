module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream> 
#include <stdexcept>
#include <iosfwd>

export module Dnn.Blocks.TransformerBlock;

import Dnn.ModuleBlock;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.CpuDevice;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Dnn.Module;
import Dnn.Modules.LayerNorm;
import Dnn.Modules.FullyConnected;
import Dnn.Modules.Attention;
import Dnn.Modules.Residual;
import Dnn.Blocks.MLP;

namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

    export
        template<typename TPrecision, typename TInput = TPrecision, DeviceType TDeviceType = DeviceType::Cuda>
		requires ValidFloatTensorType<TPrecision> && ValidTensorType<TInput>
    class TransformerBlock : public BlockModule<TPrecision, TInput, TDeviceType> {
    public:
		using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, DeviceMemoryResource, HostMemoryResource>;

        /**
         * @brief Constructs a new TransformerBlock module with the default device context.
         *
         * @param name The name of the module for identification purposes.
         * @param input_shape The shape of the input tensor, must be of rank 3 [batch_size, sequence_length, channels].
         * @param num_heads The number of attention heads to use.
         * @param is_training Whether the module is initially in training mode. Default is false.
         * @throws std::invalid_argument If the input shape doesn't have rank 3.
         */
        TransformerBlock( std::string name, const std::vector<size_t>& input_shape, const size_t num_heads, bool is_training = false )
            : Module<TPrecision, TInput, TDeviceType>(),
            input_shape_{ validate_shape( input_shape ) },
            num_heads_{ num_heads } {
            this->setName( name );
            this->setTraining( is_training );

            initializeModules();
        }

        /**
         * @brief Constructs a new TransformerBlock module with a specific device context.
         *
         * @param name The name of the module for identification purposes.
         * @param input_shape The shape of the input tensor, must be of rank 3 [batch_size, sequence_length, channels].
         * @param num_heads The number of attention heads to use.
         * @param context The device context to use for this module.
         * @param is_training Whether the module is initially in training mode. Default is false.
         * @throws std::invalid_argument If the input shape doesn't have rank 3.
         */
        TransformerBlock( std::string name, const std::vector<size_t>& input_shape, const size_t num_heads,
            std::shared_ptr<DeviceContext> context, bool is_training = false )
            : Module<TPrecision, TInput, TDeviceType>( context ),
            input_shape_{ validate_shape( input_shape ) },
            num_heads_{ num_heads } {
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
        template<typename TMR>
        void forward( const Tensor<TInput, TMR>& input, Tensor<TPrecision, TMR>& output ) {
            ln_1_->forward( input, ln_1_output_ );
            fc_qkv_->forward( ln_1_output_, fc_qkv_output_ );
            attn_->forward( fc_qkv_output_, attn_output_ );
            fc_attn_proj_->forward( attn_output_, fc_attn_proj_output_ );
            res_1_->forward( input, fc_attn_proj_output_, res_1_output_ );
            ln_2_->forward( res_1_output_, ln_2_output_ );
            mlp_->forward( ln_2_output_, mlp_output_ );
            res_2_->forward( res_1_output_, mlp_output_, output );
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
            oss << "Parameter count: " << parameterCount() << std::endl;
            oss << "Sub-Modules..." << std::endl;

            for ( const auto& [name, module] : this->getNamedModules() ) {
                oss << *module;
            }

            return oss.str();
        }

    protected:
        /**
         * @brief Called when the device context changes.
         *
         * Recreates sub-modules and output tensors for the new device.
         */
        void onDeviceChanged() override {
            initializeModules();
        }

    private:
        std::vector<size_t> input_shape_; ///< The input shape.
        size_t num_heads_; ///< The number of attention heads.

        // Sub-modules
        std::shared_ptr<LayerNorm<TPrecision>> ln_1_{ nullptr };
        std::shared_ptr<FullyConnected<TPrecision>> fc_qkv_{ nullptr };
        std::shared_ptr<MultiHeadAttention<TPrecision>> attn_{ nullptr };
        std::shared_ptr<FullyConnected<TPrecision>> fc_attn_proj_{ nullptr };
        std::shared_ptr<Residual<TPrecision>> res_1_{ nullptr };
        std::shared_ptr<LayerNorm<TPrecision>> ln_2_{ nullptr };
        std::shared_ptr<MLP<TPrecision>> mlp_{ nullptr };
        std::shared_ptr<Residual<TPrecision>> res_2_{ nullptr };

        // Intermediate tensors
        Tensor<TPrecision, MR> ln_1_output_;
        Tensor<TPrecision, MR> fc_qkv_output_;
        Tensor<TPrecision, MR> attn_output_;
        Tensor<TPrecision, MR> fc_attn_proj_output_;
        Tensor<TPrecision, MR> res_1_output_;
        Tensor<TPrecision, MR> ln_2_output_;
        Tensor<TPrecision, MR> mlp_output_;
        Tensor<TPrecision, MR> res_2_output_;

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
            // Clear existing modules if any
            for ( const auto& [name, _] : this->getNamedModules() ) {
                this->removeModule( name );
            }

            auto B = input_shape_[ 0 ]; // Batch size
            auto T = input_shape_[ 1 ]; // Sequence length
            auto C = input_shape_[ 2 ]; // Number of channels

            // Create new modules with the current device context
            ln_1_ = std::make_shared<LayerNorm<TPrecision>>(
                this->getName() + ".ln_1", input_shape_ );

            fc_qkv_ = std::make_shared<FullyConnected<TPrecision>>(
                this->getName() + ".fc_qkv", C, 3 * C );

            attn_ = std::make_shared<MultiHeadAttention<TPrecision>>(
                this->getName() + ".attn", input_shape_, num_heads_ );

            fc_attn_proj_ = std::make_shared<FullyConnected<TPrecision>>(
                this->getName() + ".fc_attn_proj", C, C );

            res_1_ = std::make_shared<Residual<TPrecision>>(
                this->getName() + ".res_1" );

            ln_2_ = std::make_shared<LayerNorm<TPrecision>>(
                this->getName() + ".ln_2", input_shape_ );

            mlp_ = std::make_shared<MLP<TPrecision>>(
                this->getName() + ".mlp", input_shape_, 4 * C );

            res_2_ = std::make_shared<Residual<TPrecision>>(
                this->getName() + ".res_2" );

            // Propagate device context to sub-modules
            ln_1_->setDeviceContext( this->getDeviceContext() );
            fc_qkv_->setDeviceContext( this->getDeviceContext() );
            attn_->setDeviceContext( this->getDeviceContext() );
            fc_attn_proj_->setDeviceContext( this->getDeviceContext() );
            res_1_->setDeviceContext( this->getDeviceContext() );
            ln_2_->setDeviceContext( this->getDeviceContext() );
            mlp_->setDeviceContext( this->getDeviceContext() );
            res_2_->setDeviceContext( this->getDeviceContext() );

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
            auto device_type = this->getDeviceContext()->getDevice()->getDeviceType();

            // Create output tensors for the intermediate steps
            ln_1_output_ = Tensor<TPrecision, MR>( input_shape_ );
            fc_qkv_output_ = Tensor<TPrecision, MR>( { B, T, 3 * C } );
            attn_output_ = Tensor<TPrecision, MR>( input_shape_ );
            fc_attn_proj_output_ = Tensor<TPrecision, MR>( { B, T, C } );
            res_1_output_ = Tensor<TPrecision, MR>( input_shape_ );
            ln_2_output_ = Tensor<TPrecision, MR>( input_shape_ );
            mlp_output_ = Tensor<TPrecision, MR>( input_shape_ );
            res_2_output_ = Tensor<TPrecision, MR>( input_shape_ );
        }
    };
}
