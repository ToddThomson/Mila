/**
 * @file TransformerBlock.ixx
 * @brief Implementation of the transformer encoder block for neural networks.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream> 
#include <stdexcept>
#include <iosfwd>

export module Dnn.Blocks.TransformerBlock;
export import :Config;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Dnn.Module;
import Dnn.CompositeModule;
import Dnn.Modules.LayerNorm;
import Dnn.Modules.Linear;
import Dnn.Modules.Attention;
import Dnn.Modules.Residual;
import Dnn.Modules.Dropout;
import Dnn.Blocks.MLP;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief TransformerBlock implements a standard transformer encoder block.
     *
     * The transformer block consists of:
     * - Multi-head self-attention mechanism with residual connection
     * - Feed-forward network (MLP) with residual connection
     * - Layer normalization before or after each sub-block (configurable)
     *
     * This is the fundamental building block of transformer architectures like BERT and GPT.
     * The implementation supports both pre-LN (more stable) and post-LN (original) architectures,
     * configurable dropout rates, and other hyperparameters.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TDataType The data type used for tensor elements throughout the network.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TDataType = float>
        requires ValidFloatTensorType<TDataType>
    class TransformerBlock : public CompositeModule<TDeviceType, TDataType> {
    public:
        /**
         * @brief Memory resource type used for tensors, selected based on device type.
         */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;

        /**
         * @brief Alias for base module type.
         */
        using CompositeModuleBase = CompositeModule<TDeviceType, TDataType>;

        /**
         * @brief Constructs a new TransformerBlock module with a device name.
         *
         * Creates a new DeviceContext internally using the provided device name.
         * This constructor is useful for creating standalone modules without
         * pre-existing device contexts.
         *
         * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
         * @param config Configuration parameters for the TransformerBlock module.
         * @throws std::invalid_argument If the device name is invalid or the configuration is invalid
         * @throws std::runtime_error If device type doesn't match template parameter TDeviceType
         */
        explicit TransformerBlock( const std::string& device_name, const TransformerBlockConfig& config )
            : CompositeModuleBase( std::make_shared<DeviceContext>( device_name ), config ), config_( config ) {

            config.validate();

            initializeModules();
        }

        /**
         * @brief Constructs a new TransformerBlock module with a provided device context.
         *
         * Uses a pre-existing DeviceContext instance. This constructor is useful when integrating
         * the module into a larger network that shares device contexts across modules.
         *
         * @param device_context The device context to use for this module.
         * @param config Configuration parameters for the TransformerBlock module.
         * @throws std::invalid_argument If device_context is null or configuration is invalid
         * @throws std::runtime_error If device context type doesn't match template parameter TDeviceType
         */
        explicit TransformerBlock( std::shared_ptr<DeviceContext> device_context, const TransformerBlockConfig& config )
            : CompositeModuleBase( device_context, config ), config_( config ) {

            config.validate();

            initializeModules();
        }

        /**
         * @brief Performs the forward pass of the TransformerBlock.
         *
         * The forward pass follows either pre-LN or post-LN architecture based on configuration:
         *
         * Pre-LN (default):
         * 1. Layer normalization 1
         * 2. Self-attention
         * 3. Residual connection
         * 4. Layer normalization 2
         * 5. Feed-forward network
         * 6. Residual connection
         *
         * Post-LN:
         * 1. Self-attention
         * 2. Residual connection
         * 3. Layer normalization 1
         * 4. Feed-forward network
         * 5. Residual connection
         * 6. Layer normalization 2
         *
         * @param input The input tensor to be processed.
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TDataType, MR>& input, Tensor<TDataType, MR>& output ) {
            if ( config_.usePreLayerNorm() ) {
                // Pre-LayerNorm architecture (more stable training)
                ln_1_->forward( input, ln_1_output_ );
                attn_block_->forward( ln_1_output_, attn_output_ );

                // First residual connection
                for ( size_t i = 0; i < input.size(); ++i ) {
                    res_1_output_.data()[ i ] = input.data()[ i ] + attn_output_.data()[ i ];
                }

                ln_2_->forward( res_1_output_, ln_2_output_ );
                mlp_->forward( ln_2_output_, mlp_output_ );

                // Second residual connection
                for ( size_t i = 0; i < res_1_output_.size(); ++i ) {
                    output.data()[ i ] = res_1_output_.data()[ i ] + mlp_output_.data()[ i ];
                }
            }
            else {
                // Post-LayerNorm architecture (original transformer)
                attn_block_->forward( input, attn_output_ );

                // First residual connection
                for ( size_t i = 0; i < input.size(); ++i ) {
                    res_1_output_.data()[ i ] = input.data()[ i ] + attn_output_.data()[ i ];
                }

                ln_1_->forward( res_1_output_, ln_1_output_ );
                mlp_->forward( ln_1_output_, mlp_output_ );

                // Second residual connection
                for ( size_t i = 0; i < res_1_output_.size(); ++i ) {
                    res_2_output_.data()[ i ] = res_1_output_.data()[ i ] + mlp_output_.data()[ i ];
                }

                ln_2_->forward( res_2_output_, output );
            }
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
         * @brief Serializes the module state to a ZIP archive.
         *
         * Saves the state of all sub-modules to the provided ZIP archive.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( ModelArchive& archive ) const override {
            for ( const auto& module : this->getModules() ) {
                module->save( archive );
            }
        }

        /**
         * @brief Deserializes the module state from a ZIP archive.
         *
         * Loads the state of all sub-modules from the provided ZIP archive.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( ModelArchive& archive ) override {
            for ( const auto& module : this->getModules() ) {
                module->load( archive );
            }
        }

        /**
         * @brief Generates a string representation of this module's configuration.
         *
         * @return std::string A formatted string with module information
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "====================" << std::endl;
            oss << "TransformerBlock: " << this->getDeviceName() << std::endl;

            const auto& input_shape = config_.getInputShape();
            oss << "Input shape: (";
            for ( size_t i = 0; i < input_shape.size(); ++i ) {
                oss << input_shape[ i ];
                if ( i != input_shape.size() - 1 ) {
                    oss << ",";
                }
            }
            oss << ")" << std::endl;

            oss << "Embedding dimension: " << input_shape[ 2 ] << std::endl;
            oss << "Number of heads: " << config_.getNumHeads() << std::endl;
            oss << "MLP hidden dimension: " << config_.getHiddenDimension() << std::endl;

            if ( config_.getDropout() > 0.0f ) {
                oss << "Dropout: " << config_.getDropout() << std::endl;
            }

            oss << "Architecture: " << (config_.usePreLayerNorm() ? "Pre-LN" : "Post-LN") << std::endl;
            oss << "Bias: " << (config_.useBias() ? "enabled" : "disabled") << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;
            oss << "Sub-Modules..." << std::endl;

            for ( const auto& [name, module] : this->getDeviceNamedModules() ) {
                oss << module->toString();
            }

            return oss.str();
        }

    protected:

        void buildImpl( const shape_t& input_shape ) override
        {
            // Attention gets {batch, seq_len, d_model}
            attention_->build( input_shape );

            // LayerNorms get same shape
            ln1_->build( input_shape );
            ln2_->build( input_shape );

            // FFN might expand/contract dimensions
            ffn_->build( input_shape );
        }

    private:
        /**
         * @brief Configuration for the TransformerBlock module.
         */
        TransformerBlockConfig config_;

        /**
         * @brief First layer normalization module.
         *
         * In pre-LN architecture, applied before attention.
         * In post-LN architecture, applied after attention and residual connection.
         */
        std::shared_ptr<LayerNorm<TDeviceType, TDataType>> ln_1_{ nullptr };

        /**
         * @brief Second layer normalization module.
         *
         * In pre-LN architecture, applied before MLP.
         * In post-LN architecture, applied after MLP and residual connection.
         */
        std::shared_ptr<LayerNorm<TDeviceType, TDataType>> ln_2_{ nullptr };

        /**
         * @brief Multi-head self-attention block including projections.
         */
        std::shared_ptr<Attention<TDeviceType, TDataType>> attn_block_{ nullptr };

        /**
         * @brief Feed-forward network (MLP).
         */
        std::shared_ptr<MLP<TDeviceType, TDataType>> mlp_{ nullptr };

        /**
         * @brief Optional dropout module.
         */
        std::shared_ptr<Dropout<TDeviceType, TDataType>> dropout_{ nullptr };

        /**
         * @brief Output tensor from first layer normalization.
         */
        Tensor<TDataType, MR> ln_1_output_;

        /**
         * @brief Output tensor from attention block.
         */
        Tensor<TDataType, MR> attn_output_;

        /**
         * @brief Output tensor from first residual connection.
         */
        Tensor<TDataType, MR> res_1_output_;

        /**
         * @brief Output tensor from second layer normalization.
         */
        Tensor<TDataType, MR> ln_2_output_;

        /**
         * @brief Output tensor from MLP.
         */
        Tensor<TDataType, MR> mlp_output_;

        /**
         * @brief Output tensor from second residual connection.
         */
        Tensor<TDataType, MR> res_2_output_;

        /**
         * @brief Initializes the sub-modules and output tensors for the transformer block.
         *
         * Creates and configures all components of the transformer block according to
         * the configuration, including layer norm, attention, and feed-forward network.
         */
        void initializeModules() {
            // Clear any existing modules
            for ( const auto& [name, _] : this->getDeviceNamedModules() ) {
                this->removeModule( name );
            }

            const auto& input_shape = config_.getInputShape();
            size_t B = input_shape[ 0 ]; // Batch size
            size_t T = input_shape[ 1 ]; // Sequence length
            size_t C = input_shape[ 2 ]; // Embedding dimension
            size_t num_heads = config_.getNumHeads();
            size_t hidden_dim = config_.getHiddenDimension();
            bool use_bias = config_.useBias();
            float dropout_rate = config_.getDropout();

            // Create layer normalization modules
            auto ln_1_config = LayerNormConfig( C )
                .withName( this->getDeviceName() + ".ln_1" )
                .withTraining( this->isTraining() );

            ln_1_ = std::make_shared<LayerNorm<TDeviceType, TDataType>>(
                this->getDeviceContext(), ln_1_config );

            auto ln_2_config = LayerNormConfig( C )
                .withName( this->getDeviceName() + ".ln_2" )
                .withTraining( this->isTraining() );

            ln_2_ = std::make_shared<LayerNorm<TDeviceType, TDataType>>(
                this->getDeviceContext(), ln_2_config );

            // Create attention module
            auto attn_config = AttentionConfig( C, num_heads )
                .withName( this->getDeviceName() + ".attn" )
                .withTraining( this->isTraining() );

            attn_block_ = std::make_shared<Attention<TDeviceType, TDataType>>(
                this->getDeviceContext(), attn_config );

            // Create MLP module
            auto mlp_config = MLPConfig( input_shape, hidden_dim )
                .withName( this->getDeviceName() + ".mlp" )
                .withBias( use_bias )
                .withActivation( config_.getActivationType() )
                .withDropout( dropout_rate )
                .withTraining( this->isTraining() );

            mlp_ = std::make_shared<MLP<TDeviceType, TDataType>>(
                this->getDeviceContext(), mlp_config );

            // Create dropout module if needed
            if ( dropout_rate > 0.0f ) {
                auto dropout_config = DropoutConfig( dropout_rate )
                    .withName( this->getDeviceName() + ".dropout" )
                    .withTraining( this->isTraining() );

                dropout_ = std::make_shared<Dropout<TDeviceType, TDataType>>(
                    this->getDeviceContext(), dropout_config );
            }

            // Add modules to composite
            this->addModule( "ln_1", ln_1_ );
            this->addModule( "attn", attn_block_ );
            this->addModule( "ln_2", ln_2_ );
            this->addModule( "mlp", mlp_ );

            if ( dropout_ ) {
                this->addModule( "dropout", dropout_ );
            }

            // Initialize intermediate tensors
            ln_1_output_ = Tensor<TDataType, MR>( input_shape );
            attn_output_ = Tensor<TDataType, MR>( input_shape );
            res_1_output_ = Tensor<TDataType, MR>( input_shape );
            ln_2_output_ = Tensor<TDataType, MR>( input_shape );
            mlp_output_ = Tensor<TDataType, MR>( input_shape );
            res_2_output_ = Tensor<TDataType, MR>( input_shape );
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