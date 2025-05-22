/**
 * @file MultiHeadAttention.ixx
 * @brief Implementation of multi-head attention mechanism for transformer architectures.
 */

module;
#include <miniz.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <cmath>

export module Dnn.Modules.Attention;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;

import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.CpuDevice;
import Compute.CudaDevice;

import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Compute.Precision;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Multi-head attention module for transformer architectures.
     *
     * This module implements the multi-head attention mechanism, which allows different
     * parts of the input to attend to different parts of the sequence. This is a core
     * component of transformer architectures.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TOutput The data type of the output tensor elements, defaults to TInput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidTensorTypes<TInput, TOutput>
    class MultiHeadAttention : public Module<TDeviceType, TInput, TOutput> {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;
        using ModuleBase = Module<TDeviceType, TInput, TOutput>;

        /**
         * @brief Construct a new MultiHeadAttention module from configuration.
         *
         * @param config The configuration for this module
         */
        explicit MultiHeadAttention( const MultiHeadAttentionConfig& config )
            : ModuleBase(
                config.getContext() ? config.getContext() : std::make_shared<DeviceContext>( config.getDeviceName() ),
                TDeviceType == DeviceType::Cpu ? ComputePrecision::Policy::Disabled : config.getPrecision() ),
            embedding_dim_( config.getEmbeddingDim() ),
            num_heads_( config.getNumHeads() ),
            dropout_( config.getDropout() ),
            use_causal_mask_( config.useCausalMask() ),
            scale_factor_( config.getScaleFactor() ),
            separate_projections_( config.useSeparateProjections() ) {

            config.validate();

            this->setName( config.getName() );
            this->setTraining( config.isTraining() );

            const auto& input_shape = config.getInputShape();
            if ( !input_shape.empty() ) {
                input_shape_ = input_shape;
            }
            else {
                // Default shape if not provided
                input_shape_ = { 1, 1, embedding_dim_ };
            }

            initializeTensors();
            createOperation();
        }

        /**
         * @brief Constructs a new MultiHeadAttention module with the default device context.
         *
         * @param name The name of the module for identification purposes.
         * @param device_name The name of the device to use for this module.
         * @param input_shape The shape of the input tensor.
         * @param num_heads The number of attention heads.
         * @param precision The compute precision policy (CPU operations always use Disabled).
         * @param is_training Whether the module is initially in training mode. Default is false.
         */
        MultiHeadAttention( std::string name, std::string device_name,
            const std::vector<size_t>& input_shape, size_t num_heads,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto,
            bool is_training = false )
            : ModuleBase( device_name, TDeviceType == DeviceType::Cpu ? ComputePrecision::Policy::Disabled : precision ),
            input_shape_( input_shape ),
            num_heads_( num_heads ),
            embedding_dim_( input_shape.empty() ? 0 : input_shape.back() ),
            dropout_( 0.0f ),
            use_causal_mask_( false ),
            scale_factor_( embedding_dim_ > 0 ? 1.0f / std::sqrt( embedding_dim_ / num_heads ) : 1.0f ),
            separate_projections_( true ) {
            this->setTraining( is_training );
            this->setName( name );
            initializeTensors();
            createOperation();
        }

        /**
         * @brief Constructs a new MultiHeadAttention module with a specific device context.
         *
         * @param name The name of the module for identification purposes.
         * @param context The device context to use for this module.
         * @param input_shape The shape of the input tensor.
         * @param num_heads The number of attention heads.
         * @param precision The compute precision policy (CPU operations always use Disabled).
         * @param is_training Whether the module is initially in training mode. Default is false.
         */
        MultiHeadAttention( std::string name, std::shared_ptr<DeviceContext> context,
            const std::vector<size_t>& input_shape, size_t num_heads,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto,
            bool is_training = false )
            : ModuleBase( context, TDeviceType == DeviceType::Cpu ? ComputePrecision::Policy::Disabled : precision ),
            input_shape_( input_shape ),
            num_heads_( num_heads ),
            embedding_dim_( input_shape.empty() ? 0 : input_shape.back() ),
            dropout_( 0.0f ),
            use_causal_mask_( false ),
            scale_factor_( embedding_dim_ > 0 ? 1.0f / std::sqrt( embedding_dim_ / num_heads ) : 1.0f ),
            separate_projections_( true ) {
            this->setTraining( is_training );
            this->setName( name );
            initializeTensors();
            createOperation();
        }

        /**
         * @brief Get the number of parameters.
         */
        size_t parameterCount() const override {
            return 0;
        }

        /**
         * @brief Performs the forward pass of the MultiHeadAttention operation.
         *
         * @param input The input tensor to be processed.
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TInput, MR>& input, Tensor<TOutput, MR>& output ) {
            operation_->forward( input, parameters_, properties_, output, output_state_ );
        }

        /**
         * @brief Performs the forward pass with an explicit attention mask.
         *
         * @param input The input tensor to be processed.
         * @param mask The attention mask tensor (0s for masked positions).
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TInput, MR>& input, const Tensor<TInput, MR>& mask, Tensor<TOutput, MR>& output ) {
            // Store the mask in operation properties
            properties_.set( "explicit_mask", true );

            // Create a temporary input that combines input and mask for the operation
            auto batch_size = input.getShape()[ 0 ];
            auto seq_length = input.getShape()[ 1 ];

            // This implementation assumes a specific operation handling for masked attention
            // The actual implementation would depend on how your operation expects to receive the mask
            operation_->forward( input, mask, parameters_, properties_, output, output_state_ );
        }

        /**
         * @brief Performs the backward pass of the MultiHeadAttention operation.
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
                parameters_,     // Parameters (empty for this base implementation)
                {},              // Parameter gradients (empty for this base implementation)
                input_grad,      // Gradient to propagate to previous layer
                properties_,     // Operation properties
                output_state_    // Cached tensors from forward pass
            );
        }

        /**
         * @brief Get the embedding dimension.
         */
        size_t getEmbeddingDim() const { return embedding_dim_; }

        /**
         * @brief Get the number of attention heads.
         */
        size_t getNumHeads() const { return num_heads_; }

        /**
         * @brief Get the input shape.
         */
        const std::vector<size_t>& getInputShape() const { return input_shape_; }

        /**
         * @brief Get the dropout rate.
         */
        float getDropout() const { return dropout_; }

        /**
         * @brief Check if causal masking is enabled.
         */
        bool useCausalMask() const { return use_causal_mask_; }

        /**
         * @brief Saves the module state to a ZIP archive.
         */
        void save( mz_zip_archive& zip ) const override {
            // MultiHeadAttention has no parameters to save in this base implementation
        }

        /**
         * @brief Loads the module state from a ZIP archive.
         */
        void load( mz_zip_archive& zip ) override {
            // MultiHeadAttention has no parameters to load in this base implementation
        }

        /**
         * @brief Convert the module information to string.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "MultiHeadAttention: " << this->getName();
            oss << ", Embedding dimension: " << embedding_dim_;
            oss << ", Number of heads: " << num_heads_;
            oss << ", Input shape: (";

            for ( size_t i = 0; i < input_shape_.size(); ++i ) {
                oss << input_shape_[ i ];
                if ( i != input_shape_.size() - 1 ) {
                    oss << ",";
                }
            }

            oss << ")";
            oss << ", Dropout: " << dropout_;
            oss << ", Causal mask: " << (use_causal_mask_ ? "enabled" : "disabled");
            oss << ", Scale factor: " << scale_factor_;
            oss << ", Separate projections: " << (separate_projections_ ? "yes" : "no");
            oss << ", Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;
            oss << this->stateToString();

            return oss.str();
        }

    private:
        std::vector<size_t> input_shape_;
        size_t num_heads_{ 0 };
        size_t embedding_dim_{ 0 };
        float dropout_{ 0.0f };
        bool use_causal_mask_{ false };
        float scale_factor_{ 1.0f };
        bool separate_projections_{ true };

        std::shared_ptr<Tensor<TOutput, MR>> attn_{ nullptr };
        std::shared_ptr<Tensor<TOutput, MR>> pre_attn_{ nullptr };

        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameters_;
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> output_state_;
        OperationAttributes properties_;
        std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput>> operation_{ nullptr };

        void initializeTensors() {
            output_state_.clear();

            auto batch_size = input_shape_[ 0 ];
            auto sequence_length = input_shape_[ 1 ];

            if ( embedding_dim_ == 0 && !input_shape_.empty() && input_shape_.size() >= 3 ) {
                embedding_dim_ = input_shape_[ 2 ];
            }

            pre_attn_ = std::make_shared<Tensor<TOutput, MR>>(
                std::vector<size_t>{batch_size, num_heads_, sequence_length, sequence_length} );
            pre_attn_->setName( this->getName() + ".pre_attn" );

            attn_ = std::make_shared<Tensor<TOutput, MR>>(
                std::vector<size_t>{batch_size, num_heads_, sequence_length, sequence_length} );
            attn_->setName( this->getName() + ".attn" );

            output_state_.emplace_back( pre_attn_ );
            output_state_.emplace_back( attn_ );

            properties_.set( "num_heads", num_heads_ );
            properties_.set( "dropout", dropout_ );
            properties_.set( "causal_mask", use_causal_mask_ );
            properties_.set( "scale_factor", scale_factor_ );
            properties_.set( "separate_projections", separate_projections_ );
        }

        void createOperation() {
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TInput, TOutput>(
                    "Cpu::MultiHeadAttentionOp",
                    this->getDeviceContext(),
                    ComputePrecision::Policy::Disabled );
                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TInput, TOutput>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TInput, TOutput>(
                    "Cuda::MultiHeadAttentionOp",
                    this->getDeviceContext(),
                    this->getComputePrecision().getPolicy() );
                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TInput, TOutput>>(base_op);
            }
        }
    };

    export template<typename TInput = float, typename TOutput = TInput>
        using CpuMultiHeadAttention = MultiHeadAttention<DeviceType::Cpu, TInput, TOutput>;

    export template<typename TInput = float, typename TOutput = TInput>
        using CudaMultiHeadAttention = MultiHeadAttention<DeviceType::Cuda, TInput, TOutput>;
}