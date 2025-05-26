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
     * The attention mechanism computes: Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
     * where Q, K, and V are the query, key, and value projections of the input.
     *
     * Multi-head attention projects the input into multiple subspaces, computes attention
     * independently in each subspace, then concatenates the results to form the output.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TOutput The data type of the output tensor elements, defaults to TInput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidTensorTypes<TInput, TOutput>
    class MultiHeadAttention : public Module<TDeviceType, TInput, TOutput> {
    public:
        /**
         * @brief Memory resource type used for tensors, selected based on device type.
         */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;

        /**
         * @brief Alias for base module type.
         */
        using ModuleBase = Module<TDeviceType, TInput, TOutput>;

        /**
         * @brief Constructs a new MultiHeadAttention module with a device name.
         *
         * Creates a new DeviceContext internally using the provided device name.
         * This constructor is useful for creating standalone modules without
         * pre-existing device contexts.
         *
         * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
         * @param config Configuration parameters for the MultiHeadAttention module.
         * @throws std::invalid_argument If the device name is invalid or the configuration is invalid
         * @throws std::runtime_error If device type doesn't match template parameter TDeviceType
         */
        explicit MultiHeadAttention( const std::string& device_name, const MultiHeadAttentionConfig& config )
            : ModuleBase( std::make_shared<DeviceContext>( device_name ), config ), config_( config ) {

            config.validate();

            initializeTensors();
            createOperation();
        }

        /**
         * @brief Constructs a new MultiHeadAttention module with a provided device context.
         *
         * Uses a pre-existing DeviceContext instance. This constructor is useful when integrating
         * the module into a larger network that shares device contexts across modules.
         *
         * @param device_context The device context to use for this module.
         * @param config Configuration parameters for the MultiHeadAttention module.
         * @throws std::invalid_argument If device_context is null or configuration is invalid
         * @throws std::runtime_error If device context type doesn't match template parameter TDeviceType
         */
        explicit MultiHeadAttention( std::shared_ptr<DeviceContext> device_context, const MultiHeadAttentionConfig& config )
            : ModuleBase( device_context, config ), config_( config ) {

            config.validate();

            initializeTensors();
            createOperation();
        }

        /**
         * @brief Gets the number of trainable parameters in this module.
         *
         * @return size_t The total number of parameters.
         */
        size_t parameterCount() const override {
            return 0;  // Base implementation has no parameters
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
         * @brief Gets the embedding dimension.
         *
         * @return size_t The embedding dimension.
         */
        size_t getEmbeddingDim() const {
            return config_.getEmbeddingDim();
        }

        /**
         * @brief Gets the number of attention heads.
         *
         * @return size_t The number of attention heads.
         */
        size_t getNumHeads() const {
            return config_.getNumHeads();
        }

        /**
         * @brief Gets the input shape.
         *
         * @return const std::vector<size_t>& The input shape.
         */
        const std::vector<size_t>& getInputShape() const {
            return config_.getInputShape();
        }

        /**
         * @brief Gets the dropout rate.
         *
         * @return float The dropout rate.
         */
        float getDropout() const {
            return config_.getDropout();
        }

        /**
         * @brief Checks if causal masking is enabled.
         *
         * @return bool True if causal masking is enabled, false otherwise.
         */
        bool useCausalMask() const {
            return config_.useCausalMask();
        }

        /**
         * @brief Saves the module state to a ZIP archive.
         *
         * Implementation of the Module interface for serialization. Currently a no-op
         * in the base implementation as there are no parameters to save.
         *
         * @param zip ZIP archive for serialization
         */
        void save( mz_zip_archive& zip ) const override {
            // MultiHeadAttention has no parameters to save in this base implementation
        }

        /**
         * @brief Loads the module state from a ZIP archive.
         *
         * Implementation of the Module interface for deserialization. Currently a no-op
         * in the base implementation as there are no parameters to load.
         *
         * @param zip ZIP archive for deserialization
         */
        void load( mz_zip_archive& zip ) override {
            // MultiHeadAttention has no parameters to load in this base implementation
        }

        /**
         * @brief Generates a string representation of this module's configuration.
         *
         * @return std::string A formatted string with module information
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "MultiHeadAttention: " << this->getName() << std::endl;
            oss << "Embedding dimension: " << config_.getEmbeddingDim();
            oss << ", Number of heads: " << config_.getNumHeads() << std::endl;

            const auto& input_shape = config_.getInputShape();
            oss << "Input shape: (";
            for ( size_t i = 0; i < input_shape.size(); ++i ) {
                oss << input_shape[ i ];
                if ( i != input_shape.size() - 1 ) {
                    oss << ",";
                }
            }
            oss << ")" << std::endl;

            oss << "Dropout: " << config_.getDropout();
            oss << ", Causal mask: " << (config_.useCausalMask() ? "enabled" : "disabled") << std::endl;
            oss << "Scale factor: " << config_.getScaleFactor();
            oss << ", Separate projections: " << (config_.useSeparateProjections() ? "yes" : "no") << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;

            return oss.str();
        }

    private:
        /**
         * @brief Configuration for the MultiHeadAttention module.
         */
        MultiHeadAttentionConfig config_;

        /**
         * @brief Attention weight tensor from the forward pass.
         *
         * Shape: [batch_size, num_heads, sequence_length, sequence_length]
         * Stores the attention weights between all token pairs.
         */
        std::shared_ptr<Tensor<TOutput, MR>> attn_{ nullptr };

        /**
         * @brief Pre-softmax attention scores from the forward pass.
         *
         * Shape: [batch_size, num_heads, sequence_length, sequence_length]
         * Stores the raw attention scores before softmax normalization.
         */
        std::shared_ptr<Tensor<TOutput, MR>> pre_attn_{ nullptr };

        /**
         * @brief Collection of parameters for this module.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameters_;

        /**
         * @brief Collection of output state tensors for caching.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> output_state_;

        /**
         * @brief Operation attributes and configuration.
         */
        OperationAttributes properties_;

        /**
         * @brief The operation that implements the attention mechanism.
         */
        std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput>> operation_{ nullptr };

        /**
         * @brief Initializes the tensors needed for attention computation.
         *
         * Creates and initializes intermediate tensors used during the attention
         * computation, including attention weights and pre-softmax scores.
         */
        void initializeTensors() {
            output_state_.clear();

            const auto& input_shape = config_.getInputShape();
            if ( input_shape.empty() ) {
                return;  // Can't initialize without shape information
            }

            size_t batch_size = input_shape[ 0 ];
            size_t sequence_length = input_shape[ 1 ];
            size_t num_heads = config_.getNumHeads();

            pre_attn_ = std::make_shared<Tensor<TOutput, MR>>(
                std::vector<size_t>{batch_size, num_heads, sequence_length, sequence_length} );
            pre_attn_->setName( this->getName() + ".pre_attn" );

            attn_ = std::make_shared<Tensor<TOutput, MR>>(
                std::vector<size_t>{batch_size, num_heads, sequence_length, sequence_length} );
            attn_->setName( this->getName() + ".attn" );

            output_state_.emplace_back( pre_attn_ );
            output_state_.emplace_back( attn_ );

            // Set properties from config
            properties_.set( "num_heads", config_.getNumHeads() );
            properties_.set( "dropout", config_.getDropout() );
            properties_.set( "causal_mask", config_.useCausalMask() );
            properties_.set( "scale_factor", config_.getScaleFactor() );
            properties_.set( "separate_projections", config_.useSeparateProjections() );
        }

        /**
         * @brief Creates the appropriate attention operation for the current device.
         *
         * Instantiates either a CPU or CUDA attention operation based on the device type.
         */
        void createOperation() {
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TInput, TOutput>(
                    "Cpu::MultiHeadAttentionOp",
                    this->getDeviceContext(),
                    config_ );
                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TInput, TOutput>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TInput, TOutput>(
                    "Cuda::MultiHeadAttentionOp",
                    this->getDeviceContext(),
                    config_ );
                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TInput, TOutput>>(base_op);
            }
        }
    };

    /**
     * @brief Type alias for CPU-based multi-head attention module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CpuMultiHeadAttention = MultiHeadAttention<DeviceType::Cpu, TInput, TOutput>;

    /**
     * @brief Type alias for CUDA-based multi-head attention module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CudaMultiHeadAttention = MultiHeadAttention<DeviceType::Cuda, TInput, TOutput>;
}