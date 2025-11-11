/**
 * @file Attention.ixx
 * @brief Device-templated Multi-Head Attention module for transformer architectures.
 *
 * Delegates compute to a UnaryOperation backend. Module has no trainable parameters
 * in the base implementation (Q, K, V are expected to be pre-projected in the input).
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <cmath>
#include <stdexcept>
#include <cstdint>

export module Dnn.Modules.Attention;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Multi-Head Attention module for transformer architectures (device-templated).
     *
     * Delegates computation to a device-specific UnaryOperation implementation
     * registered in the OperationRegistry.
     *
     * Implements the scaled dot-product attention mechanism with multiple heads:
     *   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
     *
     * Input format: Concatenated Q, K, V as [B, T, 3*C] where:
     * - B = batch size
     * - T = sequence length
     * - C = embedding dimension
     * - Q, K, V are each [B, T, C]
     *
     * The module has no trainable parameters in this base implementation.
     * Q, K, V projections are expected to be pre-computed and concatenated in the input.
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Attention : public Module<TDeviceType>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;

        /**
         * @brief Construct with an existing execution context.
         *
         * @param exec_context Shared execution context for device resources.
         * @param config Multi-head attention configuration.
         */
        explicit Attention( std::shared_ptr<ExecutionContextType> exec_context, const AttentionConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            createOperation();
        }

        ~Attention() override = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        bool isBuilt() const override
        {
            return (operation_ != nullptr) && is_built_;
        }

        /**
         * @brief Build the module using an input shape.
         *
         * Multi-head attention has no trainable parameters in this base implementation.
         * This method validates input shape and triggers backend-specific setup.
         *
         * @param input_shape Expected shape: [B, T, 3*C] (concatenated Q, K, V)
         */
        void build( const shape_t& input_shape ) override
        {
            if (is_built_)
                return;

            validateInputShape( input_shape );

            operation_->setTraining( is_training_ );

            // Multi-head attention has no parameters to bind
            operation_->setParameters( nullptr, nullptr );

            operation_->build( input_shape );

            is_built_ = true;
        }

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - delegates to backend operation.
         *
         * Computes scaled dot-product attention with multiple heads:
         * 1. Split input into Q, K, V
         * 2. Compute attention scores: scores = Q * K^T / sqrt(d_k)
         * 3. Apply causal mask (if enabled)
         * 4. Apply softmax: weights = softmax(scores)
         * 5. Compute output: output = weights * V
         *
         * @param input Input tensor [B, T, 3*C] containing concatenated Q, K, V
         * @param output Output tensor [B, T, C]
         */
        void forward( const ITensor& input, ITensor& output )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Attention module must be built before calling forward." );
            }

            validateInputShape( input );

            operation_->forward( input, output );
        }

        /**
         * @brief Backward pass - delegates to backend operation.
         *
         * Computes gradients for the attention mechanism.
         * No parameter gradients since this module has no trainable parameters.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Attention module must be built before calling backward." );
            }

            if (!is_training_)
            {
                throw std::runtime_error( "Attention module must be in training mode to call backward. Call setTraining(true) first." );
            }

            operation_->backward( input, output_grad, input_grad );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save( ModelArchive& archive ) const override
        {
            // No trainable parameters in base multi-head attention implementation
        }

        void load( ModelArchive& archive ) override
        {
            // No trainable parameters in base multi-head attention implementation
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        std::vector<ITensor*> getParameters() const override
        {
            // No trainable parameters in base implementation
            return {};
        }

        std::vector<ITensor*> getParameterGradients() const override
        {
            // No trainable parameters in base implementation
            return {};
        }

        // ====================================================================
        // Module interface
        // ====================================================================

        std::string getName() const override
        {
            return config_.getName();
        }

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            return exec_context_->getDevice();
        }

        void synchronize() override
        {
            exec_context_->synchronize();
        }

        void setTraining( bool is_training ) override
        {
            if (is_training_ == is_training)
                return;

            is_training_ = is_training;

            // Propagate training mode to operation (if created)
            if (operation_)
            {
                operation_->setTraining( is_training );
            }
        }

        bool isTraining() const override
        {
            return is_training_;
        }

        size_t parameterCount() const override
        {
            // No trainable parameters in base multi-head attention implementation
            return 0;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Attention: " << getName() << std::endl;
            oss << "Embedding dimension: " << config_.getEmbeddingDim() << std::endl;
            oss << "Number of heads: " << config_.getNumHeads() << std::endl;
            oss << "Head size: " << (config_.getEmbeddingDim() / config_.getNumHeads()) << std::endl;

            if (!config_.getInputShape().empty())
            {
                const auto& input_shape = config_.getInputShape();
                oss << "Input shape: (";
                for (size_t i = 0; i < input_shape.size(); ++i)
                {
                    oss << input_shape[i];
                    if (i != input_shape.size() - 1)
                        oss << ", ";
                }
                oss << ")" << std::endl;
            }

            oss << "Dropout: " << config_.getDropout() << std::endl;
            oss << "Causal mask: " << (config_.useCausalMask() ? "enabled" : "disabled") << std::endl;
            oss << "Scale factor: " << config_.getScaleFactor() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

        // ====================================================================
        // Configuration accessors
        // ====================================================================

        /**
         * @brief Gets the embedding dimension.
         */
        int64_t getEmbeddingDim() const noexcept
        {
            return config_.getEmbeddingDim();
        }

        /**
         * @brief Gets the number of attention heads.
         */
        int64_t getNumHeads() const noexcept
        {
            return config_.getNumHeads();
        }

        /**
         * @brief Gets the dropout rate.
         */
        float getDropout() const noexcept
        {
            return config_.getDropout();
        }

        /**
         * @brief Checks if causal masking is enabled.
         */
        bool useCausalMask() const noexcept
        {
            return config_.useCausalMask();
        }

        /**
         * @brief Gets the attention scale factor.
         */
        float getScaleFactor() const noexcept
        {
            return config_.getScaleFactor();
        }

        /**
         * @brief Get the configuration.
         */
        const AttentionConfig& getConfig() const noexcept
        {
            return config_;
        }

    private:
        
        AttentionConfig config_;
        bool is_training_{ false };
        bool is_built_{ false };

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

        /**
         * @brief Validate input shape for multi-head attention operation.
         *
         * Expected shape: [B, T, 3*C] where:
         * - B = batch size
         * - T = sequence length
         * - C = embedding dimension
         *
         * The last dimension must be 3 times the embedding dimension
         * to accommodate concatenated Q, K, V.
         */
        void validateInputShape( const ITensor& input ) const
        {
            const auto& input_shape = input.shape();
            validateInputShape( input_shape );
        }

        /**
         * @brief Validate input shape for multi-head attention operation.
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.size() != 3)
            {
                throw std::invalid_argument(
                    "Attention: input must have rank 3 (batch_size, sequence_length, 3*embedding_dim)" );
            }

            int64_t expected_qkv_dim = 3 * config_.getEmbeddingDim();

            if (input_shape[2] != expected_qkv_dim)
            {
                std::ostringstream oss;
                oss << "Attention: input last dimension must be 3*embedding_dim. Expected "
                    << expected_qkv_dim << " (3 * " << config_.getEmbeddingDim() << "), got " << input_shape[2];
                throw std::invalid_argument( oss.str() );
            }

            // Validate embedding dimension is divisible by number of heads
            if (config_.getEmbeddingDim() % config_.getNumHeads() != 0)
            {
                std::ostringstream oss;
                oss << "Attention: embedding dimension (" << config_.getEmbeddingDim()
                    << ") must be divisible by number of heads (" << config_.getNumHeads() << ")";
                throw std::invalid_argument( oss.str() );
            }
        }

        /**
         * @brief Create the backend compute operation.
         *
         * Looks up the appropriate device-specific operation from the registry
         * and creates an instance bound to this module's execution context.
         *
         * Registered operation name: "AttentionOp"
         */
        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "AttentionOp",
                    exec_context_,
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error(
                    "Failed to create Attention compute backend operation. "
                    "Ensure CPU/CUDA operation is registered in OperationRegistry." );
            }
        }
    };

    // Convenience aliases for common usages
    export template<TensorDataType TPrecision>
        using CpuAttention = Attention<DeviceType::Cpu, TPrecision>;

    export template<TensorDataType TPrecision>
        using CudaAttention = Attention<DeviceType::Cuda, TPrecision>;
}