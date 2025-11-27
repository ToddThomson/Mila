/**
 * @file Attention.ixx
 * @brief Multi-Head Attention module (concatenated QKV input).
 *
 * Module delegates compute to a device-specific UnaryOperation implementation
 * that expects a concatenated QKV input.
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

export module Dnn.Components.Attention;
export import :Config;

import Dnn.Component;
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
     * @brief Multi-Head Attention module that accepts concatenated QKV input.
     *
     * The module requires a single input tensor in model-layout containing
     * concatenated Q, K and V along the feature axis:
     *   input shape == [B, T, 3 * embedding_dim]
     *
     * The backend compute implementation (registered as "AttentionOp") must
     * accept the concatenated QKV input and produce an output of shape
     *   output shape == [B, T, embedding_dim]
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Attention : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;

        /**
         * @brief Construct with an existing execution context and config.
         *
         * @param context Shared execution context for device resources.
         * @param config Multi-head attention configuration.
         */
        explicit Attention( std::shared_ptr<ExecutionContextType> context, const AttentionConfig& config )
            : context_( context ), config_( config )
        {
            if (!context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            createOperation();
        }

        ~Attention() override = default;

        

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - delegates to backend operation.
         *
         * Input must be concatenated QKV in model layout: [B, T, 3 * embedding_dim].
         * Output must be the attention result in model layout: [B, T, embedding_dim].
         *
         * @param input Input tensor (concatenated QKV) [B, T, 3 * embedding_dim]
         * @param output Output tensor [B, T, embedding_dim]
         */
        void forward( const ITensor& input, ITensor& output )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Attention module must be built before calling forward." );
            }

            validateForwardShapes( input, output );

            operation_->forward( input, output );
        }

        /**
         * @brief Backward pass - delegates to backend operation.
         *
         * Expects:
         *  - input:     [B, T, 3 * embedding_dim]  (concatenated QKV)
         *  - output_grad: [B, T, embedding_dim]
         *  - input_grad:  [B, T, 3 * embedding_dim]
         *
         * @param input Input tensor (concatenated QKV)
         * @param output_grad Gradient w.r.t. output
         * @param input_grad Gradient w.r.t. input (written)
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Attention module must be built before calling backward." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Attention module must be in training mode to call backward. Call setTraining(true) first." );
            }

            validateBackwardShapes( input, output_grad, input_grad );

            operation_->backward( output_grad, input, input_grad );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            // No trainable parameters in base multi-head attention implementation
        }


        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        std::vector<ITensor*> getParameters() const override
        {
            return {};
        }

        std::vector<ITensor*> getGradients() const override
        {
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
            return context_->getDevice();
        }

        void synchronize() override
        {
            context_->synchronize();
        }

        size_t parameterCount() const override
        {
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

            if (context_ && context_->getDevice())
            {
                oss << "Device: " << deviceTypeToString( context_->getDevice()->getDeviceType() ) << std::endl;
            }

            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

        // ====================================================================
        // Configuration accessors
        // ====================================================================

        int64_t getEmbeddingDim() const noexcept
        {
            return config_.getEmbeddingDim();
        }

        int64_t getNumHeads() const noexcept
        {
            return config_.getNumHeads();
        }

        const AttentionConfig& getConfig() const noexcept
        {
            return config_;
        }

    protected:

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /*bool isBuilt() const override
        {
            return (operation_ != nullptr) && is_built_;
        }*/

        /**
         * @brief Build the module using an input shape.
         *
         * The input_shape must describe the concatenated QKV model-layout:
         *   [B, T, 3 * embedding_dim]
         */
        void onBuilding( const shape_t& input_shape ) override
        {
            validateConcatenatedQKVShape( input_shape );

            operation_->setTraining( this->isTraining() );

            // No learnable parameters in base attention
            operation_->setParameters( nullptr, nullptr );

            // Backend expects concatenated QKV model-layout input shape for build()
            operation_->build( input_shape );
        }

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Inform backend operation of the new training mode. When leaving
         * training, explicitly unbind any parameter-gradient pointers on the
         * backend to avoid accidental use or pinned memory.
         *
         * Called with Module's training mutex held; do not call setTraining() here.
         */
        void onTrainingChanging( bool is_training ) override
        {
            operation_->setTraining( is_training );
        }

    private:
        AttentionConfig config_;
        shape_t input_shape_;

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> context_;

        // Validate concatenated QKV model-layout shape [B, T, 3 * embedding_dim]
        void validateConcatenatedQKVShape( const shape_t& shape ) const
        {
            if (shape.size() != 3)
            {
                throw std::invalid_argument( "Attention: expected 3D model-layout shape" );
            }

            const int64_t trailing = shape.back();
            const int64_t expected = config_.getEmbeddingDim() * 3;

            if (trailing != expected)
            {
                std::ostringstream oss;
                oss << "Attention: expected concatenated QKV trailing dimension " << expected
                    << " (3 * embedding_dim), got " << trailing;
                throw std::invalid_argument( oss.str() );
            }
        }

        void validateForwardShapes( const ITensor& input, const ITensor& output ) const
        {
            const auto& in_shape = input.shape();
            const auto& out_shape = output.shape();

            validateConcatenatedQKVShape( in_shape );

            if (out_shape.size() != 3)
            {
                throw std::invalid_argument( "Attention: output must be 3D model-layout [B, T, embedding_dim]" );
            }

            const int64_t out_trailing = out_shape.back();
            const int64_t expected_out = config_.getEmbeddingDim();

            if (out_trailing != expected_out)
            {
                std::ostringstream oss;
                oss << "Attention: expected output trailing dimension " << expected_out
                    << " (embedding_dim), got " << out_trailing;
                throw std::invalid_argument( oss.str() );
            }

            // Ensure batch and sequence dims match between input and output
            if (in_shape[0] != out_shape[0] || in_shape[1] != out_shape[1])
            {
                throw std::invalid_argument( "Attention: input and output batch/sequence dimensions must match" );
            }
        }

        void validateBackwardShapes(
            const ITensor& input,
            const ITensor& output_grad,
            const ITensor& input_grad ) const
        {
            // input: [B, T, 3*D], output_grad: [B, T, D], input_grad: [B, T, 3*D]
            const auto& in_shape = input.shape();
            validateConcatenatedQKVShape( in_shape );

            const auto& outg_shape = output_grad.shape();
            if (outg_shape.size() != 3 || outg_shape.back() != config_.getEmbeddingDim())
            {
                throw std::invalid_argument( "Attention: output_grad must have model-layout trailing dim == embedding_dim" );
            }

            const auto& ing_shape = input_grad.shape();
            if (ing_shape.size() != 3 || ing_shape.back() != config_.getEmbeddingDim() * 3)
            {
                throw std::invalid_argument( "Attention: input_grad must have model-layout trailing dim == 3 * embedding_dim" );
            }

            // Ensure batch and sequence dims match across tensors
            if (in_shape[0] != outg_shape[0] || in_shape[1] != outg_shape[1] ||
                in_shape[0] != ing_shape[0] || in_shape[1] != ing_shape[1])
            {
                throw std::invalid_argument( "Attention: batch/sequence dimensions must match across input, output_grad and input_grad" );
            }
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "AttentionOp",
                    context_,
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error( "Failed to create Attention Module for (TDeviceType) compute backend operation." );
            }
        }
    };
}