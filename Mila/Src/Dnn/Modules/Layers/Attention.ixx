/**
 * @file Attention.ixx
 * @brief Device-templated Multi-Head Attention module.
 *
 * Delegates compute to a device-specific TernaryOperation implementation registered in the OperationRegistry.
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
import Compute.TernaryOperation;
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
     * Contract:
     *  - Inputs Q, K, V are expected in head-major layout: [B, NH, T, hs]
     *  - Output and gradients are produced in the same head-major layout.
     *
     * This avoids per-call memory reorganization inside the compute backend.
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
        // Lifecycle
        // ====================================================================

        bool isBuilt() const override
        {
            return (operation_ != nullptr) && is_built_;
        }

        /**
         * @brief Build the module using an input shape.
         *
         * The input_shape is interpreted as the head-major query shape:
         *   [B, NH, T, hs]
         *
         * Validates the shape against the configuration and forwards it to the
         * backend operation for any device-specific initialization.
         */
        void build( const shape_t& input_shape ) override
        {
            if (is_built_)
            {
                return;
            }

            validateHeadMajorShape( input_shape );

            operation_->setTraining( is_training_ );

            // No learnable parameters in base attention
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
         * Inputs/outputs are head-major: [B, NH, T, hs].
         *
         * @param Q Query tensor [B, NH, T, hs]
         * @param K Key tensor   [B, NH, T, hs]
         * @param V Value tensor [B, NH, T, hs]
         * @param output Output tensor [B, NH, T, hs]
         */
        void forward( const ITensor& Q, const ITensor& K, const ITensor& V, ITensor& output )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Attention module must be built before calling forward." );
            }

            validateHeadMajorShapes( Q, K, V, output );

            operation_->forward( Q, K, V, output );
        }

        /**
         * @brief Backward pass - delegates to backend operation.
         *
         * Computes gradients for Q, K, V. All tensors are head-major.
         *
         * @param Q Query tensor [B, NH, T, hs]
         * @param K Key tensor [B, NH, T, hs]
         * @param V Value tensor [B, NH, T, hs]
         * @param output_grad Gradient w.r.t. output [B, NH, T, hs]
         * @param q_grad Gradient w.r.t. Q [B, NH, T, hs] (written)
         * @param k_grad Gradient w.r.t. K [B, NH, T, hs] (written)
         * @param v_grad Gradient w.r.t. V [B, NH, T, hs] (written)
         */
        void backward(
            const ITensor& Q,
            const ITensor& K,
            const ITensor& V,
            const ITensor& output_grad,
            ITensor& q_grad,
            ITensor& k_grad,
            ITensor& v_grad )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Attention module must be built before calling backward." );
            }

            if (!is_training_)
            {
                throw std::runtime_error( "Attention module must be in training mode to call backward. Call setTraining(true) first." );
            }

            validateHeadMajorShapesForBackward( Q, K, V, output_grad, q_grad, k_grad, v_grad );

            operation_->backward( Q, K, V, output_grad, q_grad, k_grad, v_grad );
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
            return {};
        }

        std::vector<ITensor*> getParameterGradients() const override
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

        void setTraining( bool is_training ) override
        {
            if (is_training_ == is_training)
            {
                return;
            }

            is_training_ = is_training;

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
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
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

    private:

        AttentionConfig config_;
        bool is_training_{ false };
        bool is_built_{ false };

        std::shared_ptr<TernaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> context_;

        // Validate a head-major shape [B, NH, T, hs] matches config.
        void validateHeadMajorShape( const shape_t& shape ) const
        {
            if (shape.size() != 4)
            {
                throw std::invalid_argument( "Attention: expected head-major shape [B, NH, T, hs]" );
            }

            const int64_t NH = shape[1];
            const int64_t hs = shape[3];

            if (NH != config_.getNumHeads())
            {
                std::ostringstream oss;
                oss << "Attention: number of heads (shape[1]) must equal config.num_heads. Expected "
                    << config_.getNumHeads() << ", got " << NH;
                throw std::invalid_argument( oss.str() );
            }

            if ((NH * hs) != config_.getEmbeddingDim())
            {
                std::ostringstream oss;
                oss << "Attention: NH * hs must equal embedding_dim. Got NH=" << NH
                    << ", hs=" << hs << ", NH*hs=" << (NH * hs)
                    << ", embedding_dim=" << config_.getEmbeddingDim();
                throw std::invalid_argument( oss.str() );
            }
        }

        void validateHeadMajorShapes( const ITensor& Q, const ITensor& K, const ITensor& V, const ITensor& output ) const
        {
            const auto& qshape = Q.shape();
            const auto& kshape = K.shape();
            const auto& vshape = V.shape();
            const auto& oshape = output.shape();

            validateHeadMajorShape( qshape );

            if (kshape != qshape || vshape != qshape)
            {
                throw std::invalid_argument( "Attention: Q, K, V must have identical head-major shapes [B, NH, T, hs]" );
            }

            if (oshape != qshape)
            {
                throw std::invalid_argument( "Attention: output must have same head-major shape as inputs" );
            }
        }

        void validateHeadMajorShapesForBackward(
            const ITensor& Q, const ITensor& K, const ITensor& V,
            const ITensor& output_grad,
            const ITensor& q_grad, const ITensor& k_grad, const ITensor& v_grad ) const
        {
            const auto& qshape = Q.shape();

            validateHeadMajorShape( qshape );

            if (K.shape() != qshape || V.shape() != qshape)
            {
                throw std::invalid_argument( "Attention: Q, K, V must have identical head-major shapes [B, NH, T, hs]" );
            }

            if (output_grad.shape() != qshape)
            {
                throw std::invalid_argument( "Attention: output_grad must have same head-major shape as inputs" );
            }

            if (q_grad.shape() != qshape || k_grad.shape() != qshape || v_grad.shape() != qshape)
            {
                throw std::invalid_argument( "Attention: q_grad/k_grad/v_grad must have same head-major shape as inputs" );
            }
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createTernaryOperation<TDeviceType, TPrecision>(
                    "AttentionOp",
                    context_,
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error(
                    "Failed to create Attention compute backend operation. "
                    "Ensure CPU/CUDA operation is registered in OperationRegistry." );
            }
        }
    };
}