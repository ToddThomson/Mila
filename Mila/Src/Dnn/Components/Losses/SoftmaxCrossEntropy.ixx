/**
 * @file SoftmaxCrossEntropy.ixx
 * @brief Device-templated fused SoftmaxCrossEntropy loss module.
 *
 * Delegates compute to a UnaryOperation backend that implements the fused
 * softmax + cross-entropy operation for numerical stability and performance.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <cstdint>
#include <stdexcept>

export module Dnn.Components.SoftmaxCrossEntropy;
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
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Fused SoftmaxCrossEntropy loss module (device-templated).
     *
     * Delegates computation to a device-specific UnaryOperation implementation
     * registered in the OperationRegistry.
     */
    export template<DeviceType TDeviceType, TensorDataType TLogits, TensorDataType TTargets = dtype_t::INT32, TensorDataType TPrecision = TLogits>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class SoftmaxCrossEntropy : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;
        using TargetTensorType = Tensor<TTargets, MR>;

        /**
         * @brief Construct with an existing execution context.
         *
         * @param exec_context Shared execution context for device resources.
         * @param config CrossEntropy configuration (vocab_size required).
         */
        explicit SoftmaxCrossEntropy( std::shared_ptr<ExecutionContextType> exec_context, const CrossEntropyConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            // Create dummy tensor for unused target gradients in backward pass
            dummy_target_grad_ = std::make_shared<TargetTensorType>( exec_context_->getDevice(), shape_t{ 0 } );

            createOperation();
        }

        ~SoftmaxCrossEntropy() override = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Build the module using an input shape.
         *
         * Validates input shape and triggers backend-specific setup.
         * The fused operation has no trainable parameters.
         */
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            // Ensure backend is aware of the current training mode
            if (operation_)
            {
                operation_->setTraining( this->isTraining() );
            }

            operation_->build( input_shape );
        }

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - delegates to backend operation.
         *
         * Computes fused softmax + cross-entropy loss.
         */
        void forward( const ITensor& logits, const ITensor& targets, ITensor& output )
        {
            if (!this->isBuilt())
            {
                throw std::runtime_error( "SoftmaxCrossEntropy module must be built before calling forward." );
            }

            validateInputShape( logits );

            operation_->forward( logits, targets, output );
        }

        /**
         * @brief Backward pass - delegates to backend operation.
         *
         * Computes fused gradient: dL/dlogits = softmax(logits) - one_hot(targets)
         */
        void backward(
            const ITensor& logits,
            const ITensor& targets,
            const ITensor& output_grad,
            ITensor& logits_grad )
        {
            if (!this->isBuilt())
            {
                throw std::runtime_error( "SoftmaxCrossEntropy module must be built before calling backward." );
            }

            if (!this->isTraining())
            {
                throw std::runtime_error( "SoftmaxCrossEntropy module must be in training mode to call backward. Call setTraining(true) first." );
            }

            // Targets are discrete class indices (non-differentiable) - pass dummy gradient
            operation_->backward(
                logits,
                targets,
                output_grad,
                logits_grad,
                *dummy_target_grad_ );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)archive;
        }

        /*void load( ModelArchive& archive, SerializationMode mode ) override
        {
            (void)archive;
        }*/

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        std::vector<ITensor*> getParameters() const override
        {
            // No trainable parameters
            return {};
        }

        std::vector<ITensor*> getGradients() const override
        {
            // No trainable parameters
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

        size_t parameterCount() const override
        {
            // No trainable parameters in fused softmax+cross-entropy
            return 0;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "SoftmaxCrossEntropy (fused): " << getName() << std::endl;
            oss << "Vocabulary Size: " << config_.getVocabSize() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Output: Per-sample losses [B, S]" << std::endl;
            oss << "Invalid targets (< 0 or >= vocab_size) are automatically ignored" << std::endl;

            return oss.str();
        }

        // ====================================================================
        // SoftmaxCrossEntropy-specific accessors
        // ====================================================================

        int64_t getVocabSize() const
        {
            return config_.getVocabSize();
        }

        const CrossEntropyConfig& getConfig() const noexcept
        {
            return config_;
        }

    protected:
        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Propagate training mode to the backend fused operation. Called with
         * Module's training mutex held; do not call setTraining() here.
         */
        void onTrainingChanging( bool newMode ) override
        {
            operation_->setTraining( newMode );
        }

    private:
        CrossEntropyConfig config_;

        std::shared_ptr<BinaryOperation<TDeviceType, TLogits, TTargets, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

        std::shared_ptr<TargetTensorType> dummy_target_grad_{ nullptr };

        /**
         * @brief Validate input shape for fused softmax+cross-entropy operation.
         */
        void validateInputShape( const ITensor& input ) const
        {
            const auto& input_shape = input.shape();
            validateInputShape( input_shape );
        }

        /**
         * @brief Validate input shape for fused softmax+cross-entropy operation.
         *
         * Expected shapes:
         * - Input logits: [B, S, V] or [B, V]
         * - Targets: [B, S] or [B]
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.size() < 2)
            {
                throw std::invalid_argument( "SoftmaxCrossEntropy: input must have rank >= 2 (batch, ..., vocab)" );
            }

            int64_t vocab_dim = input_shape.back();

            if (vocab_dim != config_.getVocabSize())
            {
                std::ostringstream oss;
                oss << "SoftmaxCrossEntropy: vocabulary dimension mismatch. Expected "
                    << config_.getVocabSize() << ", got " << vocab_dim;
                throw std::invalid_argument( oss.str() );
            }
        }

        /**
         * @brief Create the backend compute operation.
         */
        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createBinaryOperation<TDeviceType, TLogits, TTargets, TPrecision>(
                    "SoftmaxCrossEntropyOp",
                    exec_context_,
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error(
                    "Failed to create SoftmaxCrossEntropy compute backend operation. "
                    "Ensure CPU/CUDA operation is registered in OperationRegistry." );
            }

            // Ensure backend knows current training mode
            operation_->setTraining( this->isTraining() );
        }
    };
}