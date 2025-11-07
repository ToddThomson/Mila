/**
 * @file SoftmaxCrossEntropy.ixx
 * @brief Device-templated fused SoftmaxCrossEntropy loss module.
 *
 * Delegates compute to a UnaryOperation backend that implements the fused
 * softmax + cross-entropy operation for numerical stability and performance.
 *
 * Key properties:
 * - Numerically stable: uses log-sum-exp trick
 * - Efficient: single kernel pass, no materialized probabilities
 * - Simple gradient: dL/dlogits = softmax(logits) - one_hot(targets)
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

export module Dnn.Modules.SoftmaxCrossEntropy;
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
     * @brief Fused SoftmaxCrossEntropy loss module (device-templated).
     *
     * Delegates computation to a device-specific UnaryOperation implementation
     * registered in the OperationRegistry.
     *
     * Computes the fused softmax + cross-entropy loss between predicted logits
     * and target class indices in a single numerically stable operation.
     *
     * Operation:
     * - Forward: loss = -log(softmax(logits)[target])
     * - Backward: dL/dlogits = softmax(logits) - one_hot(targets)
     *
     * Invalid target handling (automatic):
     * - Targets < 0 or >= vocab_size are ignored (loss = 0, grad = 0)
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType) for logits
     * @tparam TTargets Data type for target indices (typically int32)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision, TensorDataType TTargets = dtype_t::INT32>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class SoftmaxCrossEntropy : public Module<TDeviceType>
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

            createOperation();
        }

        ~SoftmaxCrossEntropy() override = default;

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
         * Validates input shape and triggers backend-specific setup.
         * The fused operation has no trainable parameters.
         */
        void build( const shape_t& input_shape ) override
        {
            if (is_built_)
                return;

            validateInputShape( input_shape );

            operation_->setTraining( is_training_ );
            operation_->build( input_shape );

            is_built_ = true;
        }

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - delegates to backend operation.
         *
         * Computes fused softmax + cross-entropy loss.
         *
         * @param input Input tensor containing predicted logits [B, S, V]
         * @param targets Target tensor containing class indices [B, S]
         * @param output Output tensor for loss values [B, S]
         */
        void forward( const ITensor& input, const ITensor& targets, ITensor& output )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "SoftmaxCrossEntropy module must be built before calling forward." );
            }

            validateInputShape( input );

            // Cast to concrete tensor types for operation
            const auto& input_typed = dynamic_cast<const TensorType&>(input);
            const auto& targets_typed = dynamic_cast<const TargetTensorType&>(targets);
            auto& output_typed = dynamic_cast<TensorType&>(output);

            operation_->forward( input_typed, targets_typed, output_typed );
        }

        /**
         * @brief Backward pass - delegates to backend operation.
         *
         * Computes fused gradient: dL/dlogits = softmax(logits) - one_hot(targets)
         *
         * @param input Input tensor from forward pass (logits) [B, S, V]
         * @param targets Target tensor containing class indices [B, S]
         * @param output_grad Gradient of loss with respect to output [B, S]
         * @param input_grad Tensor to store gradients with respect to input [B, S, V]
         */
        void backward( const ITensor& input, const ITensor& targets, const ITensor& output_grad, ITensor& input_grad )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "SoftmaxCrossEntropy module must be built before calling backward." );
            }

            if (!is_training_)
            {
                throw std::runtime_error( "SoftmaxCrossEntropy module must be in training mode to call backward. Call setTraining(true) first." );
            }

            // Cast to concrete tensor types
            const auto& input_typed = dynamic_cast<const TensorType&>(input);
            const auto& targets_typed = dynamic_cast<const TargetTensorType&>(targets);
            const auto& output_grad_typed = dynamic_cast<const TensorType&>(output_grad);
            auto& input_grad_typed = dynamic_cast<TensorType&>(input_grad);

            // No parameters or parameter gradients for fused operation
            std::vector<std::shared_ptr<ITensor>> empty_params;
            std::vector<std::shared_ptr<ITensor>> empty_param_grads;

            operation_->backward(
                input_typed,
                targets_typed,
                empty_params,
                empty_param_grads,
                output_grad_typed,
                input_grad_typed,
                output_state_ );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save( ModelArchive& archive ) const override
        {
            // No parameters to save for fused softmax+cross-entropy
        }

        void load( ModelArchive& archive ) override
        {
            // No parameters to load for fused softmax+cross-entropy
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        std::vector<ITensor*> getParameters() const override
        {
            // No trainable parameters
            return {};
        }

        std::vector<ITensor*> getParameterGradients() const override
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

        /**
         * @brief Gets the vocabulary size.
         */
        int64_t getVocabSize() const
        {
            return config_.getVocabSize();
        }

        /**
         * @brief Get the configuration.
         */
        const CrossEntropyConfig& getConfig() const noexcept
        {
            return config_;
        }

    private:
        CrossEntropyConfig config_;
        bool is_training_{ false };
        bool is_built_{ false };

        std::vector<std::shared_ptr<TensorType>> output_state_;  ///< Cached probabilities for backward pass
        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

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
         *
         * Validates that last dimension matches vocab_size.
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.size() < 2)
            {
                throw std::invalid_argument(
                    "SoftmaxCrossEntropy: input must have rank >= 2 (batch, ..., vocab)" );
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
         *
         * Looks up the appropriate device-specific fused operation from the registry
         * and creates an instance bound to this module's execution context.
         *
         * Registered operation name: "SoftmaxCrossEntropyOp"
         */
        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision, TTargets>(
                    "SoftmaxCrossEntropyOp",
                    exec_context_,
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error(
                    "Failed to create SoftmaxCrossEntropy compute backend operation. "
                    "Ensure CPU/CUDA operation is registered in OperationRegistry." );
            }
        }
    };

    //// Convenience aliases for common usages
    //export template<TensorDataType TPrecision, typename TTargets = int32_t>
    //    using CpuSoftmaxCrossEntropy = SoftmaxCrossEntropy<DeviceType::Cpu, TPrecision, TTargets>;

    //export template<TensorDataType TPrecision, typename TTargets = int32_t>
    //    using CudaSoftmaxCrossEntropy = SoftmaxCrossEntropy<DeviceType::Cuda, TPrecision, TTargets>;
}