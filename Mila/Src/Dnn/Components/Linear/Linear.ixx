/**
 * @file Linear.ixx
 * @brief Device-templated Linear (fully connected) component.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>

export module Dnn.Components.Linear;
export import Dnn.Components.LinearConfig;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Dnn.Quantization.Weight.Policies;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContextFactory;
import Compute.IExecutionContext;
import Compute.OperationType;
import Compute.OperationTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;

import Serialization.ModelArchive;
import Serialization.Mode;
import Serialization.Tensor;
import Serialization.Metadata;
import nlohmann.json;

import Dnn.TensorOps;
import Dnn.TensorHelpers;
import Logging.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;
    using json = nlohmann::json;

    /**
     * @brief Device-templated fully connected (linear) component.
     *
     * Delegates compute to a device-specific operation resolved at compile time via
     * OperationTraits<LinearOp, TDeviceType, TComputePrecision, TWeightQuant>. TWeightQuant
     * defaults to NoWeightQuant for unquantized paths.
     *
     * When TWeightQuant::kIsQuantized is true, the weight tensor is allocated at the
     * reduced-precision storage dtype (kWeightDtype = TWeightQuant::kStorageDtype) rather
     * than TComputePrecision. Per-channel FP32 scale factors (weight_scales_) are allocated
     * alongside the weight tensor and bound to the backend operation via setWeightScales()
     * before the first forward pass. The backend operation receives both the quantized weight
     * tensor and its scales and is responsible for dequantization during the GEMM.
     *
     * Weight quantization is performed once at model load time (quantize-on-load) during
     * loadParameter(). The source checkpoint blob is always at TComputePrecision.
     *
     * @tparam TDeviceType        Target device.
     * @tparam TComputePrecision  Activation and accumulation precision.
     * @tparam TWeightQuant       Weight quantization policy. Must satisfy WeightQuantPolicy.
     *                            Defaults to NoWeightQuant (identity -- no quantization).
     */
    export template<DeviceType TDeviceType, TensorDataType TComputePrecision, WeightQuantPolicy TWeightQuant = NoWeightQuant>
        requires PrecisionSupportedOnDevice<TComputePrecision, TDeviceType>
    class Linear : public Component<TDeviceType, TComputePrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TComputePrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TComputePrecision, MR>;
        using OpType = typename OperationTraits<OperationType::LinearOp, TDeviceType, TComputePrecision, TWeightQuant>::type;

        // LinearOpConcept<OpType, TensorType> is intentionally NOT enforced via static_assert
        // here. Placing a static_assert in the class template body forces MSVC to fully
        // instantiate OpType (including all member function bodies) before the CUDA execution
        // context is complete. A missing OperationTraits specialization already produces a hard
        // compile error on ::type -- that is the practical guard.

        static constexpr bool kIsQuantized = TWeightQuant::kIsQuantized;

        static constexpr TensorDataType kWeightDtype = kIsQuantized
            ? TWeightQuant::kStorageDtype : TComputePrecision;

        using WeightTensorType      = Tensor<kWeightDtype, MR>;
        using WeightScaleTensorType = Tensor<TWeightQuant::kScaleDtype, MR>;

        /**
         * @brief Construct a Linear component.
         *
         * Constructs with a name and configuration. If `device_id` is provided,
         * the component creates and owns an ExecutionContext (standalone mode)
         * and registers it with the base Component via setExecutionContext().
         * If `device_id` is not provided, the component expects a shared
         * ExecutionContext to be provided later via setExecutionContext().
         *
         * @param name Component name.
         * @param config Layer configuration (validated on construction).
         * @param device_id Optional device identifier. When present the component
         *                  creates an owned ExecutionContext for the device.
         *
         * @throws std::invalid_argument if config is invalid or device type mismatches.
         * @throws std::runtime_error if ExecutionContext creation fails.
         */
        explicit Linear( const std::string& name, const LinearConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "Linear: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );

                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~Linear() override = default;

        /**
         * @brief Perform forward pass: output = input * weight^T + bias.
         *
         * Delegates to the backend operation using the component-owned output buffer
         * allocated at build time. When the runtime input shape differs from the
         * build-time shape (e.g. a shorter decode sequence vs. the prefill shape),
         * a lightweight view over the output buffer is returned that reflects the
         * true output shape without reallocating device memory.
         *
         * @param input Input tensor (device-bound, rank >= 2). The last dimension
         *              must equal the configured input feature count.
         * @return      Reference to the output tensor or a shape-adjusted view of it.
         *
         * @throws std::runtime_error    if the component has not been built.
         * @throws std::invalid_argument if the input feature dimension does not match the config.
         */
        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Linear must be built before calling forward." );
            }

            validateInputShape( input.shape() );

            operation_->forward( input, *output_ );

            auto input_shape = input.shape();

            TensorType* result = nullptr;

            if ( input_shape == leading_shape_ )
            {
                result = output_.get();
            }
            else
            {
                auto output_shape = input_shape;
                output_shape.back() = config_.getOutputFeatures();
                output_view_ = std::make_unique<TensorType>( output_->view( output_shape ) );

                result = output_view_.get();
            }

            return *result;
        }

        /**
         * @brief Perform backward pass.
         *
         * Pre-zeros the component-owned input gradient buffer, then delegates to the
         * backend operation. The backend accumulates weight and bias gradients into the
         * buffers bound via setGradients() using += semantics; pre-zeroing ensures
         * clean gradient state across calls.
         *
         * Not supported on quantized paths (kIsQuantized == true) -- the backend
         * operation will throw std::logic_error if backward is attempted.
         *
         * @param input       Original forward-pass input tensor.
         * @param output_grad Upstream gradient tensor (same shape as the forward output).
         * @return            Reference to the component-owned input gradient tensor.
         *
         * @throws std::runtime_error if the component has not been built.
         * @throws std::runtime_error if called while in inference (eval) mode.
         */
        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Linear::backward: must be built" );
            }

            if ( this->isInferenceMode() )
            {
                throw std::runtime_error( "Linear::backward: must be in training mode" );
            }

            // Zero before backward: backend ops use accumulation semantics (+=) and
            // require pre-zeroed buffers to prevent gradient buildup across calls.
            zero( *input_grad_ );

            operation_->backward( input, output_grad, *input_grad_ );

            return *input_grad_;
        }

        void zeroGradients() override
        {
            if ( weight_grad_ )
            {
                zero( *weight_grad_ );
            }

            if ( config_.hasBias() && bias_grad_ )
            {
                zero( *bias_grad_ );
            }

            if ( input_grad_ )
            {
                zero( *input_grad_ );
            }
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @brief Save component state to a ModelArchive.
         *
         * Writes a "meta.json" blob with component type and name, a "config.json"
         * blob with input/output feature dimensions and bias flag, and raw tensor
         * blobs for the weight and (if present) bias parameters under "tensors/".
         *
         * On CUDA devices, each tensor is copied to a temporary host buffer before
         * writing. Weight is serialized at its storage dtype (kWeightDtype), which
         * equals kWeightDtype = TWeightQuant::kStorageDtype on the quantized path.
         *
         * @param archive ModelArchive to write to (scoped by caller).
         * @param mode    Serialization mode (currently unused; reserved for future use).
         */
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)mode;

            SerializationMetadata meta;
            meta.set( "type", "Linear" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() );

            archive.writeMetadata( "meta.json", meta );

            SerializationMetadata cfg;
            cfg.set( "input_features", config_.getInputFeatures() )
                .set( "output_features", config_.getOutputFeatures() )
                .set( "has_bias", config_.hasBias() );

            archive.writeMetadata( "config.json", cfg );

            if ( weight_ )
            {
                TensorMetadata tmeta;
                tmeta.dtype = weight_->getDataType();
                tmeta.shape = weight_->shape();
                tmeta.total_bytes = static_cast<size_t>(weight_->size()) * weight_->elementSize();

                if constexpr ( std::is_same_v<MR, CpuMemoryResource> )
                {
                    const void* data_ptr = weight_->rawData();
                    writeTensorBlob( archive, "tensors/weight", tmeta, data_ptr, tmeta.total_bytes );
                }
                else
                {
                    using HostTensorType = Tensor<dtype_t::FP32, CpuMemoryResource>;
                    HostTensorType host_weight( Device::Cpu(), weight_->shape() );

                    copy( *weight_, host_weight );

                    const void* host_ptr = host_weight.rawData();
                    writeTensorBlob( archive, "tensors/weight", tmeta, host_ptr, tmeta.total_bytes );
                }
            }

            if ( config_.hasBias() && bias_ )
            {
                TensorMetadata bmeta;
                bmeta.dtype = bias_->getDataType();
                bmeta.shape = bias_->shape();
                bmeta.total_bytes = static_cast<size_t>(bias_->size()) * bias_->elementSize();

                if constexpr ( std::is_same_v<MR, CpuMemoryResource> )
                {
                    const void* data_ptr = bias_->rawData();
                    writeTensorBlob( archive, "tensors/bias", bmeta, data_ptr, bmeta.total_bytes );
                }
                else
                {
                    using HostTensorType = Tensor<dtype_t::FP32, CpuMemoryResource>;
                    HostTensorType host_bias( Device::Cpu(), bias_->shape() );

                    copy( *bias_, host_bias );

                    const void* host_ptr = host_bias.rawData();
                    writeTensorBlob( archive, "tensors/bias", bmeta, host_ptr, bmeta.total_bytes );
                }
            }
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        size_t parameterCount() const override
        {
            size_t count = 0;

            if ( weight_ )
            {
                count += weight_->size();
            }

            if ( config_.hasBias() && bias_ )
            {
                count += bias_->size();
            }

            return count;
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::Linear;
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Linear: " << this->getName() << std::endl;
            oss << "Input features: " << config_.getInputFeatures();
            oss << ", Output features: " << config_.getOutputFeatures() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Has Bias: " << (config_.hasBias() ? "Yes" : "No") << std::endl;
            oss << "Weight dtype: " << tensorDataTypeToString( kWeightDtype ) << std::endl;
            oss << "Quantized: " << (kIsQuantized ? "Yes" : "No") << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

        bool hasBias() const noexcept
        {
            return config_.hasBias();
        }

        const LinearConfig& getConfig() const noexcept
        {
            return config_;
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> params;

            if ( weight_ )
            {
                params.push_back( weight_.get() );
            }

            if ( bias_ )
            {
                params.push_back( bias_.get() );
            }

            return params;
        }

        std::vector<ITensor*> getGradients() const override
        {
            std::vector<ITensor*> grads;

            if ( weight_grad_ )
            {
                grads.push_back( weight_grad_.get() );
            }

            if ( bias_grad_ )
            {
                grads.push_back( bias_grad_.get() );
            }

            return grads;
        }

        /**
         * @brief Load a named parameter from a serialized blob.
         *
         * Weight loading dispatches at compile time on kIsQuantized:
         *
         *   - Unquantized path (kIsQuantized == false): the blob is validated against
         *     TComputePrecision and copied directly into the weight tensor via
         *     loadParameterFromBlob.
         *
         *   - Quantized path (kIsQuantized == true): the blob dtype must be
         *     TComputePrecision (the full-precision source type). The backend operation's
         *     quantize() method performs per-channel absmax scale computation, quantizes
         *     weights from TComputePrecision to kWeightDtype (e.g. BF16 -> FP8_E4M3),
         *     and uploads both the quantized weights and FP32 scales to device.
         *     The weight_scales_ tensor was pre-allocated in initializeParameters() and
         *     its device pointer was already bound to the operation in onBuilding() via
         *     setWeightScales() -- quantize() writes directly into that allocation.
         *
         * Bias is always stored and loaded at TComputePrecision regardless of TWeightQuant.
         *
         * @param name Parameter name: "weight" or "bias".
         * @param blob Serialized tensor blob from PretrainedModelReader.
         *
         * @throws std::invalid_argument if the blob dtype does not match the expected
         *         source precision, if the blob shape does not match the config, or if
         *         name is neither "weight" nor "bias".
         */
        void loadParameter( const std::string& name, const ITensorBlob& blob ) override
        {
            if ( name == "weight" )
            {
                const shape_t expected_shape{ config_.getOutputFeatures(), config_.getInputFeatures() };

                if constexpr ( kIsQuantized )
                {
                    operation_->quantize( blob, *weight_, *weight_scales_, expected_shape );
                }
                else
                {
                    this->loadParameterFromBlob( "weight", blob, *weight_, expected_shape );
                }
            }
            else if ( name == "bias" )
            {
                if ( !hasBias() )
                {
                    return;
                }

                const shape_t expected_shape{ config_.getOutputFeatures() };
                this->loadParameterFromBlob( "bias", blob, *bias_, expected_shape );
            }
            else
            {
                throw std::invalid_argument( std::format(
                    "Linear '{}': unknown parameter '{}' (expected 'weight' or 'bias')",
                    this->getName(), name ) );
            }
        }

        /**
         * @brief Replace the owned weight with a shared tensor (e.g. a tied lm_head
         *        sharing the token embedding table). See WeightTying.md.
         *
         * Must be called after the component is built so operation_ is live. The
         * kIsQuantized guard documents the invariant that a tied head is unquantized
         * (D4); in practice an lm_head is never quantized, so this branch is
         * unreachable in the real model set.
         *
         * @param shared_weight Shared device tensor; must match the configured shape.
         */
        void installSharedWeight( std::shared_ptr<WeightTensorType> shared_weight )
        {
            if constexpr ( kIsQuantized )
            {
                throw std::logic_error( std::format(
                    "Linear '{}': installSharedWeight requires an unquantized lm_head; "
                    "tied weights and per-tensor quantization are mutually exclusive",
                    this->getName() ) );
            }
            else
            {
                weight_ = std::move( shared_weight );
                operation_->setParameters( weight_.get(), bias_.get() );
            }
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            if ( weight_ != nullptr )
            {
                stats.device_parameter_bytes += weight_->getStorageSize();
            }

            if ( weight_scales_ != nullptr )
            {
                stats.device_parameter_bytes += weight_scales_->getStorageSize();
            }

            if ( bias_ != nullptr )
            {
                stats.device_parameter_bytes += bias_->getStorageSize();
            }

            if ( output_ != nullptr )
            {
                stats.device_state_bytes += output_->getStorageSize();
            }

            if ( input_grad_ != nullptr )
            {
                stats.device_gradient_bytes += input_grad_->getStorageSize();
            }

            if ( weight_grad_ != nullptr )
            {
                stats.device_gradient_bytes += weight_grad_->getStorageSize();
            }

            if ( bias_grad_ != nullptr )
            {
                stats.device_gradient_bytes += bias_grad_->getStorageSize();
            }

            return stats;
        }

    protected:

        void onExecutionContextSet() override
        {
            createOperation();
        }

        void onBuilding( const BuildContext& context ) override
        {
            validateBuildContext( context );

            const auto& input_shape = context.inputShape();

            initializeParameters( context );

            operation_->setParameters( weight_.get(), bias_.get() );

            if constexpr ( kIsQuantized )
            {
                operation_->setWeightScales( weight_scales_.get() );
            }

            operation_->build( context );

            auto device_id = this->getExecutionContext()->getDeviceId();

            shape_t output_shape = input_shape;
            output_shape.back() = config_.getOutputFeatures();
            output_ = std::make_unique<TensorType>( device_id, output_shape, this->getName() + ".output" );

            if ( context.isTrainingMode() )
            {
                initializeGradients();
                operation_->setGradients( weight_grad_.get(), bias_grad_.get() );

                input_grad_ = std::make_unique<TensorType>( device_id, input_shape, this->getName() + ".input_grad" );
                zero( *input_grad_ );
            }
        }

        void onTrainingModeChanging( TrainingMode mode ) override
        {
            operation_->setTrainingMode( mode );

            if ( mode == TrainingMode::Eval )
            {
                operation_->clearGradients();

                if ( weight_grad_ )
                {
                    zero( *weight_grad_ );
                }

                if ( bias_grad_ )
                {
                    zero( *bias_grad_ );
                }
            }
            else
            {
                if ( weight_grad_ )
                {
                    operation_->setGradients( weight_grad_.get(), bias_grad_.get() );
                }
            }
        }

    private:

        LinearConfig config_;
        shape_t leading_shape_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<OpType> operation_{ nullptr };

        // Weight storage dtype is kWeightDtype -- equals TComputePrecision on the unquantized
        // path; equals TWeightQuant::kStorageDtype (e.g. FP8_E4M3) on the quantized path.
        std::shared_ptr<WeightTensorType> weight_{ nullptr };

        // Per-channel FP32 absmax scales [output_features]. Non-null only on the quantized
        // path (kIsQuantized == true). Allocated in initializeParameters(), bound to the
        // backend operation in onBuilding() via setWeightScales(), and filled at load time
        // by operation_->quantize() inside loadParameter().
        std::unique_ptr<WeightScaleTensorType> weight_scales_{ nullptr };

        // Bias always stored at activation precision.
        std::shared_ptr<TensorType> bias_{ nullptr };

        std::shared_ptr<TensorType> weight_grad_{ nullptr };
        std::shared_ptr<TensorType> bias_grad_{ nullptr };

        std::unique_ptr<TensorType> output_{ nullptr };
        std::unique_ptr<TensorType> output_view_{ nullptr };
        std::unique_ptr<TensorType> input_grad_{ nullptr };

        void validateBuildContext( const BuildContext& context ) const
        {
            const auto& input_shape = context.inputShape();

            if ( input_shape.size() < 2 )
            {
                throw std::invalid_argument( std::format(
                    "Linear '{}': input must be at least rank 2, got rank {}",
                    this->getName(), input_shape.size() ) );
            }

            if ( input_shape.back() != config_.getInputFeatures() )
            {
                throw std::invalid_argument( std::format(
                    "Linear '{}': input features mismatch -- expected {}, got {}",
                    this->getName(), config_.getInputFeatures(), input_shape.back() ) );
            }
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.empty() )
            {
                throw std::invalid_argument( "Linear: input must have rank >= 1" );
            }

            if ( input_shape.back() != config_.getInputFeatures() )
            {
                throw std::invalid_argument(
                    std::format( "Linear: input feature dimension mismatch. Expected {}, got {}",
                        config_.getInputFeatures(), input_shape.back() ) );
            }
        }

        void initializeGradients()
        {
            auto device_id = this->getExecutionContext()->getDeviceId();

            if ( !weight_grad_ )
            {
                weight_grad_ = std::make_shared<TensorType>(
                    device_id, weight_->shape(), this->getName() + ".weight.grad" );
                zero( *weight_grad_ );
            }

            if ( config_.hasBias() && !bias_grad_ )
            {
                bias_grad_ = std::make_shared<TensorType>(
                    device_id, bias_->shape(), this->getName() + ".bias.grad" );
                zero( *bias_grad_ );
            }
        }

        void initializeParameters( const BuildContext& context )
        {
            int64_t input_features = config_.getInputFeatures();
            int64_t output_features = config_.getOutputFeatures();
            auto device = this->getExecutionContext()->getDeviceId();

            // Packed nibble formats (INT4, FP4 E2M1) store 2 elements per UINT8 byte,
            // so the physical column count is input_features/2.
            // FP8 and unquantized formats store one element per storage byte -- full width.
            const int64_t weight_cols = ( kIsQuantized && !TWeightQuant::kPerChannel )
                ? input_features / 2
                : input_features;

            weight_ = std::make_shared<WeightTensorType>(
                device, shape_t{ output_features, weight_cols }, this->getName() + ".weight" );

            if constexpr ( kIsQuantized )
            {
                if constexpr ( TWeightQuant::kPerChannel )
                {
                    // Per-channel: one scale per output channel -- shape [out_features].
                    weight_scales_ = std::make_unique<WeightScaleTensorType>(
                        device, shape_t{ output_features }, this->getName() + ".weight.scales" );
                }
                else
                {
                    // Per-group: one scale per (output channel, K-group) -- shape [out_features, K/group_size].
                    const int64_t num_groups = input_features / TWeightQuant::kQuantizationGroupSize;
                    weight_scales_ = std::make_unique<WeightScaleTensorType>(
                        device, shape_t{ output_features, num_groups }, this->getName() + ".weight.scales" );
                }
            }

            if constexpr ( !kIsQuantized )
            {
                if ( context.shouldInitializeParameters() )
                {
                    // REVIEW: Let's get rid of these ugly static_casts. we should be using dim_t everywhere for shape dimensions,
                    // and the config should be updated to reflect that.

                    xavier( *weight_, static_cast<size_t>( input_features ), static_cast<size_t>( output_features ), this->getExecutionContext() );
                }
            }

            if ( config_.hasBias() )
            {
                bias_ = std::make_shared<TensorType>(
                    device, shape_t{ output_features }, this->getName() + ".bias" );

                if ( context.shouldInitializeParameters() )
                {
                    zero( *bias_, this->getExecutionContext() );
                }
            }
        }

        /**
         * @brief Instantiate the backend compute operation via compile-time traits dispatch.
         *
         * OpType is resolved by OperationTraits at instantiation time -- no registry lookup,
         * no string key, no runtime hash map. A missing specialization is a compile error.
         */
        void createOperation()
        {
            operation_ = std::make_shared<OpType>(
                this->getExecutionContext(), config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Linear: failed to create operation." );
            }
        }
    };
}
