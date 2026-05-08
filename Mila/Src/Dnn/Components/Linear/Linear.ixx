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
#include <numeric>
#include <algorithm>

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
import Compute.Precision;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContextFactory;
import Compute.IExecutionContext;
import Compute.LinearOpTypeMap;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
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
    using json = nlohmann::json;

    /**
     * @brief Device-templated fully connected (linear) component.
     *
     * Delegates compute to a device-specific operation resolved at compile time via
     * LinearOpTypeMap<TDeviceType, TPrecision, TWeight>. TWeight defaults to TPrecision
     * for standard paths. When TWeight differs from TPrecision, weights are stored in
     * the reduced precision format and quantized from the source blob dtype on load.
     *
     * Quantization is performed once at load time (quantize-on-load). The backend
     * operation receives the quantized weight tensor directly and is responsible for
     * any dequantization required during the GEMM.
     *
     * @tparam TDeviceType  Target device.
     * @tparam TPrecision   Activation and accumulation precision.
     * @tparam TWeight      Weight storage precision. Defaults to TPrecision.
     */
    export template<DeviceType TDeviceType, TensorDataType TComputePrecision, TensorDataType TWeight = TComputePrecision>
        requires PrecisionSupportedOnDevice<TComputePrecision, TDeviceType>
    class Linear : public Component<TDeviceType, TComputePrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TComputePrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TComputePrecision, MR>;
        using WeightTensorType = Tensor<TWeight, MR>;
        using OperationType = typename Compute::LinearOpTypeMap<TDeviceType, TComputePrecision, TWeight>::op_type;

        static constexpr bool kIsQuantized = (TWeight != TComputePrecision);

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
         * @brief Perform forward pass.
         *
         * Uses a component-owned output buffer (allocated in onBuilding()) and
         * delegates computation to the backend operation.
         *
         * @param input Input tensor (device-bound).
         * @return Reference to the component-owned output tensor.
         *
         * @throws std::runtime_error if component not built or backend not initialized.
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
         * Uses a component-owned input-gradient buffer (allocated in onBuilding())
         * and delegates computation to the backend operation.
         *
         * @param input Original forward input tensor.
         * @param output_grad Gradient with respect to the component output.
         * @return Reference to the component-owned input-gradient tensor.
         *
         * @throws std::runtime_error if component not built or not in training mode.
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
         * @param archive ModelArchive to write to (scoped by caller).
         * @param mode Serialization mode (currently unused).
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
            oss << "Weight dtype: " << tensorDataTypeToString( TWeight ) << std::endl;
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
         * For the weight parameter, when TWeight == TPrecision the blob is copied
         * directly via loadParameterFromBlob. When TWeight != TPrecision (quantized
         * path) the blob dtype must be TPrecision (the source float type) and the
         * values are cast element-wise into the TWeight storage tensor via
         * quantizeFromBlob. Bias is always stored at TPrecision.
         *
         * @param name Parameter name: "weight" or "bias".
         * @param blob Serialized tensor blob from PretrainedModelReader.
         *
         * @throws std::invalid_argument on dtype or shape mismatch.
         */
        void loadParameter( const std::string& name, const ITensorBlob& blob ) override
        {
            if ( name == "weight" )
            {
                const shape_t expected_shape{ config_.getOutputFeatures(), config_.getInputFeatures() };

                if constexpr ( kIsQuantized )
                {
                    this->quantizeFromBlob( "weight", blob, *weight_, expected_shape );
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
                this->loadParameter( name, blob );
            }
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            if ( weight_ != nullptr )
            {
                stats.device_parameter_bytes += weight_->getStorageSize();
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
        std::shared_ptr<OperationType> operation_{ nullptr };

        // Weight storage is TWeight — differs from TComputePrecision on quantized paths.
        std::shared_ptr<WeightTensorType> weight_{ nullptr };
        // TODO: std::unique_ptr<WeightScaleTensorType> weight_scales_{ nullptr };


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
                    "Linear '{}': input features mismatch — expected {}, got {}",
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

            weight_ = std::make_shared<WeightTensorType>(
                device, shape_t{ output_features, input_features }, this->getName() + ".weight" );

            if ( context.shouldInitializeParameters() )
            {
                // FIXME: xavier<TWeight, MR>( *weight_, input_features, output_features );
            }

            if ( config_.hasBias() )
            {
                bias_ = std::make_shared<TensorType>(
                    device, shape_t{ output_features }, this->getName() + ".bias" );

                if ( context.shouldInitializeParameters() )
                {
                    // FIXME: zero( *bias_ );
                }
            }
        }

        /**
         * @brief Instantiate the backend compute operation via compile-time traits dispatch.
         *
         * OpType is resolved by LinearOpTypeMap at instantiation time — no registry lookup,
         * no string key, no runtime hash map. A missing specialization is a compile error.
         */
        void createOperation()
        {
            operation_ = std::make_shared<OperationType>(
                this->getExecutionContext(), config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Linear: failed to create operation." );
            }
        }
    };
}