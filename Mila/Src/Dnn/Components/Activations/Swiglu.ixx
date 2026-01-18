/**
 * @file Swiglu.ixx
 * @brief SwiGLU activation component implementation.
 *
 * Device-templated SwiGLU component that delegates compute to a registered
 * device-specific UnaryOperation backend (registered as "SwigluOp").
 */

    module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <format>
#include <utility>
#include <optional>

export module Dnn.Components.Swiglu;
export import :Config;

import Dnn.Components.Gelu;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;
import Serialization.Tensor;
import Serialization.Mode;
import Serialization.Metadata;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief SwiGLU activation component.
     *
     * SwiGLU splits input along the feature axis into two halves x1,x2 and computes:
     *   out = x1 * GELU(x2)
     *
     * Delegates work to a device-specific UnaryOperation named "SwigluOp".
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Swiglu : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using ComponentBase = Component<TDeviceType, TPrecision>;

        explicit Swiglu( const std::string& name, const SwigluConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "Swiglu: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~Swiglu() override = default;

        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Swiglu::forward: component must be built before forward pass" );
            }

            operation_->forward( input, *output_ );

            return *output_;
        }

        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Swiglu::backward: component must be built before backward pass" );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Swiglu::backward: component must be in training mode to compute gradients" );
            }

            zero( *input_grad_ );

            operation_->backward( input, output_grad, *input_grad_ );

            return *input_grad_;
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)mode;

            SerializationMetadata meta;
            meta.set( "type", "Swiglu" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() )
                .set( "template_device", deviceTypeToString( TDeviceType ) )
                .set( "template_precision", static_cast<int64_t>(TPrecision) );

            archive.writeMetadata( "meta.json", meta );

            archive.writeMetadata( "config.json", config_.toMetadata() );
        }

        static std::unique_ptr<Swiglu> fromArchive_(
            ModelArchive& archive,
            const std::string& component_name,
            IExecutionContext* exec_context )
        {
            try
            {
                SerializationMetadata meta = archive.readMetadata( "meta.json" );
                validateMetadata_( meta, component_name );

                SerializationMetadata cfg = archive.readMetadata( "config.json" );
                SwigluConfig config;
                config.fromMetadata( cfg );
                config.validate();

                auto inst = std::make_unique<Swiglu>( component_name, config );
                if ( exec_context )
                {
                    inst->setExecutionContext( exec_context );
                }
                return inst;
            }
            catch ( const std::exception& e )
            {
                throw std::runtime_error( std::format( "Swiglu::fromArchive: error for '{}': {}", component_name, e.what() ) );
            }
        }

        size_t parameterCount() const override {
            return 0;
        }

        std::vector<ITensor*> getParameters() const override {
            return {};
        }

        std::vector<ITensor*> getGradients() const override {
            return {};
        }

        // ====================================================================
        // Identification and Description
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::Swiglu;
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Swiglu: " << this->getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Inner GELU: " << std::string( GeluConfig::toString( config_.getInnerGeluMethod() ) ) << std::endl;
            return oss.str();
        }

    protected:
        void onExecutionContextSet() override
        {
            createOperation();
        }

        void onBuilding( const shape_t& input_shape ) override
        {
            if ( !operation_ )
            {
                throw std::runtime_error( std::format( "Swiglu::onBuilding: operation backend not initialized for '{}'", this->getName() ) );
            }

            operation_->build( input_shape );
            input_shape_ = input_shape;

            DeviceId dev_id = this->getExecutionContext()->getDeviceId();

            output_ = std::make_unique<TensorType>( dev_id, input_shape_ );
            input_grad_ = std::make_unique<TensorType>( dev_id, input_shape_ );
            zero( *input_grad_ );
        }

        void onTrainingChanging( bool is_training ) override
        {
            if ( operation_ ) operation_->setTraining( is_training );
        }

    private:
        SwigluConfig config_;
        shape_t input_shape_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        std::unique_ptr<TensorType> output_{ nullptr };
        std::unique_ptr<TensorType> input_grad_{ nullptr };

        static void validateMetadata_( const SerializationMetadata& meta, const std::string& component_name )
        {
            int64_t version = meta.tryGetInt( "version" ).value_or( 0 );
            if ( version != 1 )
            {
                throw std::runtime_error( std::format( "Swiglu: unsupported version {} for '{}'", version, component_name ) );
            }

            std::string type = meta.tryGetString( "type" ).value_or( "" );
            if ( type != "Swiglu" )
            {
                throw std::runtime_error( std::format( "Swiglu: type mismatch for '{}': expected 'Swiglu', got '{}'", component_name, type ) );
            }

            std::string file_device = meta.tryGetString( "template_device" ).value_or( "" );
            int64_t file_precision = meta.tryGetInt( "template_precision" ).value_or( -1 );

            std::string expected_device = deviceTypeToString( TDeviceType );
            int64_t expected_precision = static_cast<int64_t>(TPrecision);

            if ( file_device != expected_device )
            {
                throw std::runtime_error( std::format( "Swiglu: device mismatch for '{}': archive='{}', expected='{}'", component_name, file_device, expected_device ) );
            }

            if ( file_precision != expected_precision )
            {
                throw std::runtime_error( std::format( "Swiglu: precision mismatch for '{}': archive={}, expected={}", component_name, file_precision, expected_precision ) );
            }
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>( "SwigluOp", this->getExecutionContext(), config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( std::format( "Swiglu: Failed to create compute backend operation for component '{}'", this->getName() ) );
            }
        }
    };
}