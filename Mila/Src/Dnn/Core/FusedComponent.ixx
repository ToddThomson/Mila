module;
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

export module Dnn.FusedComponent;

import Dnn.Component;
import Dnn.ComponentConfig;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.IExecutionContext;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Generic wrapper for fused backend operations.
     *
     * Replaces a sequence of components with a single backend operation
     * without exposing a new component type in the public API.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
    class FusedComponent : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using OperationType = UnaryOperation<TDeviceType, TPrecision>;

        /**
         * @brief Construct fused component from original sequence.
         *
         * @param fused_op_name Backend operation name (e.g., "LinearGeluOp")
         * @param original_components Original unfused components (for parameter extraction)
         * @param exec_context Execution context
         */
        FusedComponent(
            const std::string& fused_op_name,
            const std::vector<std::shared_ptr<ComponentBase>>& original_components,
            IExecutionContext* exec_context )
            : ComponentBase( fused_op_name ),
            original_components_( original_components ),
            exec_context_( exec_context )
        {
            // Extract parameters from original components
            for ( auto& comp : original_components_ )
            {
                auto params = comp->getParameters();
                parameters_.insert( parameters_.end(), params.begin(), params.end() );
            }

            // Create fused backend operation
            ComponentConfig empty_config;
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    fused_op_name,
                    exec_context_,
                    empty_config );
        }

        void onBuilding( const shape_t& input_shape ) override
        {
            operation_->build( input_shape );

            // Bind parameters to operation
            if ( !parameters_.empty() )
            {
                ITensor* weight = parameters_.size() > 0 ? parameters_[ 0 ] : nullptr;
                ITensor* bias = parameters_.size() > 1 ? parameters_[ 1 ] : nullptr;
                operation_->setParameters( weight, bias );
            }
        }

        void forward( const ITensor& input, ITensor& output )
        {
            operation_->forward( input, output );
        }

        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad )
        {
            operation_->backward( input, output_grad, input_grad );
        }

        std::vector<ITensor*> getParameters() const override
        {
            return parameters_;
        }

        std::vector<ITensor*> getGradients() const override
        {
            std::vector<ITensor*> grads;
            for ( auto& comp : original_components_ )
            {
                auto comp_grads = comp->getGradients();
                grads.insert( grads.end(), comp_grads.begin(), comp_grads.end() );
            }
            return grads;
        }

        size_t parameterCount() const override
        {
            size_t count = 0;
            for ( auto* param : parameters_ )
                count += param->size();
            return count;
        }

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            // Save original components for portability
            for ( auto& comp : original_components_ )
            {
                comp->save_( archive, mode );
            }
        }

        std::string toString() const override
        {
            return "Fused: " + this->getName();
        }

        DeviceId getDeviceId() const override
        {
            return exec_context_->getDeviceId();
        }

        void synchronize() override
        {
            exec_context_->synchronize();
        }

    protected:
        void onTrainingChanging( bool is_training ) override
        {
            operation_->setTraining( is_training );
        }

    private:
        std::shared_ptr<OperationType> operation_;
        std::vector<std::shared_ptr<ComponentBase>> original_components_;
        std::vector<ITensor*> parameters_;
        IExecutionContext* exec_context_;
    };
}