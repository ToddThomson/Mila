/**
 * @file LayerNorm.ixx
 * @brief Device-templated Layer Normalization module.
 *
 * Delegates compute to a UnaryOperation backend and owns weight/bias parameters.
 */

module;
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <type_traits>
#include <cstdint>
#include <stdexcept>
#include <mutex>
#include <utility>

export module Dnn.Components.LayerNorm;
export import :Config;

import Dnn.Component;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorPartitioning;
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
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LayerNorm : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;

        explicit LayerNorm( std::shared_ptr<ExecutionContextType> exec_context, const LayerNormConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            if (config_.hasNormalizedShape())
            {
                initializeParameters();
            }

            createOperation();
        }

        ~LayerNorm() override = default;

        void forward( const ITensor& input, ITensor& output )
        {
            if (!this->isBuilt())
            {
                throw std::runtime_error( "LayerNorm module must be built before calling forward." );
            }

            validateInputShape( input );

            operation_->forward( input, output );
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if (!this->isBuilt())
            {
                throw std::runtime_error( "LayerNorm module must be built before calling backward." );
            }

            if (!this->isTraining())
            {
                throw std::runtime_error( "LayerNorm must be in training mode to call backward." );
            }

            operation_->backward( input, output_grad, input_grad );
        }

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)archive;
        }

        /*void load( ModelArchive& archive, SerializationMode mode ) override
        {
            (void)archive;
        }*/

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

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> params;

            if (weight_)
                params.push_back( weight_.get() );

            if (bias_)
                params.push_back( bias_.get() );

            return params;
        }

        std::vector<ITensor*> getGradients() const override
        {
            if (!this->isTraining())
            {
                throw std::runtime_error( "LayerNorm: getGradients called when not in training mode" );
            }

            std::vector<ITensor*> grads;

            if (weight_grad_)
                grads.push_back( weight_grad_.get() );

            if (bias_grad_)
                grads.push_back( bias_grad_.get() );

            return grads;
        }

        size_t parameterCount() const override
        {
            size_t count = 0;

            if (weight_)
                count += weight_->size();

            if (config_.hasBias() && bias_)
                count += bias_->size();

            return count;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "LayerNorm: " << getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Epsilon: " << config_.getEpsilon() << std::endl;
            oss << "Has Bias: " << (config_.hasBias() ? "Yes" : "No") << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

    protected:

        /**
         * @brief Called when the LayerNorm component is being built for a specific input shape.
         *
         * Default LayerNorm behavior:
         *  - validate input shape against normalized_shape or axis
         *  - if normalized_shape was not specified at construction, allocate weight/bias parameters now
         *  - bind weight/bias parameters to backend operation
         *  - if in training mode, allocate parameter gradients and bind them to backend operation
         *  - call build() on backend operation with input shape
		 */
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            if ( !config_.hasNormalizedShape() )
            {
                allocateParametersForShape( input_shape );
            }

            // Bind forward parameters to operation
            operation_->setParameters( weight_.get(), bias_.get() );

            // If module is already in training mode, allocate and bind gradients now.
            if ( this->isTraining() )
            {
                initializeParameterGradients();
                operation_->setGradients( weight_grad_.get(), bias_grad_.get() );
            }

            operation_->build( input_shape );
        }

        /**
         * @brief Called when module training mode is about to change.
         *
         * Default LayerNorm behavior:
         *  - inform backend operation of the new training mode
         *  - when entering training (newMode && !oldMode) and already built:
         *      allocate parameter gradient buffers and bind them to the backend
         *  - when leaving training (!newMode && oldMode):
         *      unbind backend gradient pointers and free buffers
         *
         * This hook runs with Module's training mutex held; it MUST NOT call setTraining().
         */
        void onTrainingChanging( bool is_training ) override
        {
            // REVIEW: Training and Build lifecycle interaction
            operation_->setTraining( is_training );

            if ( is_training )
            {
                // Entering training: if already built, ensure gradients allocated and bound
                if ( this->isBuilt() )
                {
                    initializeParameterGradients();
                    operation_->setGradients( weight_grad_.get(), config_.hasBias() ? bias_grad_.get() : nullptr );
                }
            }
            else
            {
                // Leaving training: unbind and free gradients
                operation_->clearGradients();

                weight_grad_.reset();
                bias_grad_.reset();
            }
        }

    private:
        LayerNormConfig config_;

        std::vector<int64_t> outer_shape_;

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        // Gradients for parameters (allocated when in training mode)
        std::shared_ptr<TensorType> weight_grad_{ nullptr };
        std::shared_ptr<TensorType> bias_grad_{ nullptr };

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

        void validateInputShape( const ITensor& input ) const
        {
            const auto& norm_shape = config_.getNormalizedShape();
            const auto& input_shape = input.shape();

            if (input_shape.size() < norm_shape.size())
            {
                throw std::invalid_argument( "Input rank must be >= normalized_shape rank" );
            }

            size_t offset = input_shape.size() - norm_shape.size();

            for (size_t i = 0; i < norm_shape.size(); ++i)
            {
                if (input_shape[offset + i] != norm_shape[i])
                {
                    throw std::invalid_argument( "Input trailing dimensions don't match normalized_shape" );
                }
            }
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            const auto& norm_shape = config_.getNormalizedShape();

            if (input_shape.size() < norm_shape.size())
            {
                throw std::invalid_argument( "Input rank must be >= normalized_shape rank" );
            }

            size_t offset = input_shape.size() - norm_shape.size();

            for (size_t i = 0; i < norm_shape.size(); ++i)
            {
                if (input_shape[offset + i] != norm_shape[i])
                {
                    throw std::invalid_argument( "Input trailing dimensions don't match normalized_shape" );
                }
            }
        }

        void allocateParametersForShape( const shape_t& input_shape )
        {
            int64_t channels = 1;

            if (config_.getAxis().has_value())
            {
                const dim_t axis = config_.getAxis().value();
                AxisPartition ap = computeAxisPartition( input_shape, axis, "LayerNorm" );

                channels = ap.axis_size;

                outer_shape_.clear();

                if (ap.normalized_axis > 0)
                {
                    outer_shape_.insert( outer_shape_.end(),
                        input_shape.begin(),
                        input_shape.begin() + ap.normalized_axis );
                }

                if (ap.normalized_axis + 1 < static_cast<int64_t>(input_shape.size()))
                {
                    outer_shape_.insert( outer_shape_.end(),
                        input_shape.begin() + ap.normalized_axis + 1,
                        input_shape.end() );
                }
            }
            else
            {
                const auto& normalized_shape = config_.getNormalizedShape();
                MultiAxisPartition mp = computeNormalizedShapePartition( input_shape, normalized_shape, "LayerNorm" );

                channels = mp.normalized_size;
                outer_shape_ = std::move( mp.outer_shape );
            }

            auto device = exec_context_->getDevice();

            weight_ = std::make_shared<TensorType>( device, shape_t{ channels } );
            weight_->setName( this->getName() + ".weight" );

            if (config_.hasBias())
            {
                bias_ = std::make_shared<TensorType>( device, shape_t{ channels } );
                bias_->setName( this->getName() + ".bias" );
            }
        }

        void initializeParameters()
        {
            if (config_.getAxis().has_value())
            {
                return;
            }

            const auto& normalized_shape = config_.getNormalizedShape();

            int64_t channels = 1;

            for (const auto& dim : normalized_shape)
            {
                channels *= dim;
            }

            auto device = exec_context_->getDevice();

            weight_ = std::make_shared<TensorType>( device, shape_t{ channels } );
            weight_->setName( this->getName() + ".weight" );

            if (config_.hasBias())
            {
                bias_ = std::make_shared<TensorType>( device, shape_t{ channels } );
                bias_->setName( this->getName() + ".bias" );
            }
        }

        void initializeParameterGradients()
        {
            auto device = exec_context_->getDevice();

            if (!weight_grad_ && weight_)
            {
                weight_grad_ = std::make_shared<TensorType>( device, weight_->shape() );
                weight_grad_->setName( this->getName() + ".weight.grad" );
                zeros( *weight_grad_ );
            }

            if (config_.hasBias() && !bias_grad_ && bias_)
            {
                bias_grad_ = std::make_shared<TensorType>( device, bias_->shape() );
                bias_grad_->setName( this->getName() + ".bias.grad" );
                zeros( *bias_grad_ );
            }
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "LayerNormOp",
                    exec_context_,
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error( "Failed to create LayerNorm compute backend operation." );
            }
        }
    };
}