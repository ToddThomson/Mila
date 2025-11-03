/**
 * @file LayerNorm.ixx
 * @brief Device-templated Layer Normalization module.
 *
 * Delegates compute to a UnaryOperation backend. Module owns weight/bias
 * parameters and exposes them to callers (optimizers, serializers).
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

export module Dnn.Modules.LayerNorm;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
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

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Layer Normalization module (device-templated).
     *
     * Delegates computation to a device-specific UnaryOperation implementation
     * registered in the OperationRegistry.
     *
     * Module owns trainable parameters (weight, optional bias) and exposes them
     * via accessors. Runtime scratch/state (mean/rstd) is allocated by the
     * backend operation during build/forward.
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LayerNorm : public Module<TDeviceType>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;
        using HostType = typename TensorHostTypeMap<TPrecision>::host_type;

        /**
         * Construct with an existing execution context.
         *
         * @param exec_context Shared execution context for device resources.
         * @param config LayerNorm configuration.
         */
        explicit LayerNorm( std::shared_ptr<ExecutionContextType> exec_context, const LayerNormConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();
			this->setTraining( config_.isTraining() );

            // REVIEW: init in build or eagerly create parameter tensors if normalized_shape is configured
            if (config_.hasNormalizedShape())
            {
                initializeParameters();
            }

            createOperation();
        }

        ~LayerNorm() override = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        bool isBuilt() const override
        {
            return (operation_ != nullptr) && (weight_ != nullptr) &&  (!config_.hasBias() || (bias_ != nullptr)) &&
                built_;
        }

        /**
         * @brief Build the module using an input shape.
         *
         * Ensures parameters are allocated when they require the input shape
         * (for example when an axis is specified). Binds parameters to the
         * backend operation via setParameters and calls the operation build.
         */
        void build( const shape_t& input_shape ) override
        {
            if (built_)
                return;

            // Validate shape compatibility with configured normalized_shape
            validateInputShape( input_shape );

            if ( !config_.hasNormalizedShape() )
            {
                allocateParametersForShape( input_shape );
            }

            // Allow backend to build device-specific buffers and cache parameter views
            operation_->setParameters( weight_.get(), bias_.get() );
            operation_->build( input_shape );

            built_ = true;
        }

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        void forward( const ITensor& input, ITensor& output ) override
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "LayerNorm module must be built before calling forward." );
            }

            // Validate incoming shape against configured normalized_shape (if present)
            validateInputShape( input );

            operation_->forward( input, output );
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override
        {
            // Backward not implemented yet.
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save( ModelArchive& archive ) const override
        {
            // Persist parameters if present
            if (weight_)
            {
                //archive.saveTensor( this->getName() + ".weight", *weight_ );
            }

            if (config_.hasBias() && bias_)
            {
                //archive.saveTensor( this->getName() + ".bias", *bias_ );
            }
        }

        void load( ModelArchive& archive ) override
        {
            
        }

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
            training_mode_ = is_training;
        }

        bool isTraining() const override
        {
            return training_mode_;
        }

        // ====================================================================
        // Parameter accessors
        // ====================================================================

        /**
         * @brief Return shared ownership of the weight tensor.
         *
         * Returns nullptr if weight is not yet initialized.
         */
        std::shared_ptr<TensorType> getWeight() const noexcept
        {
            return weight_;
        }

        /**
         * @brief Return shared ownership of the bias tensor.
         *
         * Returns nullptr when the module is configured without bias or bias not initialized.
         */
        std::shared_ptr<TensorType> getBias() const noexcept
        {
            return bias_;
        }

        /**
         * @brief Return parameters in canonical order (weight, then bias if present).
         *
         * Useful for optimizers and parameter iteration helpers.
         */
        Parameters getParameters() const
        {
            Parameters p;
            if (weight_) p.emplace_back( weight_ );
            if (bias_)   p.emplace_back( bias_ );
            return p;
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

    private:
        LayerNormConfig config_;
        bool training_mode_{ false };

        bool built_{ false };

        std::vector<int64_t> outer_shape_;

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

        // Validate input shape against configured normalized_shape using an ITensor.
        void validateInputShape( const ITensor& input ) const
        {
            const auto& norm_shape = config_.getNormalizedShape();
            const auto& input_shape = input.shape();

            if (input_shape.size() < norm_shape.size())
            {
                throw std::invalid_argument( "Input rank must be >= normalized_shape rank" );
            }

            // Check trailing dimensions match normalized_shape
            size_t offset = input_shape.size() - norm_shape.size();

            for (size_t i = 0; i < norm_shape.size(); ++i)
            {
                if (input_shape[offset + i] != norm_shape[i])
                {
                    throw std::invalid_argument( "Input trailing dimensions don't match normalized_shape" );
                }
            }
        }

        // Validate input shape against configured normalized_shape using a shape_t.
        void validateInputShape( const shape_t& input_shape ) const
        {
            const auto& norm_shape = config_.getNormalizedShape();

            if (input_shape.size() < norm_shape.size())
            {
                throw std::invalid_argument( "Input rank must be >= normalized_shape rank" );
            }

            // Check trailing dimensions match normalized_shape
            size_t offset = input_shape.size() - norm_shape.size();

            for (size_t i = 0; i < norm_shape.size(); ++i)
            {
                if (input_shape[offset + i] != norm_shape[i])
                {
                    throw std::invalid_argument( "Input trailing dimensions don't match normalized_shape" );
                }
            }
        }

        // Allocate weight/bias given an input shape (used in build when channels unknown at ctor).
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

        // Eager initialization when normalized_shape is available at construction time.
        void initializeParameters()
        {
            // If axis specified, we cannot eagerly determine channels here.
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