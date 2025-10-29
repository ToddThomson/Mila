/**
 * @file LayerNorm.ixx
 * @brief Device-templated Layer Normalization module.
 *
 * Refactored to follow the Gelu module pattern:
 * - Use abstract TensorDataType (TPrecision)
 * - Accept a shared ExecutionContext<TDeviceType> at construction
 * - Delegate compute to UnaryOperation<DeviceType, TPrecision> backend
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
         * @param config LayerNorm configuration.
         * @param exec_context Shared execution context for device resources.
         */
        explicit LayerNorm( std::shared_ptr<ExecutionContextType> exec_context, const LayerNormConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            // Eagerly initialize parameters if the normalized shape is known at construction.
            if ( config_.hasNormalizedShape() )
            {
                initializeParameters();
            }

            createOperation();
        }

        ~LayerNorm() override = default;

        size_t parameterCount() const override
        {
            size_t count = 0;
            if (weight_) count += weight_->size();
            if (config_.hasBias() && bias_) count += bias_->size();

            return count;
        }

        void forward( const ITensor& input, ITensor& output ) override
        {
            // validate incoming shape against configured normalized_shape (if present)
            validateInputShape( input );

            // Initialize parameters (if not already) and allocate per-forward statistics
            // exactly once on the first forward call.
            std::call_once( init_flag_, [this, &input]() {
                lazyInitializeTensors( input );
                } );

            operation_->forward( input, parameters_, output, output_state_ );
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override
        {
            // Backward not implemented yet.
        }

		// ====================================================================
		// Lifecycle
		// ====================================================================
        
        bool isBuilt() const override
        {
            return (operation_ != nullptr) &&
                   (weight_ != nullptr) &&
				(!config_.hasBias() || (bias_ != nullptr));
		}
        
        void build( const shape_t& /*input_shape*/ ) override
        {
            // Parameters are eagerly created in the constructor or lazily
			// on first forward. No further action is needed here.
		}

		// ====================================================================
		// Serialization
		// ====================================================================

        void save( ModelArchive& /*archive*/ ) const override
        {
            // No-op: stateless activation
        }

        void load( ModelArchive& /*archive*/ ) override
        {
            // No-op: stateless activation
        }

        std::string getName() const override
        {
            return config_.getName();
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

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "LayerNorm: " << getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            // FIXME: oss << "Axis: " << config_.getAxis() << std::endl;
            oss << "Epsilon: " << config_.getEpsilon() << std::endl;
            oss << "Has Bias: " << (config_.hasBias() ? "Yes" : "No") << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

    private:
        LayerNormConfig config_;
        bool training_mode_{ false };

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };
        std::shared_ptr<TensorType> mean_{ nullptr };
        std::shared_ptr<TensorType> rstd_{ nullptr };

        std::vector<int64_t> outer_shape_;
        std::once_flag init_flag_;

        std::vector<std::shared_ptr<TensorType>> parameters_;
        OutputState output_state_;

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

        void validateInputShape( const ITensor& input ) const
        {
            const auto& norm_shape = config_.getNormalizedShape();
            const auto& input_shape = input.shape();

            if (input_shape.size() < norm_shape.size())
            {
                throw std::invalid_argument(
                    "Input rank must be >= normalized_shape rank" );
            }

            // Check trailing dimensions match normalized_shape
            size_t offset = input_shape.size() - norm_shape.size();
            for (size_t i = 0; i < norm_shape.size(); ++i)
            {
                if (input_shape[offset + i] != norm_shape[i])
                {
                    throw std::invalid_argument(
                        "Input trailing dimensions don't match normalized_shape" );
                }
            }
        }

        void validateStatisticsShape( const ITensor& input ) const
        {
            auto current_outer = getOuterDims( input );
            if (current_outer != outer_shape_)
            {
                throw std::runtime_error(
                    "Input outer dimensions changed after initialization. "
                    "Expected outer dims to remain constant across forward passes." );
            }

        }

        shape_t getOuterDims( const ITensor& input ) const
        {
            const auto& input_shape = input.shape();
            const auto& norm_shape = config_.getNormalizedShape();

            size_t num_outer = input_shape.size() - norm_shape.size();

            return shape_t(
                input_shape.begin(),
                input_shape.begin() + num_outer
            );
        }

        void lazyInitializeTensors( const ITensor& input )
        {
            // This method is executed exactly once (via call_once) on first forward.
            // It must ensure parameters exist (create if they were not eagerly created)
            // and allocate statistics tensors (mean, rstd) sized per outer grouping.

            const auto& input_shape = input.shape();
            auto device = exec_context_->getDevice();

            // Clear any partially populated containers (defensive)
            parameters_.clear();
            output_state_.clear();

            int64_t channels = 1;

            if (config_.getAxis().has_value())
            {
                // Axis-based normalization: compute axis partition and derive outer_shape_
                // (all dims except the normalized axis) and channels = axis size.
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
                // Trailing-dims normalization: verify and partition normalized_shape against input
                const auto& normalized_shape = config_.getNormalizedShape();
                MultiAxisPartition mp = computeNormalizedShapePartition( input_shape, normalized_shape, "LayerNorm" );

                channels = mp.normalized_size;
                outer_shape_ = std::move( mp.outer_shape );
            }

            // Create or re-create parameter tensors (weight and optional bias)
            weight_ = std::make_shared<TensorType>( device, shape_t{ channels } );
            weight_->setName( this->getName() + ".weight" );
            parameters_.emplace_back( weight_ );

            if (config_.hasBias())
            {
                bias_ = std::make_shared<TensorType>( device, shape_t{ channels } );
                bias_->setName( this->getName() + ".bias" );
                parameters_.emplace_back( bias_ );
            }

            // Statistics tensors are per-outer grouping; if there are no outer dims use scalar shape {1}
            shape_t stats_shape = outer_shape_.empty() ? shape_t{ 1 } : outer_shape_;

            mean_ = std::make_shared<TensorType>( device, stats_shape );
            mean_->setName( this->getName() + ".mean" );

            rstd_ = std::make_shared<TensorType>( device, stats_shape );
            rstd_->setName( this->getName() + ".rstd" );

            output_state_.emplace_back( mean_ );
            output_state_.emplace_back( rstd_ );
        }

        void initializeParameters()
        {
            parameters_.clear();
            output_state_.clear();

            // If the normalization axis is specified we must wait for the input shape to
            // determine channels and outer_shape_ (lazy initialization).
            if ( config_.getAxis().has_value() )
            {
                return;
            }

            // Eager initialization based on normalized_shape available in config_
            const auto& normalized_shape = config_.getNormalizedShape();

            int64_t channels = 1;

            for (const auto& dim : normalized_shape)
            {
                channels *= dim;
            }

            // Construct tensors bound to the execution context's device.
            auto device = exec_context_->getDevice();

            weight_ = std::make_shared<TensorType>( device, shape_t{ channels } );
            weight_->setName( this->getName() + ".weight" );
            parameters_.emplace_back( weight_ );

            if (config_.hasBias())
            {
                bias_ = std::make_shared<TensorType>( device, shape_t{ channels } );
                bias_->setName( this->getName() + ".bias" );
                parameters_.emplace_back( bias_ );
            }

            // Statistics (mean/rstd) are allocated on first forward where input outer dims
            // are known (lazyInitializeTensors will allocate them).
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