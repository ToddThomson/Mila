/**
 * @file Softmax.ixx
 * @brief Device-templated Softmax activation module.
 *
 * Delegates compute to a UnaryOperation backend. Module is stateless (no trainable
 * parameters) and exposes configuration to callers.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cstdint>

export module Dnn.Modules.Softmax;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Precision;
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
     * @brief Softmax activation module (device-templated).
     *
     * Delegates computation to a device-specific UnaryOperation implementation
     * registered in the OperationRegistry.
     *
     * Softmax is a stateless activation function with no trainable parameters.
     * The operation computes: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
     * across a specified axis.
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Softmax : public Module<TDeviceType>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;

        /**
         * @brief Construct with an existing execution context.
         *
         * @param exec_context Shared execution context for device resources.
         * @param config Softmax configuration.
         */
        explicit Softmax( std::shared_ptr<ExecutionContextType> exec_context, const SoftmaxConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();
            this->setTraining( config_.isTraining() );

            createOperation();
        }

        ~Softmax() override = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        bool isBuilt() const override
        {
            return (operation_ != nullptr) && built_;
        }

        /**
         * @brief Build the module using an input shape.
         *
         * Softmax is stateless and has no parameters to allocate. This method
         * validates the input shape and delegates to the backend operation's
         * build method to cache dimension computations.
         */
        void build( const shape_t& input_shape ) override
        {
            if (built_)
                return;

            validateInputShape( input_shape );

            operation_->setParameters( nullptr, nullptr );
            operation_->build( input_shape );

            built_ = true;
        }

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - delegates to backend operation.
         *
         * Computes softmax activation across the configured axis.
         */
        void forward( const ITensor& input, ITensor& output ) override
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Softmax module must be built before calling forward." );
            }

            validateInputShape( input );

            operation_->forward( input, output );
        }

        /**
         * @brief Backward pass - delegates to backend operation.
         *
         * Computes gradient: dX = Y * (dY - dot(Y, dY))
         * where Y is the softmax output.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Softmax module must be built before calling backward." );
            }

            Parameters parameter_grads;
            operation_->backward( input, output_grad, input_grad, parameter_grads );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save( ModelArchive& archive ) const override
        {
            // No-op: stateless activation
        }

        void load( ModelArchive& archive ) override
        {
            // No-op: stateless activation
        }

        // ====================================================================
        // Module interface
        // ====================================================================

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

        size_t parameterCount() const override
        {
            return 0;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Softmax: " << getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Axis: " << config_.getAxis() << std::endl;
            oss << "Parameter count: 0 (stateless)" << std::endl;

            return oss.str();
        }

        // ====================================================================
        // Configuration accessors
        // ====================================================================

        /**
         * @brief Get the softmax axis.
         *
         * @return The axis along which softmax is computed.
         */
        int64_t getAxis() const noexcept
        {
            return config_.getAxis();
        }

        /**
         * @brief Get the configuration.
         *
         * @return Reference to the SoftmaxConfig.
         */
        const SoftmaxConfig& getConfig() const noexcept
        {
            return config_;
        }

    private:
        SoftmaxConfig config_;
        bool training_mode_{ false };
        bool built_{ false };

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

        /**
         * @brief Validate input shape for softmax operation.
         *
         * Ensures the input has valid rank and the configured axis is within bounds.
         */
        void validateInputShape( const ITensor& input ) const
        {
            const auto& input_shape = input.shape();
            validateInputShape( input_shape );
        }

        /**
         * @brief Validate input shape for softmax operation.
         *
         * Ensures the input has valid rank and the configured axis is within bounds.
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.empty())
            {
                throw std::invalid_argument( "Softmax: input must have rank >= 1" );
            }

            int64_t axis = config_.getAxis();
            const int64_t ndim = static_cast<int64_t>(input_shape.size());

            if (axis < 0)
                axis = ndim + axis;

            if (axis < 0 || axis >= ndim)
            {
                throw std::invalid_argument( "Softmax: axis out of bounds for input shape" );
            }
        }

        /**
         * @brief Create the backend compute operation.
         *
         * Looks up the appropriate device-specific operation from the registry
         * and creates an instance bound to this module's execution context.
         */
        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "SoftmaxOp",
                    exec_context_,
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error( "Failed to create Softmax compute backend operation." );
            }
        }
    };

    // Convenience aliases for common usages
    export template<TensorDataType TPrecision>
        using CpuSoftmax = Softmax<DeviceType::Cpu, TPrecision>;

    export template<TensorDataType TPrecision>
        using CudaSoftmax = Softmax<DeviceType::Cuda, TPrecision>;
}