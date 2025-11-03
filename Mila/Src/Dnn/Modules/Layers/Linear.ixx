/**
 * @file Linear.ixx
 * @brief Device-templated Linear (fully connected) module.
 *
 * Delegates compute to a UnaryOperation backend. Module owns weight/bias
 * parameters and exposes them to callers (optimizers, serializers).
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <cstdint>

export module Dnn.Modules.Linear;
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
     * @brief Linear (fully connected) module (device-templated).
     *
     * Delegates computation to a device-specific UnaryOperation implementation
     * registered in the OperationRegistry.
     *
     * Module owns trainable parameters (weight, optional bias) and exposes them
     * via accessors. The operation implements y = x * W^T + b where W is the
     * weight matrix and b is an optional bias vector.
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Linear : public Module<TDeviceType>
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
         * @param config Linear configuration.
         */
        explicit Linear( std::shared_ptr<ExecutionContextType> exec_context, const LinearConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();
            this->setTraining( config_.isTraining() );

            initializeParameters();
            createOperation();
        }

        ~Linear() override = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        bool isBuilt() const override
        {
            return (operation_ != nullptr) &&
                (weight_ != nullptr) &&
                (!config_.hasBias() || (bias_ != nullptr)) &&
                built_;
        }

        /**
         * @brief Build the module using an input shape.
         *
         * Linear layer parameters are eagerly created in the constructor based
         * on the configuration. This method binds parameters to the backend
         * operation and triggers backend-specific setup.
         */
        void build( const shape_t& input_shape ) override
        {
            if (built_)
                return;

            validateInputShape( input_shape );

            operation_->setParameters( weight_.get(), bias_.get() );
            operation_->build( input_shape );

            built_ = true;
        }

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - delegates to backend operation.
         *
         * Computes y = x * W^T + b (if bias is enabled).
         */
        void forward( const ITensor& input, ITensor& output ) override
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Linear module must be built before calling forward." );
            }

            validateInputShape( input );

            operation_->forward( input, output );
        }

        /**
         * @brief Backward pass - delegates to backend operation.
         *
         * Computes gradients with respect to input and parameters.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Linear module must be built before calling backward." );
            }

            Parameters parameter_grads;
            operation_->backward( input, output_grad, input_grad, parameter_grads );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save( ModelArchive& archive ) const override
        {
            // Persist parameters if present
            if (weight_)
            {
                // archive.saveTensor( this->getName() + ".weight", *weight_ );
            }

            if (config_.hasBias() && bias_)
            {
                // archive.saveTensor( this->getName() + ".bias", *bias_ );
            }
        }

        void load( ModelArchive& archive ) override
        {
            // Load parameters from archive
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
            training_mode_ = is_training;
        }

        bool isTraining() const override
        {
            return training_mode_;
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
            oss << "Linear: " << getName() << std::endl;
            oss << "Input features: " << config_.getInputFeatures();
            oss << ", Output features: " << config_.getOutputFeatures() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Has Bias: " << (config_.hasBias() ? "Yes" : "No") << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

        // ====================================================================
        // Parameter accessors
        // ====================================================================

        /**
         * @brief Return shared ownership of the weight tensor.
         *
         * @returns Shared pointer to the weight tensor.
         */
        std::shared_ptr<TensorType> getWeight() const noexcept
        {
            return weight_;
        }

        /**
         * @brief Return shared ownership of the bias tensor.
         *
         * @returns Shared pointer to the bias tensor, or nullptr if no bias.
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

        /**
         * @brief Check whether the module has a bias term.
         *
         * @returns True if bias is enabled in the configuration.
         */
        bool hasBias() const noexcept
        {
            return config_.hasBias();
        }

        /**
         * @brief Get the configuration.
         *
         * @returns Reference to the LinearConfig.
         */
        const LinearConfig& getConfig() const noexcept
        {
            return config_;
        }

    private:
        LinearConfig config_;
        bool training_mode_{ false };
        bool built_{ false };

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

        /**
         * @brief Validate input shape for linear operation.
         *
         * Ensures the last dimension matches the configured input_features.
         */
        void validateInputShape( const ITensor& input ) const
        {
            const auto& input_shape = input.shape();
            validateInputShape( input_shape );
        }

        /**
         * @brief Validate input shape for linear operation.
         *
         * Ensures the last dimension matches the configured input_features.
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.empty())
            {
                throw std::invalid_argument( "Linear: input must have rank >= 1" );
            }

            int64_t input_features = input_shape.back();

            if (input_features != config_.getInputFeatures())
            {
                std::ostringstream oss;
                oss << "Linear: input feature dimension mismatch. Expected "
                    << config_.getInputFeatures() << ", got " << input_features;
                throw std::invalid_argument( oss.str() );
            }
        }

        /**
         * @brief Allocate and initialize weight and optional bias tensors.
         *
         * Tensors are created on the execution context device and initialized
         * using Xavier initialization for weights. Bias is zero-initialized.
         */
        void initializeParameters()
        {
            int64_t input_features = config_.getInputFeatures();
            int64_t output_features = config_.getOutputFeatures();

            auto device = exec_context_->getDevice();

            weight_ = std::make_shared<TensorType>( device, shape_t{ output_features, input_features } );
            weight_->setName( this->getName() + ".weight" );

            xavier<TPrecision, MR>( *weight_, input_features, output_features );

            if (config_.hasBias())
            {
                bias_ = std::make_shared<TensorType>( device, shape_t{ output_features } );
                bias_->setName( this->getName() + ".bias" );
                zeros( *bias_ );
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
                    "LinearOp",
                    exec_context_,
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error( "Failed to create Linear compute backend operation." );
            }
        }
    };

    // Convenience aliases for common usages
    export template<TensorDataType TPrecision>
        using CpuLinear = Linear<DeviceType::Cpu, TPrecision>;

    export template<TensorDataType TPrecision>
        using CudaLinear = Linear<DeviceType::Cuda, TPrecision>;
}