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
                is_built_;
        }

        /**
         * @brief Build the module using an input shape.
         *
         * Linear layer parameters are eagerly created in the constructor based
         * on the configuration. This method binds parameters to the backend
         * operation and triggers backend-specific setup.
         *
         * If in training mode, also initializes gradient tensors and binds them
         * to the operation.
         */
        void build( const shape_t& input_shape ) override
        {
            if (is_built_)
                return;

            validateInputShape( input_shape );

			operation_->setTraining( is_training_ );

            // Bind forward parameters to operation
            operation_->setParameters( weight_.get(), bias_.get() );

            // If training mode, initialize gradients and bind to operation
            if (is_training_)
            {
                initializeParameterGradients();
                operation_->setParameterGradients( weight_grad_.get(), bias_grad_.get() );
            }

            operation_->build( input_shape );

            is_built_ = true;
        }

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - delegates to backend operation.
         *
         * Computes y = x * W^T + b (if bias is enabled).
         */
        void forward( const ITensor& input, ITensor& output )
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
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Linear module must be built before calling backward." );
            }

            if (!is_training_)
            {
                throw std::runtime_error( "Linear module must be in training mode to call backward. Call setTraining(true) first." );
            }

            // Ensure gradients are initialized (defensive check)
            if (!weight_grad_)
            {
                throw std::runtime_error( "Linear module weight gradients not initialized. This is a bug." );
            }

            if (config_.hasBias() && !bias_grad_)
            {
                throw std::runtime_error( "Linear module bias gradients not initialized. This is a bug." );
            }

            operation_->backward( input, output_grad, input_grad );
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
        // Parameters and Gradients
        // ====================================================================

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> params;
            
            if (weight_)
                params.push_back( weight_.get() );
            
            if (bias_)
                params.push_back( bias_.get() );

            return params;
        }

        std::vector<ITensor*> getParameterGradients() const override
        {
            if (!is_training_)
            {
                throw std::runtime_error( "Linear: getParameterGradients called when not in training mode" );
            }

            std::vector<ITensor*> grads;

            if (weight_grad_)
                grads.push_back( weight_grad_.get() );

            if (bias_grad_)
                grads.push_back( bias_grad_.get() );

            return grads;
        }

        /**
         * @brief Get weight gradient tensor.
         *
         * @return Shared pointer to weight gradient, or nullptr if not in training mode
         */
        std::shared_ptr<TensorType> getWeightGrad() const noexcept
        {
            return weight_grad_;
        }

        /**
         * @brief Get bias gradient tensor.
         *
         * @return Shared pointer to bias gradient, or nullptr if bias disabled or not in training mode
         */
        std::shared_ptr<TensorType> getBiasGrad() const noexcept
        {
            return bias_grad_;
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

            // If switching TO training mode after build, initialize gradients and bind them
            if (is_training && is_built_ && operation_)
            {
                // Allocate gradient tensors if not already done
                initializeParameterGradients();

                // Bind gradients to operation
                operation_->setParameterGradients( weight_grad_.get(), bias_grad_.get() );
            }

            // Optional: If switching FROM training to inference, could deallocate gradients
            // to save memory, but we'll keep them for simplicity (allows toggling back)
        }

        bool isTraining() const override
        {
            return is_training_;
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
        /*Parameters getParametersTyped() const
        {
            Parameters p;
            if (weight_) p.emplace_back( weight_ );
            if (bias_)   p.emplace_back( bias_ );
            return p;
        }*/

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
        bool is_training_{ false };
        bool is_built_{ false };

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        std::shared_ptr<TensorType> weight_grad_{ nullptr };
        std::shared_ptr<TensorType> bias_grad_{ nullptr };

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
        * @brief Ensure gradient tensors are allocated with correct shapes.
        */
        void initializeParameterGradients()
        {
            auto device = exec_context_->getDevice();

            if (!weight_grad_)
            {
                weight_grad_ = std::make_shared<TensorType>(
                    device,
                    weight_->shape() );
                weight_grad_->setName( this->getName() + ".weight.grad" );
                zeros( *weight_grad_ );
            }

            if (config_.hasBias() && !bias_grad_)
            {
                bias_grad_ = std::make_shared<TensorType>(
                    device,
                    bias_->shape() );
                bias_grad_->setName( this->getName() + ".bias.grad" );
                zeros( *bias_grad_ );
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