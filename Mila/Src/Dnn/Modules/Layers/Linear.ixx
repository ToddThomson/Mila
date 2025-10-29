/**
 * @file Linear.ixx
 * @brief Device-templated Linear (fully connected) module.
 *
 * The `Linear` class implements a fully connected (dense) layer parameterized
 * by device and tensor precision. It exposes a device-agnostic interface that
 * operates on `ITensor` views and delegates numerical work to a device-specific
 * `UnaryOperation` backend obtained from the `OperationRegistry`.
 *
 * Usage notes:
 * - Construct with a shared `ExecutionContext<TDeviceType>` to bind resources
 *   to the target device.
 * - Parameters (weight and optional bias) are stored as `Tensor` instances
 *   managed by the module and are exposed through accessor methods.
 *
 * @tparam TDeviceType Compile-time device identifier (e.g. DeviceType::Cpu).
 * @tparam TPrecision  Abstract tensor precision (TensorDataType).
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
//export import :Attributes;

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
     * @brief Device-templated Linear (fully connected) module.
     *
     * This module implements a standard linear map y = x * W^T + b where W is
     * the weight matrix and b is an optional bias vector. Computation is
     * forwarded to a backend `UnaryOperation` instance created via the
     * `OperationRegistry`.
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda).
     * @tparam TPrecision  Abstract tensor precision (TensorDataType).
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
         * @brief Construct a Linear module bound to an execution context.
         *
         * @param exec_context Shared execution context providing device and memory resources.
         * @param config       Layer configuration describing input/output sizes and bias usage.
         *
         * The constructor allocates parameter tensors and creates the compute
         * backend operation. Throws std::invalid_argument if `exec_context`
         * is null or std::runtime_error if the backend operation cannot be created.
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

            if (this->isTraining())
            {
                //FIXME: initializeParameterGradients();
            }

            createOperation();
        }

        ~Linear() override = default;

        /**
         * @brief Return the total number of scalar parameters (weights + bias).
         *
         * @returns Number of scalar parameters stored by the module.
         */
        size_t parameterCount() const override
        {
            size_t num_params = weight_ ? weight_->size() : 0;
            if (config_.hasBias() && bias_) num_params += bias_->size();
            return num_params;
        }

        /**
         * @brief Execute the forward pass.
         *
         * @param input  Input tensor view.
         * @param output Output tensor that receives the result.
         *
         * The call delegates to the backend operation. The caller is expected
         * to provide a correctly shaped `output` tensor unless the backend
         * documents otherwise.
         */
        void forward( const ITensor& input, ITensor& output ) override
        {
            operation_->forward( input, parameters_, output, output_state_ );
        }

        /**
         * @brief Execute the backward pass (gradient computation).
         *
         * @param input       Original forward input.
         * @param output_grad Gradient with respect to the module output.
         * @param input_grad  Tensor to receive gradient with respect to the input.
         *
         * Backward is currently a placeholder and should be implemented by the
         * compute backend when gradient support is required.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override
        {
            /* FIXME: operation_->backward(
                input,
                output_grad,
                parameters_,
                output_state_,
                input_grad,
                parameter_grads_
            );*/
        }

		// ====================================================================
        // Lifecycle
		// ====================================================================

        bool isBuilt() const override
        {
            return (weight_ != nullptr) &&
                   (!config_.hasBias() || (bias_ != nullptr));
		}

        void build( const shape_t& input_shape ) override
        {
            // Linear layer parameters are eagerly created in the constructor
            // based on the configuration. No further action is needed here.
		}

        /**
         * @brief Synchronize device work submitted by this module.
         *
         * Blocks until all operations submitted through the associated execution
         * context have completed.
         */
        void synchronize() override
        {
            exec_context_->synchronize();
        }

        /**
         * @brief Access the weight tensor.
         *
         * @returns Shared pointer to the weight tensor.
         */
        std::shared_ptr<TensorType> getWeight()
        {
            return weight_;
        }

        /**
         * @brief Access the bias tensor if present.
         *
         * @returns Optional containing the bias tensor when configured.
         */
        std::optional<std::shared_ptr<TensorType>> getBias()
        {
            return config_.hasBias() ? std::optional{ bias_ } : std::nullopt;
        }

        /**
         * @brief Check whether the module has a bias term.
         *
         * @returns True if bias is enabled in the configuration.
         */
        bool hasBias() const
        {
            return config_.hasBias();
        }

        /**
         * @brief Serialize module parameters into the provided archive.
         *
         * @param zip Archive used to write parameter blobs.
         *
         * This implementation is a placeholder; concrete persistence should
         * serialize weight and bias tensors as named entries in the archive.
         */
        void save( ModelArchive& zip ) const override
        {
            // No-op placeholder; serialize parameter tensors if needed
        }

        /**
         * @brief Load module parameters from the provided archive.
         *
         * @param archive Archive used to read parameter blobs.
         *
         * This implementation is a placeholder; concrete loading should
         * restore weight and bias tensor contents.
         */
        void load( ModelArchive& archive ) override
        {
            // No-op placeholder; deserialize parameter tensors if needed
        }

        /**
         * @brief Set training/evaluation mode for the module.
         *
         * Some layers change behavior between training and evaluation modes.
         *
         * @param is_training True for training mode; false for evaluation.
         */
        void setTraining( bool is_training ) override
        {
            training_mode_ = is_training;
        }

        /**
         * @brief Query whether the module is in training mode.
         *
         * @returns True if training mode is enabled.
         */
        bool isTraining() const override
        {
            return training_mode_;
        }

        /**
         * @brief Get the module name from configuration.
         *
         * @returns Module name string.
         */
        std::string getName() const override
        {
            return config_.getName();
        }

        /**
         * @brief Produce a multi-line human-readable description.
         *
         * The returned string contains the module name, feature sizes,
         * device type, and parameter count.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Linear: " << this->getName() << std::endl;
            oss << "Input features: " << config_.getInputFeatures();
            oss << ", Output features: " << config_.getOutputFeatures() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;
            
            return oss.str();
        }

    private:
        LinearConfig config_;
		bool training_mode_{ false };

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        // Module-visible parameter containers (backends expect ITensor-like parameters)
        std::vector<std::shared_ptr<TensorType>> parameters_;
        // Gradients for parameters (allocated when training)
        std::vector<std::shared_ptr<TensorType>> parameter_grads_;
        // Cached forward-state tensors (if backend requires them)
        OutputState output_state_;

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

        /**
         * @brief Allocate and initialize weight and optional bias tensors.
         *
         * Tensors are created on the execution context device and appended to
         * the module parameter list. Weight is initialized using Xavier
         * initialization for the configured dimensions.
         */
        void initializeParameters()
        {
            parameters_.clear();

            int64_t input_features = config_.getInputFeatures();
            int64_t output_features = config_.getOutputFeatures();

            // Construct tensors bound to the execution context's device.
            auto device = exec_context_->getDevice();

            weight_ = std::make_shared<TensorType>( device, shape_t{ output_features, input_features } );
            weight_->setName( this->getName() + ".weight" );

            xavier<TPrecision, MR>( *weight_, input_features, output_features );

            parameters_.emplace_back( weight_ );
            //this->parameter_map_["weight"] = weight_;

            if (config_.hasBias())
            {
                bias_ = std::make_shared<TensorType>( device, shape_t{ output_features } );
                bias_->setName( this->getName() + ".bias" );
                parameters_.emplace_back( bias_ );
                //this->parameter_map_["bias"] = bias_;
            }
        }

        /**
         * @brief Allocate gradient tensors for parameters.
         *
         * Gradients are created on the execution context device and named
         * consistently with their parameter counterparts. This is invoked when
         * training-mode gradient storage is required.
         */
        void initializeParameterGradients()
        {
            parameter_grads_.clear();

            size_t input_features = config_.getInputFeatures();
            size_t output_features = config_.getOutputFeatures();

            auto device = exec_context_->getDevice();

            auto weight_grad = std::make_shared<TensorType>( device, shape_t{ output_features, input_features } );
            weight_grad->setName( this->getName() + ".weight_grad" );
            parameter_grads_.push_back( weight_grad );

            if (config_.hasBias())
            {
                auto bias_grad = std::make_shared<TensorType>( device, shape_t{ output_features } );
                bias_grad->setName( this->getName() + ".bias_grad" );
                parameter_grads_.emplace_back( bias_grad );
            }
        }

        /**
         * @brief Create the device-specific compute operation for this layer.
         *
         * The operation is obtained from the global OperationRegistry and is
         * expected to implement the forward (and optionally backward) kernels.
         * Throws std::runtime_error if operation creation fails.
         */
        void createOperation()
        {
            // Create backend operation using the ExecutionContext
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

    // Convenience aliases
    export template<TensorDataType TPrecision>
        using CpuLinear = Linear<DeviceType::Cpu, TPrecision>;

    export template<TensorDataType TPrecision>
        using CudaLinear = Linear<DeviceType::Cuda, TPrecision>;
}