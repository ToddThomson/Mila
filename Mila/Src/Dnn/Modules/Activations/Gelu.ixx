/**
 * @file Gelu.ixx
 * @brief GELU activation module implementation.
 *
 * Device-templated GELU module that delegates computation to a registered
 * device-specific UnaryOperation backend.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <type_traits>
#include <stdexcept>

export module Dnn.Modules.Gelu;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    // TJT: Review: Does this pollute our API surface?
	// For Mila namespaces the following using directives are acceptable.
	// They improve readability without significant risk of name collisions.
    // Possible patch..
	// The Compute and Serialization API used here is small and it is likely
	// that use of Compute:: and Serialization:: prefixes would be better

    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Gaussian Error Linear Unit (GELU) activation module.
     *
     * Device-templated GELU module that performs forward (and optionally
     * backward) computation by delegating to a registered device-specific
     * UnaryOperation implementation found via the OperationRegistry.
     *
     * @tparam TDeviceType Compile-time device identifier (DeviceType::Cpu or DeviceType::Cuda).
     * @tparam TPrecision  Tensor data precision used by this module.
     *
     * Preconditions:
     * - A valid (non-null) std::shared_ptr<ExecutionContext<TDeviceType>> must be
     *   supplied to the constructor.
     * - The module's `build(const shape_t&)` must be called before `forward()` to
     *   fully initialize the backend operation.
     *
     * Behavior:
     * - Stateless: no trainable parameters; `parameterCount()` returns 0 and
     *   `save()`/`load()` are no-ops.
     * - `forward()` delegates to the backend UnaryOperation. The caller is
     *   responsible for providing device-compatible `ITensor` objects.
     * - `backward()` is currently not implemented and will not compute parameter
     *   gradients (placeholder in source).
     *
     * Threading / Synchronization:
     * - Module does not guarantee thread-safety; call `synchronize()` to wait
     *   for outstanding device work to complete when needed.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Gelu : public Module<TDeviceType>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;

        /**
         * @brief Construct GELU with an existing execution context.
         *
         * @param exec_context Shared execution context for this module. Must be non-null.
         * @param config GELU configuration (name and approximation method).
         *
         * @throws std::invalid_argument if `exec_context` is null.
         * @throws std::invalid_argument or std::runtime_error if `config` validation fails.
         */
        explicit Gelu(  std::shared_ptr<ExecutionContextType> exec_context, const GeluConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            createOperation();
        }

        ~Gelu() override = default;
       

        // ====================================================================
        // Lifecycle
        // ====================================================================
        
        /**
         * @brief Check whether the module has completed its build process.
         *
         * Returns true when the backend operation has been prepared and the
         * module is ready for forward/backward calls.
         */
        bool isBuilt() const override
        {
            return is_built_;
        }
        
        /**
         * @brief Initialize backend operation and perform any shape-dependent allocation.
         *
         * Allocates or configures the underlying UnaryOperation using the provided
         * input shape. Must be called before `forward()`.
         *
         * @param input_shape Expected shape for input tensors.
         *
         * @throws std::invalid_argument If `input_shape` is incompatible with the module configuration.
         * @throws std::runtime_error If backend allocation or build fails.
         */
        void build( const shape_t& input_shape ) override
        {
			operation_->build( input_shape );
        }

		// ====================================================================
		// Computation
		// ====================================================================

        /**
         * @brief Run the forward computation for this GELU module.
         *
         * Delegates the computation to the device-specific UnaryOperation backend.
         * The caller must ensure `input` and `output` are allocated on the same
         * device as this module and that `build()` has been called.
         *
         * @param input Const reference to the input tensor (non-owning).
         * @param output Reference to the tensor that will receive results (caller-allocated).
         */
        void forward( const ITensor& input, ITensor& output ) override
        {
            operation_->forward( input, output );
        }

        /**
         * @brief Compute gradients with respect to the module input.
         *
         * NOTE: Backward is currently a placeholder. The implementation should
         * delegate to the backend `UnaryOperation::backward` when available.
         *
         * @param input Const reference to the original forward input.
         * @param output_grad Const reference to the gradient w.r.t. module output.
         * @param input_grad Reference to the tensor to be populated with the gradient
         *                   w.r.t. the module input (caller-allocated).
         *
         * Implementations should document whether they overwrite or accumulate into `input_grad`.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override
        {
            std::vector<std::shared_ptr<ITensor>> parameter_gradients;
            
            //FIXME:Loperation_->backward(
            //    //input,
            //    output_grad,
            //    parameters_,
            //    parameter_gradients,
            //    input_grad,
            //    output_state_
            //);
        }

        /**
         * @brief Wait for all asynchronous work submitted by this module to complete.
         *
         * On CPU implementations this may be a no-op. Use to ensure results are
         * visible on the host or to measure synchronous timings.
         */
        void synchronize() override
        {
            exec_context_->synchronize();
        }

        /**
         * @brief Return the configured GELU approximation method.
         *
         * @return Configured GeluConfig::ApproximationMethod value.
         */
        GeluConfig::ApproximationMethod getApproximationMethod() const
        {
            return config_.getApproximationMethod();
        }

        /**
         * @brief Persist module state.
         *
         * No-op for stateless activations; kept to satisfy the Module interface.
         */
        void save( ModelArchive& /*archive*/ ) const override
        {
            // No-op: stateless activation
        }

        /**
         * @brief Restore module state.
         *
         * No-op for stateless activations; kept to satisfy the Module interface.
         */
        void load( ModelArchive& /*archive*/ ) override
        {
            // No-op: stateless activation
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Gelu: " << getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Approximation Method: " << config_.toString( config_.getApproximationMethod() ) << std::endl;
            return oss.str();
        }

        // ====================================================================
        // State and Configuration Implementation
        // ====================================================================

        /**
         * @brief Set training/evaluation mode for this module.
         *
         * GELU has no mode-dependent state internally, however this flag is
         * stored so composite modules can propagate mode to children.
         *
         * @param is_training True to enable training behavior.
         */
        void setTraining( bool is_training ) override
        {
            training_mode_ = is_training;
        }

        /**
         * @brief Query whether the module is in training mode.
         *
         * @return True if training mode is enabled.
         */
        bool isTraining() const override
        {
            return training_mode_;
        }

        /**
         * @brief Number of trainable parameters.
         *
         * GELU is stateless and exposes no trainable parameters.
         *
         * @return 0
         */
        size_t parameterCount() const override
        {
            return 0;
        }

        std::vector<ITensor*> getParameters() const override
        {
            return {};
		}

        std::vector<ITensor*> getParameterGradients() const override
        {
            return {};
        }

        /**
         * @brief Module name for logging and diagnostics.
         *
         * Returns the name stored in the GeluConfig (may be empty).
         */
        std::string getName() const override
        {
            return config_.getName();
        }

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            return exec_context_->getDevice();
		}

    private:
		
        bool is_built_{ false };
        bool training_mode_{ false };
        GeluConfig config_;
        std::shared_ptr<ExecutionContextType> exec_context_;
        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "GeluOp",  exec_context_, config_ );

            if (!operation_)
            {
                throw std::runtime_error( "Failed to create GELU compute backend operation." );
            }
        }
    };
}