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
#include <format>
#include <utility>

export module Dnn.Components.Gelu;
export import :Config;

import Dnn.Component;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;
import Serialization.Tensor;
import Serialization.Mode;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;
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
     * Ownership:
     * - Standalone mode: creates and owns an ExecutionContext via the public constructor.
     * - Child mode: shares parent's ExecutionContext via the protected constructor.
     *
     * Preconditions:
     * - The module's `build(const shape_t&)` must be called before `forward()` to
     *   fully initialize the backend operation.
     *
     * Behavior:
     * - Stateless: no trainable parameters; `parameterCount()` returns 0 and
     *   `save()`/`load()` are minimal but include template metadata so a loader
     *   can validate instantiation parameters.
     * - `forward()` delegates to the backend UnaryOperation. The caller is
     *   responsible for providing device-compatible `ITensor` objects.
     * - `backward()` computes input gradients for GELU activation (no parameter gradients).
     *
     * Threading / Synchronization:
     * - Module does not guarantee thread-safety; call `synchronize()` to wait
     *   for outstanding device work to complete when needed.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Gelu : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;

        /**
         * @brief Construct GELU in standalone mode with owned ExecutionContext.
         *
         * Creates and owns an ExecutionContext for the specified device.
         * Suitable for top-level, independently-used components.
         *
         * @param device_id DeviceId identifying the device for this module.
         * @param config GELU configuration (name and approximation method).
         *
         * @throws std::invalid_argument if device_id.type does not match TDeviceType.
         * @throws std::runtime_error if ExecutionContext creation fails.
         */
        explicit Gelu( DeviceId device_id, const GeluConfig& config )
            : owned_context_( createOwnedContext( device_id ) ), exec_context_( owned_context_.get() ), config_( config )
        {
            config_.validate();
            createOperation();
        }

        ~Gelu() override = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================
        
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
        void onBuilding( const shape_t& input_shape ) override
        {
            operation_->build( input_shape );
            input_shape_ = input_shape;
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
        void forward( const ITensor& input, ITensor& output )
        {
            operation_->forward( input, output );
        }

        /**
         * @brief Compute gradients with respect to the module input.
         *
         * Delegates to the backend UnaryOperation::backward implementation to compute
         * the gradient of GELU with respect to the input. For GELU, there are no
         * trainable parameters, so this only computes input gradients.
         *
         * The gradient computation follows the chain rule:
         * dL/dinput = dL/doutput * dGELU(input)/dinput
         *
         * @param input Const reference to the original forward input.
         * @param output_grad Const reference to the gradient w.r.t. module output (?L/?output).
         * @param input_grad Reference to the tensor to be populated with the gradient
         *                   w.r.t. the module input (?L/?input, caller-allocated).
         *
         * @throws std::runtime_error if module has not been built via build()
         * @throws std::runtime_error if backend operation is not available
         *
         * @note GELU has no parameters, so no parameter gradients are computed
         * @note The implementation accumulates into input_grad (does not overwrite)
         * @note Requires setTraining(true) for gradient computation in some backends
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if (!this->isBuilt())
            {
                throw std::runtime_error( "Gelu::backward: module must be built before backward pass" );
            }

            if (!this->isTraining())
            {
                throw std::runtime_error( "Gelu::backward: module must be in training mode to compute gradients" );
            }

            operation_->backward(
                input,           // Forward input
                output_grad,     // Gradient w.r.t. output
                input_grad       // Output: gradient w.r.t. input
            );
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

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @brief Persist module state.
         *
         * GELU is stateless (no trainable tensors) but we persist:
         * - module type and name
         * - template parameters (device and precision) so a loader can validate
         * - serialized GeluConfig via its toJson() helper
         */
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)mode;

            json meta = json::object();
            meta["type"] = "Gelu";
            meta["version"] = 1;
            meta["name"] = this->getName();
            meta["template_device"] = deviceTypeToString( TDeviceType );
            meta["template_precision"] = static_cast<int>( TPrecision );

            archive.writeJson( "meta.json", meta );

            json cfg = config_.toJson();
            archive.writeJson( "config.json", cfg );
        }

        /**
         * @internal
         * @brief Factory method to reconstruct from archive
         *
         * Called by ComponentFactory during deserialization.
         *
         * @param archive Archive to read from
         * @param module_name Name of the module in the archive
         * @param exec_context Execution context for the new module (shared from parent)
         * @return Unique pointer to reconstructed Gelu module
         */
        static std::unique_ptr<Gelu> fromArchive_(
            ModelArchive& archive,
            const std::string& module_name,
            IExecutionContext* exec_context )
        {
            try
            {
                json meta = archive.readJson( "meta.json" );
                validateMetadata_( meta, module_name );

                json cfg = archive.readJson( "config.json" );
                GeluConfig config;
                config.fromJson( cfg );
                config.validate();

                // Use protected constructor to share execution context
                return std::unique_ptr<Gelu>( new Gelu( exec_context, config ) );
            }
            catch (const json::exception& e)
            {
                throw std::runtime_error(
                    std::format( "Gelu::fromArchive: JSON error for '{}': {}",
                        module_name, e.what() )
                );
            }
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

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

        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        // ====================================================================
        // State and Configuration
        // ====================================================================

        /**
         * @brief Module name for logging and diagnostics.
         *
         * Returns the name stored in the GeluConfig (may be empty).
         */
        std::string getName() const override
        {
            return config_.getName();
        }

        DeviceId getDeviceId() const override
        {
            return exec_context_->getDeviceId();
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

    protected:
        
        /**
         * @brief Construct GELU as child component sharing parent's ExecutionContext.
         *
         * Used internally by CompositeComponent/Network to create children that share
         * a common execution context. The context is not owned; lifecycle is managed
         * by the parent.
         *
         * @param exec_context Non-owning pointer to shared execution context (must be non-null).
         * @param config GELU configuration.
         *
         * @throws std::invalid_argument if exec_context is null.
         */
        explicit Gelu( IExecutionContext* exec_context, const GeluConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "Gelu: ExecutionContext cannot be null." );
            }

            validateExecutionContext_<TDeviceType>( exec_context_, "Gelu" );
            config_.validate();
            createOperation();
        }

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Propagate training mode to the backend operation. Called with the
         * Module's training mutex held; do not call setTraining() here.
         */
        void onTrainingChanging( bool is_training ) override
        {
            operation_->setTraining( is_training );
        }

    private:

        GeluConfig config_;
        shape_t input_shape_;
        
        // Execution context ownership model:
        // - Standalone: owned_context_ is populated, exec_context_ points to it
        // - Child: owned_context_ is empty, exec_context_ points to parent's context
        std::unique_ptr<IExecutionContext> owned_context_{ nullptr };
        IExecutionContext* exec_context_{ nullptr };

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        static std::unique_ptr<IExecutionContext> createOwnedContext( DeviceId device_id )
        {
            if (device_id.type != TDeviceType)
            {
                throw std::invalid_argument(
                    std::format( "Gelu: constructor device type mismatch: expected {}, got {}",
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( device_id.type ) )
                );
            }

            auto context = createExecutionContext( device_id );

            if (!context)
            {
                throw std::runtime_error( "Gelu: failed to create execution context for device" );
            }

            return context;
        }

        static void validateMetadata_( const json& meta, const std::string& module_name )
        {
            int version = meta.value( "version", 0 );
            if (version != 1)
            {
                throw std::runtime_error(
                    std::format( "Gelu: unsupported version {} for '{}'",
                        version, module_name )
                );
            }

            std::string type = meta.value( "type", "" );
            if (type != "Gelu")
            {
                throw std::runtime_error(
                    std::format( "Gelu: type mismatch for '{}': expected 'Gelu', got '{}'",
                        module_name, type )
                );
            }

            std::string file_device = meta.value( "template_device", "" );
            std::string file_precision = meta.value( "template_precision", "" );

            std::string expected_device = deviceTypeToString( TDeviceType );
            std::string expected_precision = "FP32"; // FIXME: precisionToString( TPrecision );

            if (file_device != expected_device)
            {
                throw std::runtime_error(
                    std::format( "Gelu: device mismatch for '{}': archive='{}', expected='{}'",
                        module_name, file_device, expected_device )
                );
            }

            if (file_precision != expected_precision)
            {
                throw std::runtime_error(
                    std::format( "Gelu: precision mismatch for '{}': archive='{}', expected='{}'",
                        module_name, file_precision, expected_precision )
                );
            }
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "GeluOp", exec_context_, config_ );

            if (!operation_)
            {
                throw std::runtime_error( "Failed to create GELU compute backend operation." );
            }
        }
    };
}