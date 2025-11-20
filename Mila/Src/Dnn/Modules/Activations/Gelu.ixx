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
import Serialization.Tensor;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;

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
     *   `save()`/`load()` are minimal but include template metadata so a loader
     *   can validate instantiation parameters.
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
            if (is_built_)
            {
                throw std::runtime_error( "Gelu::build: module already built" );
            }

            operation_->build( input_shape );

            input_shape_ = input_shape;
            is_built_ = true;
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
            if (!is_built_)
            {
                throw std::runtime_error( "Gelu::backward: module must be built before backward pass" );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Gelu::backward: module must be in training mode to compute gradients" );
			}

            operation_->backward(
                output_grad,     // Gradient w.r.t. output (?L/?output)
                input,           // Forward input (required for GELU gradient computation)
                input_grad       // Output: gradient w.r.t. input (?L/?input)
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

            const std::string prefix = "modules/" + this->getName();

            json meta = json::object();
            meta["type"] = "Gelu";
            meta["version"] = 1;
            meta["name"] = this->getName();

            meta["template_device"] = deviceTypeToString( TDeviceType );
            meta["template_precision"] = static_cast<int>( TPrecision );

            archive.writeJson( prefix + "/meta.json", meta );

            json cfg = config_.toJson();
            archive.writeJson( prefix + "/config.json", cfg );
        }

        /**
         * @internal
         * @brief Factory method to reconstruct from archive
         *
         * Called by ModuleFactory during deserialization.
         *
         * @param archive Archive to read from
         * @param module_name Name of the module in the archive
         * @param exec_context Execution context for the new module
         * @return Unique pointer to reconstructed Gelu module
         */
        static std::unique_ptr<Gelu> fromArchive_(
            ModelArchive& archive,
            const std::string& module_name,
            std::shared_ptr<ExecutionContextType> exec_context )
        {
            const std::string prefix = "modules/" + module_name;

            try
            {
                json meta = archive.readJson( prefix + "/meta.json" );
                validateMetadata_( meta, module_name );

                // Load configuration
                json cfg = archive.readJson( prefix + "/config.json" );
                GeluConfig config;
                config.fromJson( cfg );
                //config.setName( module_name );  // Ensure name matches
                config.validate();

                // Construct new instance
                auto gelu = std::make_unique<Gelu>( exec_context, config );

                return gelu;
            }
            catch (const json::exception& e)
            {
                throw std::runtime_error(
                    std::format( "Gelu::fromArchive: JSON error for '{}': {}",
                        module_name, e.what() )
                );
            }
        }

        /**
         * @brief Restore module state.
         *
         * Validates template parameters saved in the archive against the
         * compile-time template parameters of this instantiation and loads
         * the config via GeluConfig::fromJson().
         */
        //void load( ModelArchive& archive, SerializationMode mode ) override
        //{
        //    (void)mode;

        //    const std::string prefix = "modules/" + this->getName();

        //    // Read and validate meta
        //    json meta = archive.readJson( prefix + "/meta.json" );

        //    if (!meta.contains( "template_device" ) || !meta.contains( "template_precision" ))
        //    {
        //        throw std::runtime_error( "Gelu::load: missing template metadata in archive for module: " + this->getName() );
        //    }

        //    std::string file_device = meta.at( "template_device" ).get<std::string>();
        //    int file_precision = meta.at( "template_precision" ).get<int>();

        //    // Use canonical device string produced by deviceTypeToString for comparison.
        //    std::string expected_device = deviceTypeToString( TDeviceType );
        //    int expected_precision = static_cast<int>( TPrecision );

        //    if (file_device != expected_device)
        //    {
        //        std::ostringstream oss;
        //        oss << "Gelu::load: device template mismatch for module '" << this->getName()
        //            << "'. archive='" << file_device << "' expected='" << expected_device << "'";
        //        throw std::runtime_error( oss.str() );
        //    }

        //    if (file_precision != expected_precision)
        //    {
        //        std::ostringstream oss;
        //        oss << "Gelu::load: precision template mismatch for module '" << this->getName()
        //            << "'. archive=" << file_precision << " expected=" << expected_precision;
        //        throw std::runtime_error( oss.str() );
        //    }

        //    // Load config using GeluConfig helper
        //    json cfg = archive.readJson( prefix + "/config.json" );
        //    config_.fromJson( cfg );

        //    // Validate loaded configuration
        //    config_.validate();

        //    // Ensure backend operation exists and is configured with the loaded config.
        //    // Recreate operation to ensure it's bound to the new config if necessary.
        //    if (!operation_)
        //    {
        //        createOperation();
        //    }
        //}

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
        // State and Configuration Implementation
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

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            return exec_context_->getDevice();
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
		
        bool is_built_{ false };
		shape_t input_shape_;
        GeluConfig config_;
        
        std::shared_ptr<ExecutionContextType> exec_context_;
        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        /**
         * @internal
         * @brief Validate metadata from archive
         */
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

            // Validate template parameters match this instantiation
            std::string file_device = meta.value( "template_device", "" );
            std::string file_precision = meta.value( "template_precision", "" );

            std::string expected_device = deviceTypeToString( TDeviceType );
            std::string expected_precision = precisionToString( TPrecision );

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
                    "GeluOp",  exec_context_, config_ );

            if (!operation_)
            {
                throw std::runtime_error( "Failed to create GELU compute backend operation." );
            }
        }

        /**
         * @brief Register GELU creators for supported DeviceType/Precision combinations.
         *
         * This function is called by the ModuleRegistrarManager registration pass
         * (it was declared `extern` in ModuleRegistrar). Each creator delegates to
         * the module's existing `fromArchive_` helper and returns a shared_ptr to
         * the module base for the given device.
         */
        void registerGeluCreators()
        {
            
        }
    };
}