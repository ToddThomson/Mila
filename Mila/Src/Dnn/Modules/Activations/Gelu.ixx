/**
 * @file Gelu.ixx
 * @brief GELU activation module implementation (device-templated).
 *
 * Provides a device-aware GELU activation Module that delegates computation to a
 * device-specific UnaryOperation backend. The module is templated on the device
 * type only and accepts either a device id (creates its own ExecutionContext)
 * or a shared ExecutionContext<TDeviceType> provided by the caller.
 *
 * The module is stateless (no trainable parameters). It validates tensor device
 * compatibility and forwards ITensor instances to the backend operation.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <type_traits>

export module Dnn.Modules.Gelu;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
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
	// The Compute and Serialization API used here is small and it is likely
	// that use of Compute:: and Serialization:: prefixes be better though

    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Gaussian Error Linear Unit (GELU) activation function module.
     *
     * Device-templated module. GELU delegates computation to a device-specific
     * UnaryOperation implementation registered in the OperationRegistry.
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
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
         * @param exec_context Shared execution context for this module.
         * @param config GELU configuration.
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

        size_t parameterCount() const override
        {
            return 0;
        }

        void forward( const ITensor& input, ITensor& output ) override
        {
            operation_->forward( input, parameters_, output, output_state_ );
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override
        {

            if (!operation_)
            {
                createOperation();
            }

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

        void synchronize() override
        {
            exec_context_->synchronize();
        }

        GeluConfig::ApproximationMethod getApproximationMethod() const
        {
            return config_.getApproximationMethod();
        }

        void save( ModelArchive& /*archive*/ ) const override
        {
            // No-op: stateless activation
        }

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

        void setTraining( bool is_training ) override
        {
            training_mode_ = is_training;
        }

        bool isTraining() const override
        {
            return training_mode_;
        }

        std::string getName() const override
        {
            return config_.getName();
        }

    private:

        bool training_mode_{ false };
        GeluConfig config_;
        std::shared_ptr<ExecutionContextType> exec_context_;
        std::vector<std::shared_ptr<TensorType>> parameters_;
        std::vector<std::shared_ptr<TensorType>> output_state_;
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