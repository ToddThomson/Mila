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

export module Dnn.Modules.Gelu;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
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
        /* TODO: requires isValid??? */
    class Gelu : public Module<TDeviceType> {
    public:
        using ModuleBase = Module<TDeviceType>;
		using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = typename ModuleBase::ExecutionContextType; // ExecutionContext<TDeviceType>
        using TensorType = Tensor<TPrecision, MR>;
        
        /**
         * @brief Construct GELU with a device id.
         *
         * Creates an ExecutionContext<TDeviceType> using the provided device id.
         *
         * @param device_id Device id (0-based for CUDA; ignored/unused for CPU if not applicable).
         * @param config GELU configuration.
         */
        explicit Gelu( int device_id, const GeluConfig& config )
            : ModuleBase( device_id, config ), config_( config ) {
            config_.validate();
            
            createOperation();
        }

        /**
         * @brief Construct GELU with an existing execution context.
         *
         * @param exec_context Shared execution context for this module.
         * @param config GELU configuration.
         */
        explicit Gelu( std::shared_ptr<ExecutionContextType> exec_context, const GeluConfig& config )
            : ModuleBase( exec_context, config ), config_( config ) {
            if (!exec_context) {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }
            config_.validate();
            
            createOperation();
        }

        ~Gelu() override = default;

        size_t parameterCount() const override {
            return 0;
        }

        void forward( const ITensor& input, ITensor& output ) override {
            // Validate device compatibility
            //validateTensorDevice( input, "input" );
            //validateTensorDevice( output, "output" );

            if (!operation_) {
                createOperation();
            }

            operation_->forward( input, parameters_, output, output_state_ );
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override {
            //validateTensorDevice( input, "input" );
            //validateTensorDevice( output_grad, "output_grad" );
            //validateTensorDevice( input_grad, "input_grad" );

            if (!operation_) {
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

        GeluConfig::ApproximationMethod getApproximationMethod() const {
            return config_.getApproximationMethod();
        }

        void save( ModelArchive& /*archive*/ ) const override {
            // No-op: stateless activation
        }

        void load( ModelArchive& /*archive*/ ) override {
            // No-op: stateless activation
        }

        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Gelu: " << this->getDeviceName() << std::endl;
            oss << "Device: " << deviceToString( this->getExecutionContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << "Approximation Method: " << approximationMethodToString( config_.getApproximationMethod() ) << std::endl;
            return oss.str();
        }

    private:
        GeluConfig config_;
        std::vector<std::shared_ptr<TensorType>> parameters_;
        std::vector<std::shared_ptr<TensorType>> output_state_;
        std::shared_ptr<UnaryOperation<TDeviceType, TensorDataType::FP32>> operation_{ nullptr };

        static std::string approximationMethodToString( GeluConfig::ApproximationMethod method ) {
            switch (method) {
                case GeluConfig::ApproximationMethod::Exact: return "Exact";
                case GeluConfig::ApproximationMethod::Tanh: return "Tanh";
                case GeluConfig::ApproximationMethod::Sigmoid: return "Sigmoid";
                default: return "Unknown";
            }
        }

        void createOperation() {
            // Use the module's execution context to create the backend operation.
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                operation_ = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TensorDataType::FP32 >(
                    "Cpu::GeluOp",
                    this->getExecutionContext(),
                    config_
                );
            } else {
                operation_ = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TensorDataType::FP32>(
                    "Cuda::GeluOp",
                    this->getExecutionContext(),
                    config_
                );
            }

            if (!operation_) {
                throw std::runtime_error( "Failed to create GELU backend operation." );
            }
        }
    };
}