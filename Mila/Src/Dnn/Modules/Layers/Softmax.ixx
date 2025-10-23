/**
 * @file Softmax.ixx
 * @brief Implementation of the Softmax activation module (device-templated).
 *
 * Device-templated module that delegates computation to a device-specific
 * UnaryOperation backend registered in the OperationRegistry.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

export module Dnn.Modules.Softmax;
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
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Softmax module that delegates to a device-specific compute backend.
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
         * @brief Construct Softmax with an existing execution context.
         *
         * @param config Softmax configuration.
         * @param exec_context Shared execution context for this module.
         */
        explicit Softmax( const SoftmaxConfig& config, std::shared_ptr<ExecutionContextType> exec_context )
            : config_( config ), exec_context_( exec_context )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            createOperation();
        }

        ~Softmax() override = default;

        size_t parameterCount() const override
        {
            return 0;
        }

        void forward( const ITensor& input, ITensor& output ) override
        {
            if (!operation_)
            {
                createOperation();
            }

            operation_->forward( input, parameters_, output, output_state_ );
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override
        {
            if (!operation_)
            {
                createOperation();
            }

            std::vector<std::shared_ptr<ITensor>> parameter_gradients;
            
            /*FIXME: operation_->backward(
                output_grad,
                input,
                parameters_,
                output_state_,
                input_grad,
                parameter_gradients
            );*/
        }

        void synchronize() override
        {
            exec_context_->synchronize();
        }

        int64_t getAxis() const
        {
            return config_.getAxis();
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
            oss << "Softmax: " << getName() << std::endl;
            oss << "Axis: " << config_.getAxis() << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceType() ) << std::endl;
            return oss.str();
        }

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
        SoftmaxConfig config_;
        std::shared_ptr<ExecutionContextType> exec_context_;
        Parameters parameters_;
        OutputState output_state_;
        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "SoftmaxOp", exec_context_, config_ );

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