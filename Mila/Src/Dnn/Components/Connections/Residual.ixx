/**
 * @file Residual.ixx
 * @brief Device-templated Residual connection module.
 *
 * The `Residual` module implements a residual shortcut y = x + F(x) with
 * configurable connection types (Addition, ScaledAddition, Gated) and optional
 * projection when input/output dimensions differ. Computation is delegated to
 * a device-specific binary operation backend obtained from the OperationRegistry.
 *
 * This implementation is device- and precision-parameterized and follows the
 * same module interface used by other layers (see `Module.ixx` and `Linear.ixx`).
 *
 * @tparam TDeviceType Compile-time device identifier (DeviceType::Cpu or DeviceType::Cuda).
 * @tparam TPrecision  Abstract tensor precision (TensorDataType).
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <cstdint>

export module Dnn.Components.Residual;
export import :Config;

import Dnn.Component;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.OperationBase;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;

import Dnn.Components.Linear;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Device-templated Residual connection module.
     *
     * Delegates binary residual computation to a device-specific backend
     * operation. Parameters (if any) and any projection tensors are stored as
     * `Tensor` instances bound to the associated execution context.
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda).
     * @tparam TPrecision  Abstract tensor precision (TensorDataType).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Residual : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;

        /**
         * @brief Construct with an existing execution context.
         *
         * @param exec_context Shared execution context for device resources.
         * @param config       Residual configuration.
         *
         * Throws std::invalid_argument if exec_context is null.
         */
        explicit Residual( std::shared_ptr<ExecutionContextType> exec_context, const ResidualConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            createOperation();
        }

        ~Residual() override = default;

        /**
         * @brief Return the total number of scalar parameters in this module.
         *
         * This includes gating/scaling parameters and any projection parameters.
         */
        size_t parameterCount() const override
        {
            return 0;
        }

        /**
         * @brief Execute the forward pass.
         *
         * Delegates to the backend binary operation. Inputs and outputs are
         * provided as abstract `ITensor` references to remain device-agnostic.
         */
        void forward( const ITensor& input_a, const ITensor& input_b, ITensor& output )
        {
            operation_->forward( input_a, input_b, output );
        }

        /**
         * @brief Execute the backward pass (gradient computation).
         *
         * Currently a placeholder; backend gradient support should be invoked
         * here when available.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            operation_->backward(
                input,
                output_grad,
                input_grad
            );
        }


        /**
         * @brief Block until all device operations submitted by this module complete.
         */
        void synchronize() override
        {
            exec_context_->synchronize();
        }

        /**
         * @brief Serialize module parameters into the provided archive.
         *
         * Placeholder; concrete implementations should write named parameter
         * tensors into the archive.
         */
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            // No-op placeholder; serialize parameter tensors if needed
        }

        std::vector<ITensor*> getParameters() const override
        {
            return {};
        }

        std::vector<ITensor*> getGradients() const override
        {
            return {};
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

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            return exec_context_->getDevice();
        }

        /**
         * @brief Return a human-readable description of the module.
         *
         * Includes configured name, training/built state, backend presence,
         * device information and parameter count to aid debugging and logging.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "Residual: " << getName() << std::endl;
            oss << "Training mode: " << (this->isTraining() ? "true" : "false") << std::endl;
            oss << "Built: " << (this->isBuilt() ? "true" : "false") << std::endl;
            oss << "Device: " << deviceTypeToString( exec_context_->getDevice()->getDeviceType() ) << std::endl;

            return oss.str();
        }

    protected:

        // ====================================================================
        // Lifecycle
        // ====================================================================

        void onBuilding( const shape_t& input_shape ) override
        {
            operation_->build( input_shape );

            input_shape_ = input_shape;
        }

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Inform backend operation of the new training mode. When leaving
         * training, explicitly unbind any parameter-gradient pointers on the
         * backend to avoid accidental use or pinned memory.
         *
         * Called with Module's training mutex held; do not call setTraining() here.
         */
        void onTrainingChanging( bool is_training ) override
        {
            operation_->setTraining( is_training );
        }

    private:

        ResidualConfig config_;
        shape_t input_shape_;

        std::shared_ptr<BinaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createBinaryOperation<TDeviceType, TPrecision>(
                    "ResidualOp",
                    exec_context_,
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error( "Failed to create Residual compute backend operation." );
            }
        }
    };
}