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

export module Dnn.Modules.Residual;
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
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;

import Dnn.Modules.Linear;

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
    class Residual : public Module<TDeviceType>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;
        //using Parameters = std::vector<std::shared_ptr<TensorType>>;
        //using OutputState = std::vector<std::shared_ptr<TensorType>>;

        /**
         * @brief Construct with an existing execution context.
         *
         * @param exec_context Shared execution context for device resources.
         * @param config       Residual configuration (connection type, scaling, projection rules).
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

		// ====================================================================
		// Module lifecycle
		// ====================================================================
        
        bool isBuilt() const override
        {
            // Placeholder; concrete implementations may track build state
            return true;
		}
        
        void build( const shape_t& input_shape ) override
        {
            // Placeholder; concrete implementations may infer shapes and
            // allocate parameters as needed based on input_shape.
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
        void save( ModelArchive& archive ) const override
        {
            // No-op placeholder; serialize parameter tensors if needed
        }

        /**
         * @brief Load module parameters from the provided archive.
         *
         * Placeholder; concrete implementations should restore parameter tensor contents.
         */
        void load( ModelArchive& archive ) override
        {
            // No-op placeholder; deserialize parameter tensors if needed
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
         * @brief Set training/evaluation mode for this module.
         */
        void setTraining( bool is_training ) override
        {
            training_mode_ = is_training;
        }

        /**
         * @brief Query training mode.
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

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            return exec_context_->getDevice();
        }

        /**
         * @brief Return a human-readable description of the module.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Residual" << std::endl;
			//oss << "Name: " << this->getName() << std::endl;
			//oss << config_ << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;
            
            return oss.str();
        }

    private:
        ResidualConfig config_;
        bool training_mode_{ false };

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