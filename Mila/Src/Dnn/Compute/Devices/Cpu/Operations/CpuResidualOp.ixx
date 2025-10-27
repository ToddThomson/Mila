/**
 * @file CpuResidualOp.ixx
 * @brief CPU implementation of the residual (y = x + F(x)) binary operation.
 *
 * Provides a CPU device implementation of the Residual operation using the
 * device-agnostic BinaryOperation interface. This implementation uses the
 * abstract TensorDataType::FP32 precision and ITensor interfaces for inputs
 * and outputs. Registration uses the canonical operation name "ResidualOp".
 *
 * Notes:
 *  - The class accepts an optional CPU execution context; if provided it will
 *    be validated to ensure it's for a CPU device.
 *  - Forward/backward are implemented in terms of raw host buffers obtained
 *    from the ITensor abstraction. OpenMP is used when available.
 *
 * @since Alpha
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#define _USE_MATH_DEFINES
#include <math.h>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuResidualOp;

import Dnn.Modules.Residual;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorHostTypeMap;
import Dnn.ConfigurationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.CpuExecutionContext;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.OperationBase;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CpuDevice;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU Residual operation (FP32) implementing BinaryOperation interface.
     *
     * This class implements forward and backward using ITensor raw data pointers
     * and targets CPU execution. It follows the same interface style as other
     * CPU operations (e.g., CpuGeluOp).
     */
    export class CpuResidualOp : public BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using MR = CpuMemoryResource;
        using BinaryOperationBase = BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorType = Tensor<TensorDataType::FP32, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;
        using HostType = typename TensorHostTypeMap<TensorDataType::FP32>::host_type;
        using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;

        /**
         * @brief Construct with optional CPU execution context.
         *
         * @param context Optional CPU execution context. If provided, it is validated.
         * @param config Residual operation configuration.
         */
        CpuResidualOp( std::shared_ptr<CpuExecutionContext> context, const ResidualConfig& config )
            :  context_( context ), config_( config )
        {
            if (!context_ )
            {
                throw std::invalid_argument( "CpuResidualOp requires a valid CpuExecutionContext" );
			}
        }

        /**
         * @brief Forward pass: element-wise add inputA and inputB into output.
         *
         * Parameters and output_state are currently unused.
         */
        void forward(
            const ITensor& inputA,
            const ITensor& inputB,
            [[maybe_unused]] const Parameters& parameters,
            ITensor& output,
            [[maybe_unused]] OutputState& output_state ) const override
        {
            const HostType* a = static_cast<const HostType*>(inputA.rawData());
            const HostType* b = static_cast<const HostType*>(inputB.rawData());
            HostType* y = static_cast<HostType*>(output.rawData());

            if (!a || !b || !y)
            {
                throw std::runtime_error( "CpuResidualOp::forward - null tensor data pointer" );
            }

            const size_t N = inputA.size();

#pragma omp parallel for if(N > 1000)
            for (int i = 0; i < static_cast<int>( N ); ++i)
            {
                y[i] = a[i] + b[i];
            }
        }

        /**
         * @brief Backward pass: propagate output gradient equally to both inputs.
         *
         * Adds output_gradient into both input gradients (in-place accumulation).
         */
        void backward(
            const ITensor& inputA,
            const ITensor& inputB,
            const ITensor& output,
            const ITensor& output_gradient,
            [[maybe_unused]] const Parameters& parameters,
            [[maybe_unused]] Parameters& parameter_gradients,
            ITensor& inputA_gradient,
            ITensor& inputB_gradient,
            [[maybe_unused]] const OutputState& output_state ) const override
        {
            const HostType* dout = static_cast<const HostType*>(output_gradient.rawData());
            HostType* dinp1 = static_cast<HostType*>(inputA_gradient.rawData());
            HostType* dinp2 = static_cast<HostType*>(inputB_gradient.rawData());

            if (!dout || !dinp1 || !dinp2)
            {
                throw std::runtime_error( "CpuResidualOp::backward - null tensor data pointer" );
            }

            const size_t N = inputA.size();

#pragma omp parallel for if(N > 1000)
            for (int i = 0; i < static_cast<int>( N ); ++i)
            {
                dinp1[i] += dout[i];
                dinp2[i] += dout[i];
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::ResidualOp;
        }

        std::string getName() const override
        {
            return "Cpu::ResidualOp";
        }

    private:
        
        std::shared_ptr<CpuExecutionContext> context_;
        ResidualConfig config_;
    };

    /**
     * @brief Registrar for CPU Residual operation (FP32).
     *
     * Registers the CPU residual implementation with the OperationRegistry
     * under the canonical name "ResidualOp" for CPU/FP32.
     */
    export class CpuResidualOpRegistrar
    {
    public:
        static void registerOperations()
        {
            OperationRegistry::instance().registerBinaryOperation<DeviceType::Cpu, TensorDataType::FP32>(
                "ResidualOp",
                []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context, const ConfigurationBase& config )
                -> std::shared_ptr<BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>>
                {
                    const auto& residualConfig = static_cast<const ResidualConfig&>(config);
                    auto ctx = std::static_pointer_cast<CpuExecutionContext>(context);
                    return std::make_shared<CpuResidualOp>( ctx, residualConfig );
                }
            );
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}