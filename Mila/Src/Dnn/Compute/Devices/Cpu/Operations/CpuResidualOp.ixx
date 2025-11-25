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

import Dnn.Components.Residual;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.ComponentConfig;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.CpuExecutionContext;
import Compute.OperationType;
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
        //using BinaryOperationBase = BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>;
        //using TensorType = Tensor<TensorDataType::FP32, MR>;
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

			config.validate();
        }

        /**
         * @brief Forward pass: element-wise add inputA and inputB into output.
         *
         * Parameters and output_state are currently unused.
         */
        void forward( const ITensor& input_a, const ITensor& input_b, ITensor& output ) const override
        {
            const float* A = static_cast<const float*>(input_a.rawData());
            const float* B = static_cast<const float*>(input_b.rawData());
            float* Y = static_cast<float*>(output.rawData());

            if (!A || !B || !Y)
            {
                throw std::runtime_error( "CpuResidualOp::forward - null tensor data pointer" );
            }

            const size_t N = input_a.size();

#pragma omp parallel for if(N > 1000)
            for (int i = 0; i < static_cast<int>( N ); ++i)
            {
                Y[i] = A[i] + B[i];
            }
        }

        /**
         * @brief Backward pass: propagate output gradient equally to both inputs.
         *
         * Adds output_gradient into both input gradients (in-place accumulation).
         */
        void backward(
            const ITensor& input_a,
            const ITensor& input_b,
            const ITensor& output_grad,
            ITensor& input_a_grad,
            ITensor& input_b_grad ) const override
        {
            const float* dY = static_cast<const float*>(output_grad.rawData());
            float* dX1 = static_cast<float*>(input_a_grad.rawData());
            float* dX2 = static_cast<float*>(input_b_grad.rawData());

            if (!dY || !dX1 || !dX2)
            {
                throw std::runtime_error( "CpuResidualOp::backward - null tensor data pointer" );
            }

            const size_t N = input_a.size();

#pragma omp parallel for if(N > 1000)
            for (int i = 0; i < static_cast<int>( N ); ++i)
            {
                dX1[i] += dY[i];
                dX2[i] += dY[i];
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
            OperationRegistry::instance().registerBinaryOperation<DeviceType::Cpu, TensorDataType::FP32, TensorDataType::FP32, TensorDataType::FP32>(
                "ResidualOp",
                []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context, const ComponentConfig& config )
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