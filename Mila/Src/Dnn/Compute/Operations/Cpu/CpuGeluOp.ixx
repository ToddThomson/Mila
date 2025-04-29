module;
#include <memory>
#include <vector>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#ifdef USE_OMP
#include <omp.h>
#endif
#include <iostream>
#include <stdexcept>

export module Compute.CpuGeluOp;

import Dnn.Tensor;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CpuDevice;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    const float GELU_SCALING_FACTOR = sqrtf( 2.0f / M_PI );
    // REVIEW: constexpr float GELU_SCALING_FACTOR = sqrtf(2.0f / M_PI);

    export class CpuGeluOp : public UnaryOperation<float, float, DeviceType::Cpu> {
    public:
        using MR = typename CpuDevice::MR;

        /**
         * @brief Constructs a new CpuGeluOp with the default device context.
         */
        CpuGeluOp() : UnaryOperation<float, float, DeviceType::Cpu>( OperationType::GeluOp ) {}

        /**
         * @brief Constructs a new CpuGeluOp with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CPU device.
         */
        CpuGeluOp( std::shared_ptr<DeviceContext> context )
            : UnaryOperation<float, float, DeviceType::Cpu>( OperationType::GeluOp, context ) {
        }

        /**
         * @brief Performs the forward pass of the GELU activation function.
         *
         * Implements the Gaussian Error Linear Unit (GELU) activation function:
         * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/?) * (x + 0.044715 * x^3)))
         *
         * @param input The input tensor.
         * @param parameters Parameter tensors (not used in this operation).
         * @param properties Additional operation attributes.
         * @param output The output tensor.
         * @param output_state Cache for intermediate results (not used).
         */
        void forward(
            const Tensor<float, MR>& input,
            const std::vector<std::shared_ptr<Tensor<float, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<float, MR>& output,
            std::vector<std::shared_ptr<Tensor<float, MR>>>& output_state ) const override {

            // Verify we're operating on CPU memory
            if ( this->getDeviceContext()->getDevice()->getDeviceType() != DeviceType::Cpu ) {
                throw std::runtime_error( "CpuGeluOp::forward can only be executed on CPU memory" );
            }

            // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
            const float* X = input.raw_data();
            float* Y = output.raw_data();
            const int N = input.size();

            for ( int i = 0; i < N; i++ ) {
                float x = X[ i ];
                float cube = 0.044715f * x * x * x;
                Y[ i ] = 0.5f * x * (1.0f + tanhf( GELU_SCALING_FACTOR * (x + cube) ));
            }
        }

        // we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
    #pragma float_control(precise, on, push)
    #if defined(__GNUC__) && !defined(__clang__)
        __attribute__( (optimize( "no-finite-math-only" )) )
        #endif
            /**
             * @brief Performs the backward pass of the GELU activation function.
             *
             * Computes the gradient of the GELU function with respect to its input.
             *
             * @param dinp Pointer to gradient buffer for input.
             * @param inp Pointer to original input values.
             * @param dout Pointer to gradient from output.
             * @param N Number of elements.
             */
            void backward( float* dinp, float* inp, float* dout, int N ) {
            // Verify we're operating on CPU memory
            if ( this->getDeviceContext()->getDevice()->getDeviceType() != DeviceType::Cpu ) {
                throw std::runtime_error( "CpuGeluOp::backward can only be executed on CPU memory" );
            }

            for ( int i = 0; i < N; i++ ) {
                float x = inp[ i ];
                float cube = 0.044715f * x * x * x;
                float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
                float tanh_out = tanhf( tanh_arg );
                float coshf_out = coshf( tanh_arg );
                float sech_out = 1.0f / (coshf_out * coshf_out);
                float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
                dinp[ i ] += local_grad * dout[ i ];
            }
        #pragma float_control(pop)
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cpu::GeluOp").
         */
        std::string getName() const override {
            return "Cpu::GeluOp";
        }
    };

    /**
    * @brief Class responsible for registering the CpuGeluOp operation.
    *
    * The CpuGeluOpRegistrar class registers the CpuGeluOp operation with the OperationRegistry.
    * It associates the operation name "Cpu::GeluOp" with a factory function that creates instances of CpuGeluOp.
    */
    export class CpuGeluOpRegistrar {
    public:
        /**
        * @brief Registers the CpuGeluOp operation with the OperationRegistry.
        *
        * This function registers the CpuGeluOp operation for the CPU device type
        * with the OperationRegistry. It associates the operation name "Cpu::GeluOp"
        * with a factory function that creates instances of CpuGeluOp.
        */
        static void registerOperations() {
            const std::string opName = "Cpu::GeluOp";

            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cpu>(
                opName,
                "Default",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, float, DeviceType::Cpu>> {
                    return context ? std::make_shared<CpuGeluOp>( context )
                        : std::make_shared<CpuGeluOp>();
                }
            );
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}