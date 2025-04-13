/**
 * @file CpuEncoderOp.ixx
 * @brief Implementation of the CPU-based encoder operation for neural networks.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#define _USE_MATH_DEFINES
#include <math.h>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuEncoderOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CpuDevice;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU implementation of the encoder operation for neural networks.
     *
     * This class provides a CPU-based implementation of the encoder operation,
     * which combines token embeddings and positional embeddings.
     *
     * @tparam TInput The data type of the input tensor elements (typically int for token indices).
     * @tparam TPrecision The data type used for computation and output (typically float).
     */
    export
        template<typename TInput = int>
    class CpuEncoderOp : public UnaryOperation<TInput, float, DeviceType::Cpu> {
    public:
        using MR = typename CpuDevice::MR;

        /**
         * @brief Constructs a new CPU Encoder operation with the default device context.
         *
         * Initializes the operation with a CPU device context.
         */
        CpuEncoderOp() : UnaryOperation<TInput, float, DeviceType::Cpu>( OperationType::EncoderOp ) {}

        /**
         * @brief Constructs a new CPU Encoder operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CPU device.
         */
        CpuEncoderOp( std::shared_ptr<DeviceContext> context )
            : UnaryOperation<TInput, float, DeviceType::Cpu>( OperationType::EncoderOp, context ) {
            if ( !context->isDeviceType( DeviceType::Cpu ) ) {
                throw std::runtime_error( "CpuEncoderOp requires a CPU device context." );
            }
        }

        /**
         * @brief Performs the forward pass of the encoder operation.
         *
         * Combines token embeddings and positional embeddings for input token indices.
         *
         * @param input Input tensor containing token indices.
         * @param parameters Parameters tensor containing embeddings and other parameters.
         * @param attributes Additional attributes for the operation.
         * @param output Output tensor to store the resulting embeddings.
         * @param output_cache Cache for storing intermediate results (used in backward pass).
         */
        void forward(
            const Tensor<TInput, MR>& input,
            const std::vector<std::shared_ptr<Tensor<float, MR>>>& parameters,
            const OperationAttributes& attributes,
            Tensor<float, MR>& output,
            std::vector<std::shared_ptr<Tensor<float, MR>>>& output_cache ) const override {

            // Verify we're operating on CPU memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cpu ) ) {
                throw std::runtime_error( "CpuEncoderOp::forward can only be executed on CPU memory" );
            }

            auto X = input.raw_data();
            auto Y = output.raw_data();

            auto wte = parameters[ 0 ];
            auto wpe = parameters[ 1 ];

            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = wte->shape()[ 1 ];

        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    float* out_bt = Y + b * T * C + t * C;
                    TInput ix = X[ b * T + t ];
                    float* wte_ix = wte->raw_data() + ix * C;
                    float* wpe_t = wpe->raw_data() + t * C;

                    for ( int i = 0; i < C; i++ ) {
                        out_bt[ i ] = wte_ix[ i ] + wpe_t[ i ];
                    }
                }
            }
        }

        /**
         * @brief Performs the backward pass of the encoder operation.
         *
         * Computes gradients with respect to inputs and parameters.
         *
         * @param input Input tensor from the forward pass.
         * @param output Output tensor from the forward pass.
         * @param output_gradient Gradient of the loss with respect to the output.
         * @param parameters Parameters tensor from forward pass.
         * @param parameter_gradients Gradients for parameters.
         * @param input_gradient Gradient of the loss with respect to the input.
         * @param attributes Additional attributes for the operation.
         * @param output_cache Cache tensors from forward pass.
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<float, MR>& output,
            const Tensor<float, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<float, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<float, MR>>>& parameter_gradients,
            Tensor<TInput, MR>& input_gradient,
            const OperationAttributes& attributes,
            const std::vector<std::shared_ptr<Tensor<float, MR>>>& output_cache ) const {

            // Verify we're operating on CPU memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cpu ) ) {
                throw std::runtime_error( "CpuEncoderOp::backward can only be executed on CPU memory" );
            }

			// TODO backward pass implementation
            
        //    int B = input.shape()[ 0 ];
        //    int T = input.shape()[ 1 ];
        //    int C = wte->shape()[ 1 ];

        //#pragma omp parallel for collapse(2)
        //    for ( int b = 0; b < B; b++ ) {
        //        for ( int t = 0; t < T; t++ ) {
        //            float* dout_bt = dout + b * T * C + t * C;
        //            TInput ix = input[ b * T + t ];
        //            float* dwte_ix = dwte + ix * C;
        //            float* dwpe_t = dwpe + t * C;

        //            for ( int i = 0; i < C; i++ ) {
        //                float d = dout_bt[ i ];
        //            #pragma omp atomic
        //                dwte_ix[ i ] += d;
        //            #pragma omp atomic
        //                dwpe_t[ i ] += d;
        //            }
        //        }
        //    }
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cpu::EncoderOp").
         */
        std::string getName() const override {
            return "Cpu::EncoderOp";
        }
    };

    /**
     * @brief Class responsible for registering the CpuEncoderOp operation.
     *
     * The CpuEncoderOpRegistrar class registers the CpuEncoderOp operation with the OperationRegistry.
     * It associates the operation name "Cpu::EncoderOp" with a factory function that creates
     * instances of CpuEncoderOp.
     */
    export class CpuEncoderOpRegistrar {
    public:
        /**
         * @brief Registers the CpuEncoderOp operation with the OperationRegistry.
         *
         * This function registers the CpuEncoderOp operation for the CPU device type
         * with the OperationRegistry. It associates the operation name "Cpu::EncoderOp"
         * with a factory function that creates instances of CpuEncoderOp.
         */
        static void registerOperations() {
            const std::string opName = "Cpu::EncoderOp";

            // Register with device context-aware operation registry
            OperationRegistry::instance().registerOperation<int, float, DeviceType::Cpu>(
                opName,
                "Default",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<int, float, DeviceType::Cpu>> {
                    return context ? std::make_shared<CpuEncoderOp<int>>( context )
                        : std::make_shared<CpuEncoderOp<int>>();
                }
            );
        }

        /**
         * @brief Self-registration mechanism that registers the operation during startup.
         *
         * This static member ensures the operation is registered when the program starts
         * without requiring explicit registration calls.
         */
        /*static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();*/
    };
}
