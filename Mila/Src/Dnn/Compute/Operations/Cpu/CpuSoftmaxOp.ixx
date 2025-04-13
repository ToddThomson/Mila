/**
 * @file CpuSoftmaxOp.ixx
 * @brief Implementation of the CPU-based softmax operation for neural networks.
 */

module;
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>  
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuSoftmaxOp;

import Dnn.Tensor;  
import Compute.DeviceType;  
import Compute.DeviceContext;
import Compute.OperationBase;  
import Compute.UnaryOperation;
import Compute.OperationRegistry;  
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.MemoryResource;  
import Compute.CpuMemoryResource;
import Compute.CpuDevice;  

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU implementation of the softmax operation for neural networks.
     *
     * This class provides a CPU-based implementation of the softmax operation,
     * which converts a vector of real numbers into a probability distribution.
     * The softmax function is commonly used in classification tasks as the
     * final activation function of a neural network.
     *
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TPrecision The data type used for computation and output (defaults to the input type).
     */
    export
        template<typename TInput = float, typename TPrecision = float>
    class CpuSoftmaxOp final : public UnaryOperation<TInput, TPrecision, DeviceType::Cpu> {
    public:
        using MR = typename CpuDevice::MR;
        /**
         * @brief Constructs a new CPU Softmax operation with the default device context.
         *
         * Initializes the operation with a CPU device context.
         */
        CpuSoftmaxOp() : UnaryOperation<TInput, TPrecision, DeviceType::Cpu>( OperationType::SoftmaxOp ) {}

        /**
         * @brief Constructs a new CPU Softmax operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CPU device.
         */
        CpuSoftmaxOp( std::shared_ptr<DeviceContext> context )
            : UnaryOperation<TInput, TPrecision, DeviceType::Cpu>( OperationType::SoftmaxOp, context ) {
            if ( !context->isDeviceType( DeviceType::Cpu ) ) {
                throw std::runtime_error( "CpuSoftmaxOp requires a CPU device context." );
            }
        }

        /**
         * @brief Performs the forward pass of the softmax operation.
         *
         * Converts input logits into a probability distribution by taking the
         * exponential of each element and normalizing by their sum.
         *
         * @param input Input tensor containing logits.
         * @param parameters Additional input parameters (not used in this operation).
         * @param properties Additional attributes for the operation.
         * @param output Output tensor to store the resulting probability distribution.
         * @param output_cache Cache for storing intermediate results (used in backward pass).
         */
        void forward(
            const Tensor<TInput, MR>& input,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TPrecision, MR>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const override {

            // Verify we're operating on CPU memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cpu ) ) {
                throw std::runtime_error( "CpuSoftmaxOp::forward can only be executed on CPU memory" );
            }

            const TInput* logits = input.raw_data();
            TPrecision* probs = output.raw_data();

            // Get the axis parameter from operation properties
            int64_t axis = properties.axis;

            // Convert negative axis to positive for easier handling
            const int64_t ndim = input.shape().size();
            if ( axis < 0 ) {
                axis = ndim + axis;
            }

            // Validate the axis is within bounds
            if ( axis < 0 || axis >= ndim ) {
                throw std::runtime_error( "Softmax axis out of bounds" );
            }

            // Determine the shapes needed for the computation
            int64_t outer_size = 1;
            for ( int64_t i = 0; i < axis; ++i ) {
                outer_size *= input.shape()[ i ];
            }

            const int64_t dim_size = input.shape()[ axis ];

            int64_t inner_size = 1;
            for ( int64_t i = axis + 1; i < ndim; ++i ) {
                inner_size *= input.shape()[ i ];
            }

            // Compute softmax for each slice along the specified axis
        #pragma omp parallel for collapse(2) if(outer_size * inner_size > 100)
            for ( int64_t outer = 0; outer < outer_size; ++outer ) {
                for ( int64_t inner = 0; inner < inner_size; ++inner ) {
                    // Calculate the starting position for this slice
                    const TInput* slice_input = logits + (outer * dim_size * inner_size) + inner;
                    TPrecision* slice_output = probs + (outer * dim_size * inner_size) + inner;

                    // Find the maximum value for numerical stability
                    TPrecision max_val = -std::numeric_limits<TPrecision>::infinity();
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        TPrecision val = static_cast<TPrecision>( slice_input[ i * inner_size ] );
                        if ( val > max_val ) {
                            max_val = val;
                        }
                    }

                    // Compute exp(x - max_val) and sum
                    TPrecision sum = 0.0f;
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        TPrecision val = std::exp( static_cast<TPrecision>( slice_input[ i * inner_size ] ) - max_val );
                        slice_output[ i * inner_size ] = val;
                        sum += val;
                    }

                    // Normalize by sum
                    TPrecision inv_sum = 1.0f / sum;
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        slice_output[ i * inner_size ] *= inv_sum;
                    }
                }
            }
        }

        /**
         * @brief Performs the backward pass of the softmax operation.
         *
         * Computes gradients with respect to inputs based on the output gradient.
         * For softmax: dL/dx_i = ?_j (dL/dy_j * (y_i * (?_ij - y_j)))
         * where ?_ij is the Kronecker delta.
         *
         * @param input Input tensor from the forward pass.
         * @param output Output tensor from the forward pass (softmax probabilities).
         * @param output_gradient Gradient of the loss with respect to the output.
         * @param parameters Parameters used in forward pass (not used in this operation).
         * @param parameter_gradients Gradients for parameters (not used in this operation).
         * @param input_gradient Gradient of the loss with respect to the input.
         * @param properties Additional attributes for the operation.
         * @param output_cache Cache tensors from forward pass.
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TPrecision, MR>& output,
            const Tensor<TPrecision, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameter_gradients,
            Tensor<TInput, MR>& input_gradient,
            const OperationAttributes& properties,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const {

            // Verify we're operating on CPU memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cpu ) ) {
                throw std::runtime_error( "CpuSoftmaxOp::backward can only be executed on CPU memory" );
            }

            // Get the softmax probabilities (output of forward pass)
            const TPrecision* probs = output.raw_data();
            // Get the gradient of the loss with respect to the output
            const TPrecision* dout = output_gradient.raw_data();
            // Get the gradient buffer for the input
            TInput* dinp = input_gradient.raw_data();

            // Get the axis parameter from operation properties
            int64_t axis = properties.axis;

            // Convert negative axis to positive for easier handling
            const int64_t ndim = input.shape().size();
            if ( axis < 0 ) {
                axis = ndim + axis;
            }

            // Validate the axis is within bounds
            if ( axis < 0 || axis >= ndim ) {
                throw std::runtime_error( "Softmax axis out of bounds" );
            }

            // Determine the shapes needed for the computation
            int64_t outer_size = 1;
            for ( int64_t i = 0; i < axis; ++i ) {
                outer_size *= input.shape()[ i ];
            }

            const int64_t dim_size = input.shape()[ axis ];

            int64_t inner_size = 1;
            for ( int64_t i = axis + 1; i < ndim; ++i ) {
                inner_size *= input.shape()[ i ];
            }

            // Compute gradient for each slice along the specified axis
        #pragma omp parallel for collapse(2) if(outer_size * inner_size > 100)
            for ( int64_t outer = 0; outer < outer_size; ++outer ) {
                for ( int64_t inner = 0; inner < inner_size; ++inner ) {
                    // Calculate the starting positions for this slice
                    const TPrecision* slice_probs = probs + (outer * dim_size * inner_size) + inner;
                    const TPrecision* slice_dout = dout + (outer * dim_size * inner_size) + inner;
                    TInput* slice_dinp = dinp + (outer * dim_size * inner_size) + inner;

                    // Compute dot product of probabilities and output gradients
                    TPrecision dot_product = 0.0f;
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        dot_product += slice_probs[ i * inner_size ] * slice_dout[ i * inner_size ];
                    }

                    // Compute gradients for each element in the slice
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        TPrecision p_i = slice_probs[ i * inner_size ];
                        TPrecision grad = p_i * (slice_dout[ i * inner_size ] - dot_product);
                        slice_dinp[ i * inner_size ] += static_cast<TInput>( grad );
                    }
                }
            }
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cpu::SoftmaxOp").
         */
        std::string getName() const override {
            return "Cpu::SoftmaxOp";
        }
    };

    /**
     * @brief Class responsible for registering the CpuSoftmaxOp operation.
     *
     * The CpuSoftmaxOpRegistrar class registers the CpuSoftmaxOp operation with the OperationRegistry.
     * It associates the operation name "Cpu::SoftmaxOp" with a factory function that creates instances of CpuSoftmaxOp.
     */
    export class CpuSoftmaxOpRegistrar {
    public:
        /**
         * @brief Registers the CpuSoftmaxOp operation with the OperationRegistry.
         *
         * This function registers the CpuSoftmaxOp operation for the CPU device type
         * with the OperationRegistry. It associates the operation name "Cpu::SoftmaxOp"
         * with a factory function that creates instances of CpuSoftmaxOp.
         */
        static void registerOperations() {
            const std::string opName = "Cpu::SoftmaxOp";

            // Updated to use device context-aware registration
            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cpu>(
                opName,
                "Default",  // Default empty variant for backward compatibility
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, float, DeviceType::Cpu>> {
                    return context ? std::make_shared<CpuSoftmaxOp<float, float>>( context )
                        : std::make_shared<CpuSoftmaxOp<float, float>>();
                }
            );
        }

        /**
         * @brief Self-registration mechanism that registers the operation during startup.
         *
         * This static member ensures the operation is registered when the program starts
         * without requiring explicit registration calls.
         */
        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}
