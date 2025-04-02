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
import Compute.OperationBase;  
import Compute.UnaryOperation;
import Compute.OperationRegistry;  
import Compute.OperationType;  
import Compute.MemoryResource;  
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
     * @tparam TCompute The data type used for computation and output (defaults to the input type).
     */
    export
    class CpuSoftmaxOp final : public UnaryOperation<float, float, DeviceType::Cpu> {
    public:
        /**
         * @brief Constructs a new CPU Softmax operation.
         *
         * Initializes the operation with the CPU device type and SoftmaxOp operation type.
         */
        CpuSoftmaxOp() : UnaryOperation<float, float, DeviceType::Cpu>( DeviceType::Cpu, OperationType::SoftmaxOp ) {}

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
            const Tensor<float, HostMemoryResource>& input,
            const std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>>& parameters,
            const OperationAttributes& properties,
            Tensor<float, HostMemoryResource>& output,
            std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>>& output_cache ) const override {

            const float* logits = input.data();
            float* probs = output.data();

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
            for ( int64_t outer = 0; outer < outer_size; ++outer ) {
                for ( int64_t inner = 0; inner < inner_size; ++inner ) {
                    // Calculate the starting position for this slice
                    const float* slice_input = logits + (outer * dim_size * inner_size) + inner;
                    float* slice_output = probs + (outer * dim_size * inner_size) + inner;

                    // Find the maximum value for numerical stability
                    float max_val = -std::numeric_limits<float>::infinity();
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        float val = slice_input[ i * inner_size ];
                        if ( val > max_val ) {
                            max_val = val;
                        }
                    }

                    // Compute exp(x - max_val) and sum
                    float sum = 0.0f;
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        float val = std::exp( slice_input[ i * inner_size ] - max_val );
                        slice_output[ i * inner_size ] = val;
                        sum += val;
                    }

                    // Normalize by sum
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        slice_output[ i * inner_size ] /= sum;
                    }
                }
            }
        }


        //void forward_old (
        //    const Tensor<float, HostMemoryResource>& input,
        //    const std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>>& parameters,
        //    const OperationAttributes& properties,
        //    Tensor<float, HostMemoryResource>& output,
        //    std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>>& output_cache ) const override {

        //    const float* logits = input.data();
        //    float* probs = output.data();

        //    int B = input.shape()[ 0 ];
        //    int TElementType = input.shape()[ 1 ];
        //    int V = input.shape()[ 2 ];  // 50257;
        //    int Vp = input.shape()[ 2 ];

        //    for ( int b = 0; b < B; b++ ) {
        //        for ( int t = 0; t < TElementType; t++ ) {
        //            // probs <- softmax(logits)  
        //            const float* logits_bt = logits + b * TElementType * Vp + t * Vp;
        //            float* probs_bt = probs + b * TElementType * Vp + t * Vp;

        //            // maxval is only calculated and subtracted for numerical stability
        //            float maxval = -10000.0f; // TODO something better
        //            for ( int i = 0; i < V; i++ ) {
        //                if ( logits_bt[ i ] > maxval ) {
        //                    maxval = logits_bt[ i ];
        //                }
        //            }

        //            float sum = 0.0f;
        //            for ( int i = 0; i < V; i++ ) {
        //                probs_bt[ i ] = expf( logits_bt[ i ] - maxval );
        //                sum += probs_bt[ i ];
        //            }

        //            // note we only loop to V, leaving the padded dimensions  
        //            for ( int i = 0; i < V; i++ ) {
        //                probs_bt[ i ] /= sum;
        //            }

        //            // for extra super safety we may wish to include this too,  
        //            // forcing the probabilities here to be zero, but it shouldn't matter  
        //            for ( int i = V; i < Vp; i++ ) {
        //                probs_bt[ i ] = 0.0f;
        //            }
        //        }
        //    }
        //}

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cpu::SoftmaxOp").
         */
        std::string getName() const override {
            return "Cpu::SoftmaxOp";
        }

        /**
         * @brief Gets the class name of this operation.
         *
         * @return const std::string& The class name of the operation.
         */
        static const std::string& className() {
            static std::string name = "Cpu::SoftmaxOp";
            return name;
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

            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cpu>(
                opName,
                []() -> std::shared_ptr<OperationBase<float, float, DeviceType::Cpu>> {
                    return std::make_shared<CpuSoftmaxOp>();
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
