module;  
#include <string>  
#include <memory>  
#include <vector>  
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
     * @brief CpuSoftmaxOp class implements the softmax operation on CPU.  
     *  
     * @tparam T Data type of the tensor elements.  
     */  
    export
	template<typename T>
    class CpuSoftmaxOp : public UnaryOperation<float, float, CpuDevice> {  
    public:  
        /**  
         * @brief Constructor for CpuSoftmaxOp.  
         */  
        CpuSoftmaxOp() : UnaryOperation<float, float, CpuDevice>(DeviceType::Cpu, OperationType::SoftmaxOp) {}  

        /**  
         * @brief Forward pass of the softmax operation.  
         *  
         * @param input Input tensor containing logits.  
         * @param parameters Additional input parameters (not used).  
         * @param output Output tensor to store probabilities.  
         * @param output_cache Cache for output tensors (not used).  
         */  
        void forward(
            const Tensor<T, CpuMemoryResource>& input,
            const std::vector<std::shared_ptr<Tensor<T, CpuMemoryResource>>>& parameters,
            Tensor<T, CpuMemoryResource>& output,
            std::vector<std::shared_ptr<Tensor<T, CpuMemoryResource>>>& output_cache ) const override {

            const float* logits = input.data();
            float* probs = output.data();

            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int V = input.shape()[ 2 ];  // 50257;
            int Vp = input.shape()[ 2 ];

            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    // probs <- softmax(logits)  
                    const float* logits_bt = logits + b * T * Vp + t * Vp;
                    float* probs_bt = probs + b * T * Vp + t * Vp;

                    // maxval is only calculated and subtracted for numerical stability
                    float maxval = -10000.0f; // TODO something better
                    for ( int i = 0; i < V; i++ ) {
                        if ( logits_bt[ i ] > maxval ) {
                            maxval = logits_bt[ i ];
                        }
                    }

                    float sum = 0.0f;
                    for ( int i = 0; i < V; i++ ) {
                        probs_bt[ i ] = expf( logits_bt[ i ] - maxval );
                        sum += probs_bt[ i ];
                    }

                    // note we only loop to V, leaving the padded dimensions  
                    for ( int i = 0; i < V; i++ ) {
                        probs_bt[ i ] /= sum;
                    }

                    // for extra super safety we may wish to include this too,  
                    // forcing the probabilities here to be zero, but it shouldn't matter  
                    for ( int i = V; i < Vp; i++ ) {
                        probs_bt[ i ] = 0.0f;
                    }
                }
            }
        }

        /**  
         * @brief Registers the CpuSoftmaxOp operation in the operation registry.  
         */  
        static void registerOperation() {  
            OperationRegistry<float, float, CpuDevice>::instance().registerOperation(DeviceType::Cpu, "Cpu::SoftmaxOp", []() -> std::unique_ptr<OperationBase<float, float, CpuDevice>> {  
                return std::make_unique<CpuSoftmaxOp>();  
            });  
        }  

        /**  
         * @brief Gets the name of the operation.  
         *  
         * @return The name of the operation.  
         */  
        std::string getName() const override {  
            return "Cpu::SoftmaxOp";  
        }  
    };  
}