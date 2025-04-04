/**
 * @file CpuMultiHeadAttention.ixx
 * @brief Implementation of the CPU-based attention operation for neural networks.
 */

module;
#include <string>
#include <memory>
#include <vector>
#include <cmath>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuAttention;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.CpuComputeResource;
import Compute.MemoryResource;
import Compute.CpuDevice;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU implementation of the Multi-Head Attention operation for neural networks.
     *
     * This class provides a CPU-based implementation of the Multi-Head Attention operation,
     * which is a key component of transformer architectures. The operation performs
     * scaled dot-product attention with multiple attention heads operating in parallel,
     * allowing the model to jointly attend to information from different representation
     * subspaces at different positions.
     *
     * The implementation handles the full attention process:
     * - Query-Key dot products
     * - Scaling
     * - Softmax computation
     * - Attention weighting of values
     *
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TPrecision The data type used for computation and output (defaults to the input type).
     */
    export
        template<typename TInput = float, typename TPrecision = float>
        requires ValidTensorTypes<TInput, TPrecision>
    class CpuMultiHeadAttentionOp: public UnaryOperation<TInput, TPrecision, DeviceType::Cpu> {
    public:
        /**
         * @brief Constructs a new CPU Attention operation.
         *
         * Initializes the operation with the CPU device type and MultiHeadAttentionOp operation type.
         */
        CpuMultiHeadAttentionOp() : UnaryOperation<TInput, TPrecision, DeviceType::Cpu>( DeviceType::Cpu, OperationType::MultiHeadAttentionOp ) {}

        /**
         * @brief Performs the forward pass of the Multi-Head Attention operation.
         *
         * Computes attention scores between queries and keys, applies softmax to get attention weights,
         * and uses these weights to compute a weighted sum of value vectors.
         *
         * @param input Input tensor of shape [B, TElementType, 3*C] containing concatenated query, key, and value vectors.
         * @param parameters Additional parameters (not used in this operation).
         * @param properties Additional attributes for the operation.
         * @param output Output tensor of shape [B, TElementType, C] containing the attention output.
         * @param output_cache Cache for intermediate results [preatt, att] that are used in the backward pass.
         */
        void forward( const Tensor<TInput, HostMemoryResource>& input,
            const std::vector<std::shared_ptr<Tensor<TPrecision, HostMemoryResource>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TPrecision, HostMemoryResource>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, HostMemoryResource>>>& output_cache ) const override {

            auto X = input.raw_data();
            auto Y = output.raw_data();

            auto preatt = output_cache[ 0 ];
            auto att = output_cache[ 1 ];

            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C3 = input.shape()[ 2 ]; // qkv vectors
            int NH = att->shape()[ 1 ];

            int C = C3 / 3;
            int hs = C / NH;
            float scale = 1.0 / sqrtf( hs );

        #pragma omp parallel for collapse(3)
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    for ( int h = 0; h < NH; h++ ) {
                        const float* query_t = X + b * T * C3 + t * C3 + h * hs;
                        float* preatt_bth = preatt->raw_data() + b * NH * T * T + h * T * T + t * T;
                        float* att_bth = att->raw_data() + b * NH * T * T + h * T * T + t * T;

                        float maxval = -10000.0f;
                        for ( int t2 = 0; t2 <= t; t2++ ) {
                            const float* key_t2 = X + b * T * C3 + t2 * C3 + h * hs + C;
                            float val = 0.0f;
                            for ( int i = 0; i < hs; i++ ) {
                                val += query_t[ i ] * key_t2[ i ];
                            }
                            val *= scale;
                            if ( val > maxval ) {
                                maxval = val;
                            }
                            preatt_bth[ t2 ] = val;
                        }

                        float expsum = 0.0f;
                        for ( int t2 = 0; t2 <= t; t2++ ) {
                            float expv = expf( preatt_bth[ t2 ] - maxval );
                            expsum += expv;
                            att_bth[ t2 ] = expv;
                        }
                        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                        for ( int t2 = 0; t2 < T; t2++ ) {
                            if ( t2 <= t ) {
                                att_bth[ t2 ] *= expsum_inv;
                            }
                            else {
                                att_bth[ t2 ] = 0.0f;
                            }
                        }

                        float* out_bth = Y + b * T * C + t * C + h * hs;
                        for ( int i = 0; i < hs; i++ ) {
                            out_bth[ i ] = 0.0f;
                        }

                        for ( int t2 = 0; t2 <= t; t2++ ) {
                            const float* value_t2 = X + b * T * C3 + t2 * C3 + h * hs + C * 2;
                            float att_btht2 = att_bth[ t2 ];
                            for ( int i = 0; i < hs; i++ ) {
                                out_bth[ i ] += att_btht2 * value_t2[ i ];
                            }
                        }
                    }
                }
            }
        }

        /**
         * @brief Performs the backward pass of the Multi-Head Attention operation.
         *
         * Computes gradients with respect to inputs (query, key, value vectors),
         * pre-softmax attention scores, and attention weights based on the output gradient.
         *
         * @param dinp Pointer to the gradient buffer for the input (query, key, value).
         * @param dpreatt Pointer to the gradient buffer for pre-softmax attention scores.
         * @param datt Pointer to the gradient buffer for attention weights.
         * @param dout Pointer to the gradient buffer from the output.
         * @param inp Pointer to the original input values (query, key, value).
         * @param att Pointer to the attention weights computed during forward pass.
         * @param B Batch size.
         * @param TElementType Sequence length.
         * @param C Feature dimension (divided by 3 for query, key, value).
         * @param NH Number of attention heads.
         */
        void backward( float* dinp, float* dpreatt, float* datt, float* dout, float* inp, float* att, int B, int T, int C, int NH ) {
            int C3 = C * 3;
            int hs = C / NH;
            float scale = 1.f / sqrtf( hs );

            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    for ( int h = 0; h < NH; h++ ) {
                        float* att_bth = att + b * NH * T * T + h * T * T + t * T;
                        float* datt_bth = datt + b * NH * T * T + h * T * T + t * T;
                        float* dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
                        float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                        float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                        float* dout_bth = dout + b * T * C + t * C + h * hs;
                        for ( int t2 = 0; t2 <= t; t2++ ) {
                            float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                            float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                            for ( int i = 0; i < hs; i++ ) {
                                datt_bth[ t2 ] += value_t2[ i ] * dout_bth[ i ];
                                dvalue_t2[ i ] += att_bth[ t2 ] * dout_bth[ i ];
                            }
                        }

                        for ( int t2 = 0; t2 <= t; t2++ ) {
                            for ( int t3 = 0; t3 <= t; t3++ ) {
                                float indicator = t2 == t3 ? 1.0f : 0.0f;
                                float local_derivative = att_bth[ t2 ] * (indicator - att_bth[ t3 ]);
                                dpreatt_bth[ t3 ] += local_derivative * datt_bth[ t2 ];
                            }
                        }

                        for ( int t2 = 0; t2 <= t; t2++ ) {
                            float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                            float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C;
                            for ( int i = 0; i < hs; i++ ) {
                                dquery_t[ i ] += key_t2[ i ] * dpreatt_bth[ t2 ] * scale;
                                dkey_t2[ i ] += query_t[ i ] * dpreatt_bth[ t2 ] * scale;
                            }
                        }
                    }
                }
            }
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cpu::AttentionOp").
         */
        std::string getName() const override {
            return "Cpu::AttentionOp";
        }

        /**
         * @brief Gets the class name of this operation.
         *
         * @return const std::string& The class name of the operation.
         */
        static const std::string& className() {
            static std::string name = "Cpu::MultiHeadAttentionOp";
            return name;
        }
    };

    /**
     * @brief Class responsible for registering the CpuMultiHeadAttention operation.
     *
     * The CpuAttentionOpRegistrar class registers the CpuMultiHeadAttention operation with the OperationRegistry.
     * It associates the operation name "Cpu::MultiHeadAttentionOp" with a factory function that creates
     * instances of CpuMultiHeadAttention.
     */
    export class CpuMultiHeadAttentionOpRegistrar {
    public:
        /**
         * @brief Registers the CpuMultiHeadAttention operation with the OperationRegistry.
         *
         * This function registers the CpuMultiHeadAttention operation for the CPU device type
         * with the OperationRegistry. It associates the operation name "Cpu::MultiHeadAttentionOp"
         * with a factory function that creates instances of CpuMultiHeadAttention.
         */
        static void registerOperations() {
            const std::string opName = "Cpu::MultiHeadAttentionOp";

            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cpu>(
                opName,
                []() -> std::shared_ptr<OperationBase<float, float, DeviceType::Cpu>> {
                    return std::make_shared<CpuMultiHeadAttentionOp<float, float>>();
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