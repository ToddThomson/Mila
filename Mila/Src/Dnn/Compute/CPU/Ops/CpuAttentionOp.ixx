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
    export
    template<typename TInput = float, typename TPrecision = float>
        requires ValidTensorTypes<TInput, TPrecision>
    class CpuAttentionOp : public UnaryOperation<TInput, TPrecision, DeviceType::Cpu> {
    public:

        CpuAttentionOp() : UnaryOperation<TInput, TPrecision, DeviceType::Cpu>(DeviceType::Cpu, OperationType::MultiHeadAttentionOp) {}
    
        void forward( const Tensor<TInput, HostMemoryResource>& input,
            const std::vector<std::shared_ptr<Tensor<TPrecision, HostMemoryResource>>>& parameters,
			const OperationAttributes& properties,
            Tensor<TPrecision, HostMemoryResource>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, HostMemoryResource>>>& output_cache ) const override {

			auto X = input.data();
			auto Y = output.data();

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
                        float* preatt_bth = preatt->data() + b * NH * T * T + h * T * T + t * T;
                        float* att_bth = att->data() + b * NH * T * T + h * T * T + t * T;

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
                        for ( int i = 0; i < hs; i++ ) 
                        { 
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
        
        static void registerOperation() {
            // Registration
            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cpu>(
                "Cpu::MultiHeadAttentionOp",
                []() -> std::unique_ptr<OperationBase<float, float, DeviceType::Cpu>> {
                return std::make_unique<CpuAttentionOp<float, float>>();
            });
        }

        std::string getName() const override {
            return "Cpu::AttentionOp";
        }
    };
}
