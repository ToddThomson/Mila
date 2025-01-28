module;  
#include <corecrt_math.h>  
#include <corecrt_math_defines.h>
#include <thrust/host_vector.h>

export module Compute.Cpu.Ops.Encoder;

import Dnn.Tensor;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute::Cpu::Ops
{
    /*export class Op {
    public:
        virtual void forward(float* out, int* inp, float* wte, float* wpe, int B, int T, int C) = 0;
        virtual void backward(float* dwte, float* dwpe, float* dout, int* inp, int B, int T, int C) = 0;
        virtual ~Op() = default;
    };*/

    //export static class Encoder : public Op {  
    //public:  
    export void encoder_forward( float* out, const Tensor<int>& inp, float* wte, float* wpe, int B, int T, int C ) {
        for ( int b = 0; b < B; b++ ) {
            for ( int t = 0; t < T; t++ ) {
                float* out_bt = out + b * T * C + t * C;
                int ix = inp[ b * T + t ];
                float* wte_ix = wte + ix * C;
                float* wpe_t = wpe + t * C;
                for ( int i = 0; i < C; i++ ) {
                    out_bt[ i ] = wte_ix[ i ] + wpe_t[ i ];
                }
            }
        }
    }

    export void encoder_backward( float* dwte, float* dwpe, float* dout, const Tensor<int>& inp, int B, int T, int C ) {
        for ( int b = 0; b < B; b++ ) {
            for ( int t = 0; t < T; t++ ) {
                float* dout_bt = dout + b * T * C + t * C;
                int ix = inp[ b * T + t ];
                float* dwte_ix = dwte + ix * C;
                float* dwpe_t = dwpe + t * C;
                for ( int i = 0; i < C; i++ ) {
                    float d = dout_bt[ i ];
                    dwte_ix[ i ] += d;
                    dwpe_t[ i ] += d;
                }
            }
        }
    }

}
