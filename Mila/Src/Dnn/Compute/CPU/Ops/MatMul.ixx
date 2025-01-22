module;
#include <corecrt_math.h>
#include <corecrt_math_defines.h>

export module Compute.Cpu.Ops.MatMul;

namespace Mila::Dnn::Compute::Cpu::Ops
{
    export void matmul_forward_naive( float* out, const float* inp, const float* weight, const float* bias,
        int B, int T, int C, int OC ) {
        // the most naive implementation of matrix multiplication
        // this serves as an algorithmic reference, and as a fallback for
        // unfriendly input shapes inside matmul_forward(), below.
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int bt = b * T + t;
                for (int o = 0; o < OC; o++) {
                    float val = (bias != NULL) ? bias[o] : 0.0f;
                    for (int i = 0; i < C; i++) {
                        val += inp[bt * C + i] * weight[o * C + i];
                    }
                    out[bt * OC + o] = val;
                }
            }
        }
    }

    export void matmul_forward( float* out, const float* inp, const float* weight, const float* bias, 
        int B, int T, int C, int OC ) {
        const int LOOP_UNROLL = 8;
        if (B * T % LOOP_UNROLL != 0) {
            matmul_forward_naive( out, inp, weight, bias, B, T, C, OC );
            return;
        }

        #pragma omp parallel for
        for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
            for (int o = 0; o < OC; o++) {
                float result[LOOP_UNROLL];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
                }
                for (int i = 0; i < C; i++) {
                    float w = weight[i + o * C];
                    for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                        int bt = obt + ibt;
                        result[ibt] += inp[bt * C + i] * w;
                    }
                }
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    out[bt * OC + o] = result[ibt];
                }
            }
        }
    }

    export void matmul_backward( float* dinp, float* dweight, float* dbias, const float* dout, const float* inp, const float* weight, 
        int B, int T, int C, int OC ) {
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                const float* dout_bt = dout + b * T * OC + t * OC;
                float* dinp_bt = dinp + b * T * C + t * C;
                for (int o = 0; o < OC; o++) {
                    const float* wrow = weight + o * C;
                    float d = dout_bt[o];
                    for (int i = 0; i < C; i++) {
                        dinp_bt[i] += wrow[i] * d;
                    }
                }
            }
        }
        #pragma omp parallel for
        for (int o = 0; o < OC; o++) {
            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T; t++) {
                    const float* dout_bt = dout + b * T * OC + t * OC;
                    const float* inp_bt = inp + b * T * C + t * C;
                    float* dwrow = dweight + o * C;
                    float d = dout_bt[o];
                    if (dbias != NULL) { dbias[o] += d; }
                    for (int i = 0; i < C; i++) {
                        dwrow[i] += inp_bt[i] * d;
                    }
                }
            }
        }
    }
}
