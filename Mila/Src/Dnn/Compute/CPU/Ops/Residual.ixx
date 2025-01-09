export module Compute.Cpu.Ops.Residual;

export void residual_forward( float* out, float* inp1, float* inp2, int N ) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

export void residual_backward( float* dinp1, float* dinp2, float* dout, int N ) {
    for (int i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}