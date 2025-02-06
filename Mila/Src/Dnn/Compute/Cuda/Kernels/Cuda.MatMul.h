#pragma once

void cuda_matmul_forward( 
    float* out,
    const float* inp, 
    const float* weight, const float* bias,
    int B, int T, int C, int OC );