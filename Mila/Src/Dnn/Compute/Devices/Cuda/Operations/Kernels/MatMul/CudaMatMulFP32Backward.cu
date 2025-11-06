namespace Mila::Dnn::Compute
{
    // NOTES:
    // - Backward kernel using cublas for FP32 matmul
	// - Unused. Reference only. To be removed later.

    //void matmul_backward( float* dinp, float* dweight, float* dbias,
    //    float* dout, float* inp, float* weight, int B, int T, int C, int OC )
    //{
    //    float one = 1.0f;
    //    float zero = 0.0f;
    //    
    //    // backward to input, uses = in the backward pass (set the gradient)
    //    cublasCheck( cublasSgemm( cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B * T, OC, &one, weight, C, dout, OC, &zero, dinp, C ) );
    //    
    //    // backward to weight, uses += in the backward pass (accumulate the gradient)
    //    cublasCheck( cublasSgemm( cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B * T, &one, inp, C, dout, OC, &one, dweight, C ) );
    //    
    //    // backward to bias, if given, does a +=
    //    if (dbias != NULL)
    //    {
    //        const int block_size = 1024;
    //        const int grid_size = OC / 32; // for now, OC must be divisible by 32 for this kernel to work
    //        
    //        matmul_backward_bias_kernel4 << <grid_size, block_size, block_size * sizeof( float ) >> > (dbias, dout, B, T, OC);
    //        
    //        cudaCheck( cudaGetLastError() );
    //    }
    //}
}