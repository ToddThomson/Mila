
void cuda_mha_forward_fp32( 
    float* out, 
    float* qkvr, float* att,
    const float* inp,
    int B, int T, int C, int NH,
    cudaStream_t stream ) {
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const int softmax_block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float* q, * k, * v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div( total_threads, block_size );
    
    permute_kernel << <num_blocks, block_size >> > (q, k, v, inp, B, T, NH, HS);
    
    cudaCheck( cudaGetLastError() );

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* preatt = inp;
    cublasCheck( cublasSgemmStridedBatched( cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH ) );

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf( HS );
    int grid_size = CEIL_DIV( B * NH * T * 32, softmax_block_size );
    softmax_forward_kernel5 << <grid_size, softmax_block_size >> > (att, scale, preatt, B * NH, T);
    cudaCheck( cudaGetLastError() );

    // new approach: first cuBLAS another batched matmul
    float* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck( cublasSgemmStridedBatched( cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH ) );

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV( B * T * C, block_size );
    unpermute_kernel << <num_blocks, block_size >> > (vaccum, out, B, T, NH, HS);
    
    cudaCheck( cudaGetLastError() );
}