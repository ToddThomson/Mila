module;
#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

export module Dnn.Utils.Cuda.Common;

namespace Mila::Dnn::Utils::Cuda
{
    // Specific configurations based on the enabled precision
#if defined(ENABLE_FP32)
    export typedef float floatX;
#define PRECISION_MODE PRECISION_FP32
    // use fp16 (note: this may require gradient scaler, currently not implemented!)
#elif defined(ENABLE_FP16)
    export typedef half floatX;
#define PRECISION_MODE PRECISION_FP16
#else // Default to bfloat16
    export typedef __nv_bfloat16 floatX;

#define PRECISION_MODE PRECISION_BF16
#endif
}