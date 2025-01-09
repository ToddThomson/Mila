/*
 * Copyright 2024 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the Mila end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

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