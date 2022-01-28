/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
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
#include <cudnn.h>

export module Dnn.RnnModelOptions;

namespace Mila::Dnn
{
    export struct RnnModelOptions
    {
    public:

        cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
        cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED;

        int numLayers = 2;

        /// <summary>
        /// The number of timesteps to unroll RNN model for.
        /// </summary>
        int sequenceLength = 20;
        
        /// <summary>
        /// The number of sequences to train on in parallel.
        /// </summary>
        int batchSize = 64;

        /// <summary>
        /// 
        /// </summary>
        int vectorSize = 512;
        
        /// <summary>
        /// 
        /// </summary>
        int inputSize = 512;
        
        /// <summary>
        /// 
        /// </summary>
        int hiddenSize = 512;
        
        /// <summary>
        /// 
        /// </summary>
        int projSize = 512;
        
        double paddingFill = 0.0;

        cudnnDirectionMode_t dirMode = CUDNN_UNIDIRECTIONAL;
        cudnnRNNInputMode_t inputMode = CUDNN_LINEAR_INPUT;
        
        /// <summary>
        /// The type of network used for inference and training routines.
        /// </summary>
        cudnnRNNMode_t cellMode = CUDNN_RNN_RELU;
        
        cudnnRNNBiasMode_t biasMode = CUDNN_RNN_DOUBLE_BIAS;
        
        cudnnRNNAlgo_t algorithm = CUDNN_RNN_ALGO_STANDARD;
        
        cudnnMathType_t mathType = CUDNN_DEFAULT_MATH;
        cudnnDataType_t mathPrecision = CUDNN_DATA_FLOAT;
    };
}