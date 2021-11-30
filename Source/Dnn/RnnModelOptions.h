#ifndef MILA_DNN_RNN_MODEL_OPTIONS_H_
#define MILA_DNN_RNN_MODEL_OPTIONS_H_

#include <cudnn.h>

namespace Mila::Dnn
{
    struct RnnModelOptions
    {
    public:

        cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
        cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED;
        int numLayers = 2;
        int seqLength = 20;
        int batchSize = 64;
        int vectorSize = 512;
        int inputSize = 512;
        int hiddenSize = 512;
        int projSize = 512;
        double paddingFill = 0.0;

        cudnnDirectionMode_t dirMode = CUDNN_UNIDIRECTIONAL;
        cudnnRNNInputMode_t inputMode = CUDNN_LINEAR_INPUT;
        
        /// <summary>
        /// The type of network used for inference and training routines
        /// </summary>
        cudnnRNNMode_t cellMode = CUDNN_RNN_RELU;
        
        cudnnRNNBiasMode_t biasMode = CUDNN_RNN_DOUBLE_BIAS;
        
        cudnnRNNAlgo_t algorithm = CUDNN_RNN_ALGO_STANDARD;
        
        cudnnMathType_t mathType = CUDNN_DEFAULT_MATH;
        cudnnDataType_t mathPrecision = CUDNN_DATA_FLOAT;
    };
}
#endif