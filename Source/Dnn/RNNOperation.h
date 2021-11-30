
#ifndef MILA_NN_RNN_OPERATION_H_
#define MILA_NN_RNN_OPERATION_H_

#include <sstream>

#include <cudnn.h>

#include "CuDNN/CudnnContext.h"
#include "CuDNN/Descriptor.h"
#include "CuDNN/Error.h"
#include "CuDNN/Utils.h"

using namespace Mila::Dnn::CuDNN;

namespace Mila::Dnn
{
    /// <summary>
    /// The descriptor of an RNN operation.
    /// Use <seealso cref="RNNOperationBuilder"/> to build an instance of this class.
    /// </summary>
    class RnnOperation : public CuDNN::Descriptor
    {
        friend class DnnModelBuilder;

    public:

        RnnOperation() = default;

        RnnOperation( ManagedCudnnHandle& cudnnHandle )
            : Descriptor( cudnnHandle, CUDNN_RNN_DESCRIPTOR )
        {
            std::cout << "RnnOperation()\n";
        }

        auto SetAlgorithm( cudnnRNNAlgo_t algorithm ) -> RnnOperation&
        {
            algorithm_ = algorithm;
            return *this;
        };

        auto SetCellMode( cudnnRNNMode_t cellMode ) -> RnnOperation&
        {
            cellMode_ = cellMode;
            return *this;
        };

        auto SetBiasMode( cudnnRNNBiasMode_t biasMode ) -> RnnOperation&
        {
            biasMode_ = biasMode;
            return *this;
        };

        auto SetDirectionMode( cudnnDirectionMode_t dirMode ) -> RnnOperation&
        {
            dirMode_ = dirMode;
            return *this;
        };

        auto SetInputMode( cudnnRNNInputMode_t inputMode ) -> RnnOperation&
        {
            inputMode_ = inputMode;
            return *this;
        };

        auto SetDataType( cudnnDataType_t dataType ) -> RnnOperation&
        {
            dataType_ = dataType;
            return *this;
        };

        auto SetMathType( cudnnMathType_t mathType ) -> RnnOperation&
        {
            mathType_ = mathType;
            return *this;
        };

        auto SetMathPrecision( cudnnDataType_t mathPrecision ) -> RnnOperation&
        {
            mathPrecision_ = mathPrecision;
            return *this;
        };

        auto SetInputSize( int32_t inputSize ) -> RnnOperation&
        {
            inputSize_ = inputSize;
            return *this;
        };

        auto SetHiddenSize( int32_t hiddenSize ) -> RnnOperation&
        {
            hiddenSize_ = hiddenSize;
            return *this;
        }

        auto SetProjSize( int32_t projSize ) -> RnnOperation&
        {
            projSize_ = projSize;
            return *this;
        }

        /// <summary>
        /// Number of stacked, physical layers in the deep RNN model.
        /// When dirMode= CUDNN_BIDIRECTIONAL, the physical layer consists of two 
        /// pseudo-layers corresponding to forward and backward directions.
        /// </summary>
        /// <param name="numLayers"></param>
        /// <returns></returns>
        auto SetNumLayers( int32_t numLayers ) -> RnnOperation&
        {
            numLayers_ = numLayers;
            return *this;
        }

        auto SetDropout( Dropout& dropout ) -> RnnOperation&
        {
            dropout_ = std::move( dropout );
            return *this;
        }

        cudnnStatus_t Finalize() override
        {
            auto status = cudnnSetRNNDescriptor_v8(
                static_cast<cudnnRNNDescriptor_t>(GetOpaqueDescriptor()),
                algorithm_,
                cellMode_,
                biasMode_,
                dirMode_,
                inputMode_,
                dataType_,
                mathPrecision_,
                mathType_,
                inputSize_,
                hiddenSize_,
                projSize_,
                numLayers_,
                static_cast<cudnnDropoutDescriptor_t>(dropout_.GetOpaqueDescriptor()),
                0 );

            if ( status != CUDNN_STATUS_SUCCESS )
            {
                SetErrorAndThrow(
                    this, status, "Failed to set RNN descriptor." );

                return status;
            }

            return SetFinalized();
        }

        std::string ToString() const override
        {
            std::stringstream ss;
            ss << "RNNOperation:: " << std::endl
                << " Algorithm: " << CuDNN::to_string( algorithm_ ) << std::endl
                << " CellMode: " << CuDNN::to_string( cellMode_ ) << std::endl
                << " DataType: " << CuDNN::to_string( dataType_ ) << std::endl;

            return ss.str();
        };

    private:

        Dropout dropout_;

        cudnnRNNAlgo_t algorithm_ = CUDNN_RNN_ALGO_STANDARD;
        cudnnRNNMode_t cellMode_ = CUDNN_RNN_RELU;
        cudnnRNNBiasMode_t biasMode_ = CUDNN_RNN_NO_BIAS;
        cudnnRNNInputMode_t inputMode_ = CUDNN_LINEAR_INPUT;

        cudnnDataType_t dataType_ = CUDNN_DATA_FLOAT;
        cudnnMathType_t mathType_ = CUDNN_DEFAULT_MATH;
        cudnnDataType_t mathPrecision_ = CUDNN_DATA_FLOAT;
        cudnnDirectionMode_t dirMode_ = CUDNN_UNIDIRECTIONAL;

        int32_t inputSize_ = 0;
        int32_t hiddenSize_ = 0;
        int32_t projSize_ = 0;
        int32_t numLayers_ = 0;

        uint32_t auxFlags = 0;
    };
}
#endif