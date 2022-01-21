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
#include <string>
#include <iostream>
#include <vector>
#include <cudnn.h>
#include "CuDNN/init_data.h"

export module Dnn.RnnModel;

import Cuda.Memory;
import Cuda.Helpers;
import CuDnn.Utils;
import CuDnn.Error;
import Dnn.DropoutDescriptor;
import Dnn.TensorDescriptor;
import Dnn.RnnOpDescriptor;
import Dnn.RnnDataSetDescriptor;
import Dnn.Model;
import Dnn.RnnModelOptions;
import Dnn.RnnLayerCollection;

using namespace Mila::Dnn::Cuda;

namespace Mila::Dnn
{
    export template <typename T_ELEM>
    class RnnModel : public DnnModel
    {
    public:

        RnnModel() : DnnModel( DnnModelOptions() )
        {
            options_ = RnnModelOptions();
        }

        /*const Dropout& GetDropout()
        {
            return dropout_;
        }*/

        const RnnOperation& GetRnnOp() const
        {
            return rnnOp_;
        }

        void Train( const int max_epochs = 10 )
        {
            std::cout << "Train()" << " Allocating Tensor memory.." << std::endl;

            CudaMemory x_data = CudaMemory( inputTensorSize_ * sizeof( T_ELEM ) );
            CudaMemory dx_data = CudaMemory( inputTensorSize_ * sizeof( T_ELEM ) );

            CudaMemory y_data = CudaMemory( outputTensorSize_ * sizeof( T_ELEM ) );
            CudaMemory dy_data = CudaMemory( outputTensorSize_ * sizeof( T_ELEM ) );

            CudaMemory hx_data = CudaMemory( hiddenTensorSize_ * sizeof( T_ELEM ) );
            CudaMemory hy_data = CudaMemory( hiddenTensorSize_ * sizeof( T_ELEM ) );

            CudaMemory dhx_data = CudaMemory( hiddenTensorSize_ * sizeof( T_ELEM ) );
            CudaMemory dhy_data = CudaMemory( hiddenTensorSize_ * sizeof( T_ELEM ) );

            CudaMemory cx_data = CudaMemory( hiddenTensorSize_ * sizeof( T_ELEM ) );
            CudaMemory cy_data = CudaMemory( hiddenTensorSize_ * sizeof( T_ELEM ) );

            CudaMemory dcx_data = CudaMemory( hiddenTensorSize_ * sizeof( T_ELEM ) );
            CudaMemory dcy_data = CudaMemory( hiddenTensorSize_ * sizeof( T_ELEM ) );

            std::cout << "Train()" << " Initializing Tensor memory" << std::endl;

            InitializeData( x_data, hx_data, cx_data, dy_data, dhy_data, dcy_data );

            std::cout << "Train()" << " Starting timer" << std::endl;

            cudaCheckStatus( cudaDeviceSynchronize() );

            //auto profiler = CudaTimer();
            //profiler.Start();

            std::cout << "Train() Calling ForwardStep()" << std::endl;

            ForwardStep( x_data, y_data, hx_data, hy_data, cx_data, cy_data );

            std::cout << "Train() Calling BackwardsStep()" << std::endl;

            BackwardsStep(
                y_data, dy_data,
                x_data, dx_data,
                hx_data, dhy_data, dhx_data,
                cx_data, dcy_data, dcx_data );

            //profiler.Stop();

            //Checksum( y_data, hy_data, cy_data );
        }

        //~RnnModel()
        //{
        //    //CUDA_CALL( cudaFree( workSpace_ ) );
        //    //CUDA_CALL( cudaFree( reserveSpace_ ) );

        //    //CUDA_CALL( cudaFree( dweightBiasSpace_ ) );
        //    //CUDA_CALL( cudaFree( weightBiasSpace_ ) );

        //}

    protected:

        void OnModelBuilding( const DnnModelBuilder& builder ) override
        {
            // TJT: This is so ugly/complicated!
            cudnnHandle_ = builder.GetCudnnContext()->GetCudnnHandle()->GetOpaqueHandle();

            CalculateParameterSizes();

            CreateInputOutputDataSets( builder );

            CreateDropout( builder );

            CreateRnnOp( builder );

            CreateHiddenStateTensors( builder );

            SetupMemorySpace( builder );

            CreateLayers( builder );
        }

    private:

        void CalculateParameterSizes()
        {
            // Compute local parameters
            bidirectionalScale_ = (options_.dirMode == CUDNN_BIDIRECTIONAL ? 2 : 1);

            // Calculating total elements per each tensor
            inputTensorSize_ = options_.sequenceLength * options_.batchSize * options_.inputSize;
            outputTensorSize_ = options_.sequenceLength * options_.batchSize * options_.hiddenSize * bidirectionalScale_;
            hiddenTensorSize_ = options_.numLayers * options_.batchSize * options_.hiddenSize * bidirectionalScale_;
        }

        /// <summary>
        /// Gets the number of linear layers in the Recurrent network.
        /// </summary>
        /// <returns>number of layers</returns>
        int GetNumberOfLinearLayers()
        {
            auto linear_layers = 0;

            if ( options_.cellMode == CUDNN_RNN_RELU || options_.cellMode == CUDNN_RNN_TANH )
            {
                linear_layers = 2;
            }
            else if ( options_.cellMode == CUDNN_LSTM )
            {
                linear_layers = 8;
            }
            else if ( options_.cellMode == CUDNN_GRU )
            {
                linear_layers = 6;
            }

            return linear_layers;
        }

        void CreateInputOutputDataSets( const DnnModelBuilder& builder )
        {
            cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

            double paddingFill = options_.paddingFill;

            seqLengthArray_.reserve( options_.batchSize );

            for ( int i = 0; i < options_.batchSize; i++ )
            {
                seqLengthArray_.push_back( options_.sequenceLength );
            }

            x = builder.Create<RnnDataSet>();

            x.SetDataType( dataType )
                .SetLayout( CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED )
                .SetMaxSequenceLength( options_.sequenceLength )
                .SetBatchSize( options_.batchSize )
                .SetVectorSize( options_.vectorSize )
                .SetSequenceLengthArray( seqLengthArray_ )
                .SetMemorySize( inputTensorSize_ * sizeof( T_ELEM ) )
                .SetPaddingFill( paddingFill );

            x.Finalize();

            y = builder.Create<RnnDataSet>();

            y.SetDataType( dataType )
                .SetLayout( CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED )
                .SetMaxSequenceLength( options_.sequenceLength )
                .SetBatchSize( options_.batchSize )
                .SetVectorSize( options_.hiddenSize * bidirectionalScale_ )
                .SetSequenceLengthArray( seqLengthArray_ )
                .SetMemorySize( inputTensorSize_ * sizeof( T_ELEM ) )
                .SetPaddingFill( paddingFill );

            y.Finalize();
        }

        void CreateHiddenStateTensors( DnnModelBuilder builder )
        {
            std::cout << ">>> CreateHiddenStateTensors()\n";

            // TJT: Generalize this
            int dimensions = 3;

            // Shape and Strides for hidden state tensors
            std::vector<int> shape = {
                options_.numLayers * bidirectionalScale_,
                options_.batchSize,
                options_.hiddenSize};

            std::vector<int> strides = {
                shape[ 1 ] * shape[ 2 ],
                shape[ 2 ],
                1};

            h = builder.Create<Tensor>();

            h.SetDataType( options_.dataType )
                //.SetMemorySize( hiddenTensorSize * sizeof( T_ELEM ) )
                .SetDimensions( 3, shape, strides );

            h.Finalize();

            // The c or CellMode tensor is only needed for LSTM network types.
            c = builder.Create<Tensor>();
            c.SetDataType( options_.dataType )
                .SetDimensions( 3, shape, strides );

            c.Finalize();

            std::cout << "<<< CreateHiddenStateTensors()" << std::endl;
        }

        void CreateDropout( DnnModelBuilder builder )
        {
            dropout_ = builder.Create<Dropout>();

            dropout_.SetProbability( 0.1f )
                .SetSeed( 739134ull );

            dropout_.Finalize();
        }

        void CreateRnnOp( DnnModelBuilder builder )
        {
            rnnOp_ = builder.Create<RnnOperation>();
            rnnOp_.SetAlgorithm( options_.algorithm )
                .SetCellMode( options_.cellMode )
                .SetBiasMode( options_.biasMode )
                .SetDirectionMode( options_.dirMode )
                .SetInputMode( options_.inputMode )
                .SetDataType( options_.dataType )
                .SetMathPrecision( options_.mathPrecision )
                .SetMathType( options_.mathType )
                .SetInputSize( options_.inputSize )
                .SetHiddenSize( options_.hiddenSize )
                .SetProjSize( options_.projSize )
                .SetNumLayers( options_.numLayers )
                .SetDropout( dropout_ );

            rnnOp_.Finalize();
        }

        void SetupMemorySpace( DnnModelBuilder builder )
        {
            // Set up weights and bias parameters
            auto status = cudnnGetRNNWeightSpaceSize(
                builder.GetCudnnContext()->GetCudnnHandle()->GetOpaqueHandle(),
                static_cast<cudnnRNNDescriptor_t>(rnnOp_.GetOpaqueDescriptor()),
                &weightSpaceSize_ );

            if ( status != CUDNN_STATUS_SUCCESS )
            {
                throw CuDnn::cudnnException( "Failed to get RNN WeightBias space size.", status );
            }

            std::cout << "Allocating Cuda weightBiasSpace: " << std::to_string( weightSpaceSize_ ) << std::endl;
            weightBiasSpace_ = CudaMemory( weightSpaceSize_ );
            dweightBiasSpace_ = CudaMemory( weightSpaceSize_ );

            memoryAllocated_ += (2 * weightSpaceSize_);

            status = cudnnGetRNNTempSpaceSizes(
                builder.GetCudnnContext()->GetCudnnHandle()->GetOpaqueHandle(),
                static_cast<cudnnRNNDescriptor_t>(rnnOp_.GetOpaqueDescriptor()),
                CUDNN_FWD_MODE_TRAINING,
                static_cast<cudnnRNNDataDescriptor_t>(x.GetOpaqueDescriptor()),
                &workSpaceSize_,
                &reserveSpaceSize_ );

            if ( status != CUDNN_STATUS_SUCCESS )
            {
                throw CuDnn::cudnnException( "Failed to get workspace/reserve space size", status );
            }

            std::cout << "Allocating Cuda workSpace: " << std::to_string( workSpaceSize_ ) << std::endl;
            std::cout << "Allocating Cuda reserveSpace: " << std::to_string( reserveSpaceSize_ ) << std::endl;
            workSpace_ = CudaMemory( workSpaceSize_ );
            reserveSpace_ = Cuda::CudaMemory( reserveSpaceSize_ );

            memoryAllocated_ += (workSpaceSize_ + reserveSpaceSize_);
        }

        /// <summary>
        /// Creates the linear layers <seealso ref="RnnLinearLayer"/> for the Recurrent network type.
        /// </summary>
        /// <param name="builder">Model builder</param>
        void CreateLayers( const DnnModelBuilder& builder )
        {
            std::cout << ">>> CreateLayers()\n";

            auto linLayers = GetNumberOfLinearLayers();

            for ( int layerId = 0; layerId < options_.numLayers * bidirectionalScale_; layerId++ )
            {
                for ( int linLayerId = 0; linLayerId < linLayers; linLayerId++ )
                {
                    std::cout << "Initializing layer ("
                        << std::to_string( layerId ) << ", "
                        << std::to_string( linLayerId ) << ")" << std::endl;

                    StateTensor w = builder.Create<StateTensor>();
                    StateTensor b = builder.Create<StateTensor>();

                    cudnnDataType_t dataTypeTemp;
                    int nbDims = 0;
                    int shape[ 8 ], stride[ 8 ];

                    T_ELEM* weightMatrixAddress = NULL;
                    T_ELEM* biasVectorAddress = NULL;

                    auto status = cudnnGetRNNWeightParams(
                        builder.GetCudnnContext()->GetCudnnHandle()->GetOpaqueHandle(),
                        static_cast<cudnnRNNDescriptor_t>(rnnOp_.GetOpaqueDescriptor()),
                        layerId,
                        weightBiasSpace_.GetBufferSize(),
                        weightBiasSpace_.GetBuffer(),
                        linLayerId,
                        static_cast<cudnnTensorDescriptor_t>(w.GetOpaqueDescriptor()),
                        (void**)&weightMatrixAddress,
                        static_cast<cudnnTensorDescriptor_t>(b.GetOpaqueDescriptor()),
                        (void**)&biasVectorAddress );

                    if ( status != CUDNN_STATUS_SUCCESS )
                    {
                        throw CuDnn::cudnnException( "Failed to get RNN Weight Params", status );
                    }

                    if ( weightMatrixAddress )
                    {
                        status = cudnnGetTensorNdDescriptor(
                            static_cast<cudnnTensorDescriptor_t>(w.GetOpaqueDescriptor()),
                            3, &dataTypeTemp, &nbDims, shape, stride );

                        if ( status != CUDNN_STATUS_SUCCESS )
                        {
                            throw CuDnn::cudnnException( "Failed to get TensorNdDescriptor for weight states.", status );
                        }

                        w.SetDataType( dataTypeTemp );
                        w.SetAddress( weightMatrixAddress );
                        w.SetDimensions( nbDims, shape, stride );

                        // TJT: MOVE - Needs to be done in Layer initialization 
                        InitGPUData<T_ELEM>(
                            weightMatrixAddress,
                            shape[ 0 ] * shape[ 1 ] * shape[ 2 ],
                            T_ELEM( 1.0 ) / (shape[ 0 ] * shape[ 1 ] * shape[ 2 ]) );
                    }

                    if ( biasVectorAddress )
                    {
                        status = cudnnGetTensorNdDescriptor(
                            static_cast<cudnnTensorDescriptor_t>(b.GetOpaqueDescriptor()),
                            3, &dataTypeTemp, &nbDims, shape, stride );

                        if ( status != CUDNN_STATUS_SUCCESS )
                        {
                            throw CuDnn::cudnnException( "Failed to get TensorNdDescriptor for weight states.", status );
                        }

                        b.SetDataType( dataTypeTemp );
                        b.SetAddress( biasVectorAddress );
                        b.SetDimensions( nbDims, shape, stride );

                        // TJT: See above
                        InitGPUData<T_ELEM>(
                            biasVectorAddress,
                            shape[ 0 ] * shape[ 1 ] * shape[ 2 ],
                            1.0 );
                    }

                    RnnLinearLayer layer = RnnLinearLayer( layerId, linLayerId, w, b );

                    layers_.Add( layer );
                }
            }

            std::cout << "<<< CreateLayers()" << std::endl;
        }

        void InitializeData(
            CudaMemory& x_data,
            CudaMemory& hx_data,
            CudaMemory& cx_data,
            CudaMemory& dy_data,
            CudaMemory& dhy_data,
            CudaMemory& dcy_data )
        {
            // TJT: Generalize this with to API

            // We initialise to something simple.
            // Matrices are initialised to 1 / matrixSize,
            // biases to 1, 
            // data is 1.

            std::cout << "InitializeData()" << " Initializing x_data memory to 1.0" << std::endl;

            // Initialize inputs
            InitGPUData<T_ELEM>( (T_ELEM*)x_data.GetBuffer(), inputTensorSize_, 1.0 );

            if ( hx_data.GetBuffer() != NULL )
            {
                std::cout << "InitializeData()" << " Initializing hx_data memory to 1.0" << std::endl;

                InitGPUData<T_ELEM>( (T_ELEM*)hx_data.GetBuffer(), hiddenTensorSize_, 1.0 );
            }

            if ( cx_data.GetBuffer() != NULL )
            {
                std::cout << "InitializeData()" << " Initializing cx_data memory to 1.0" << std::endl;

                InitGPUData<T_ELEM>( (T_ELEM*)cx_data.GetBuffer(), hiddenTensorSize_, 1.0 );
            }

            // Initialize outputs
            InitGPUData<T_ELEM>( (T_ELEM*)dy_data.GetBuffer(), outputTensorSize_, 1.0 );

            if ( dhy_data.GetBuffer() != NULL )
            {
                std::cout << "InitializeData()" << " Initializing dhy_data memory to 1.0" << std::endl;
                InitGPUData<T_ELEM>( (T_ELEM*)dhy_data.GetBuffer(), hiddenTensorSize_, 1.0 );
            }

            if ( dcy_data.GetBuffer() != NULL )
            {
                std::cout << "InitializeData()" << " Initializing dcy_data memory to 1.0" << std::endl;
                InitGPUData<T_ELEM>( (T_ELEM*)dcy_data.GetBuffer(), hiddenTensorSize_, 1.0 );
            }

            std::cout << "InitializeData()" << " Initializing tensor memory completed" << std::endl;
        }

    private:

        void Checksum(
            CudaMemory& ds,
            CudaMemory& h,
            CudaMemory& c )
        {
            T_ELEM* ds_output;
            T_ELEM* h_output;
            T_ELEM* c_output;

            ds_output = (T_ELEM*)malloc( ds.GetBufferSize() );
            h_output = (T_ELEM*)malloc( h.GetBufferSize() );
            c_output = (T_ELEM*)malloc( c.GetBufferSize() );

            cudaCheckStatus( cudaMemcpy(
                ds_output,
                ds.GetBuffer(),
                ds.GetBufferSize(),
                cudaMemcpyDeviceToHost ) );

            if ( h.GetBuffer() != nullptr )
            {
                cudaCheckStatus( cudaMemcpy(
                    h_output,
                    h.GetBuffer(),
                    h.GetBufferSize(),
                    cudaMemcpyDeviceToHost ) );
            }

            if ( c.GetBuffer() != nullptr )
            {
                cudaCheckStatus( cudaMemcpy( c_output,
                    c.GetBuffer(),
                    c.GetBufferSize(),
                    cudaMemcpyDeviceToHost ) );
            }

            double ds_checksum = 0.f;
            double h_checksum = 0.f;
            double c_checksum = 0.f;

            for ( int m = 0; m < options_.batchSize; m++ )
            {
                double localSumi = 0;
                double localSumh = 0;
                double localSumc = 0;

                for ( int j = 0; j < options_.sequenceLength; j++ )
                {
                    for ( int i = 0; i < hiddenTensorSize_ * bidirectionalScale_; i++ )
                    {
                        localSumi += (double)ds_output[ j * options_.batchSize * hiddenTensorSize_ * bidirectionalScale_ + m * hiddenTensorSize_ * bidirectionalScale_ + i ];
                    }
                }
                for ( int j = 0; j < numLinearLayers_ * bidirectionalScale_; j++ )
                {
                    for ( int i = 0; i < hiddenTensorSize_; i++ )
                    {
                        if ( h.GetBuffer() != nullptr )
                        {
                            localSumh += (double)h_output[ j * hiddenTensorSize_ * options_.batchSize + m * hiddenTensorSize_ + i ];
                        }
                        if ( c.GetBuffer() != nullptr )
                        {
                            localSumc += (double)c_output[ j * hiddenTensorSize_ * options_.batchSize + m * hiddenTensorSize_ + i ];
                        }
                    }
                }

                ds_checksum += localSumi;
                h_checksum += localSumh;
                c_checksum += localSumc;
            }

            std::cout << "ds checksum: " << std::to_string( ds_checksum ) << std::endl;
            std::cout << "h checksum: " << std::to_string( h_checksum ) << std::endl;

            if ( c.GetBuffer() != nullptr )
            {
                std::cout << "c checksum: " << std::to_string( c_checksum ) << std::endl;
            }

            free( ds_output );
            free( c_output );
            free( h_output );
        }

        void ForwardStep(
            CudaMemory& x_data,
            CudaMemory& y_data,
            CudaMemory& hx_data,
            CudaMemory& hy_data,
            CudaMemory& cx_data,
            CudaMemory& cy_data )
        {
            std::cout << ">>> ForwardStep()..";

            auto status = cudnnRNNForward(
                cudnnHandle_,
                static_cast<cudnnRNNDescriptor_t>(rnnOp_.GetOpaqueDescriptor()),
                CUDNN_FWD_MODE_TRAINING,
                x.GetSeqLengthArray(),
                static_cast<cudnnRNNDataDescriptor_t>(x.GetOpaqueDescriptor()),
                x_data.GetBuffer(),
                static_cast<cudnnRNNDataDescriptor_t>(y.GetOpaqueDescriptor()),
                y_data.GetBuffer(),
                static_cast<cudnnTensorDescriptor_t>(h.GetOpaqueDescriptor()),
                hx_data.GetBuffer(),
                hy_data.GetBuffer(),
                static_cast<cudnnTensorDescriptor_t>(c.GetOpaqueDescriptor()),
                cx_data.GetBuffer(),
                cy_data.GetBuffer(),
                weightBiasSpace_.GetBufferSize(),
                weightBiasSpace_.GetBuffer(),
                workSpace_.GetBufferSize(),
                workSpace_.GetBuffer(),
                reserveSpace_.GetBufferSize(),
                reserveSpace_.GetBuffer() );

            if ( status != CUDNN_STATUS_SUCCESS )
            {
                std::cout << "CudnnForward() failed: " << Mila::Dnn::CuDnn::to_string( status ) << std::endl;

                throw CuDnn::cudnnException( "Failed CudnnForward() call", status );
            }

            std::cout << " Done." << std::endl;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x_data">RNN primary input</param>
        /// <param name="y_data">primary output</param>
        /// <param name="dy_data">gradient deltas with respect to y</param>
        /// <param name=""></param>
        void BackwardsStep(
            CudaMemory& y_data, CudaMemory& dy_data,
            CudaMemory& x_data, CudaMemory& dx_data,
            CudaMemory& hx_data, CudaMemory& dhy_data, CudaMemory& dhx_data,
            CudaMemory& cx_data, CudaMemory& dcy_data, CudaMemory& dcx_data )
        {
            std::cout << ">>> BackwardsStep()..";

            auto status = cudnnRNNBackwardData_v8(
                cudnnHandle_,
                static_cast<cudnnRNNDescriptor_t>(rnnOp_.GetOpaqueDescriptor()),
                x.GetSeqLengthArray(),
                static_cast<cudnnRNNDataDescriptor_t>(y.GetOpaqueDescriptor()),
                y_data.GetBuffer(), dy_data.GetBuffer(),
                static_cast<cudnnRNNDataDescriptor_t>(x.GetOpaqueDescriptor()),
                dx_data.GetBuffer(),
                static_cast<cudnnTensorDescriptor_t>(h.GetOpaqueDescriptor()),
                hx_data.GetBuffer(),
                dhy_data.GetBuffer(),
                dhx_data.GetBuffer(),
                static_cast<cudnnTensorDescriptor_t>(c.GetOpaqueDescriptor()),
                cx_data.GetBuffer(),
                dcy_data.GetBuffer(),
                dcx_data.GetBuffer(),
                weightBiasSpace_.GetBufferSize(),
                weightBiasSpace_.GetBuffer(),
                workSpace_.GetBufferSize(),
                workSpace_.GetBuffer(),
                reserveSpace_.GetBufferSize(),
                reserveSpace_.GetBuffer() );

            if ( status != CUDNN_STATUS_SUCCESS )
            {
                std::cout << "CudnnBackwardData_v8() failed: " << CuDnn::to_string( status ) << std::endl;

                throw CuDnn::cudnnException( "Failed CudnnBackwardData_v8()", status );
            }

            dweightBiasSpace_.Set( 0 );

            status = cudnnRNNBackwardWeights_v8(
                cudnnHandle_,
                static_cast<cudnnRNNDescriptor_t>(rnnOp_.GetOpaqueDescriptor()),
                CUDNN_WGRAD_MODE_ADD,
                x.GetSeqLengthArray(),
                static_cast<cudnnRNNDataDescriptor_t>(x.GetOpaqueDescriptor()),
                x_data.GetBuffer(),
                static_cast<cudnnTensorDescriptor_t>(h.GetOpaqueDescriptor()),
                hx_data.GetBuffer(),
                static_cast<cudnnRNNDataDescriptor_t>(y.GetOpaqueDescriptor()),
                y_data.GetBuffer(),
                dweightBiasSpace_.GetBufferSize(),
                dweightBiasSpace_.GetBuffer(),
                workSpace_.GetBufferSize(),
                workSpace_.GetBuffer(),
                reserveSpace_.GetBufferSize(),
                reserveSpace_.GetBuffer() );
        }

    private:

        cudnnHandle_t cudnnHandle_ = nullptr;

        RnnModelOptions options_;

        CudaMemory weightBiasSpace_;
        CudaMemory dweightBiasSpace_;
        CudaMemory workSpace_;
        CudaMemory reserveSpace_;

        // TJT: available with CudaMemory struct. To be removed.
        size_t weightSpaceSize_ = 0;
        size_t workSpaceSize_ = 0;
        size_t reserveSpaceSize_ = 0;

        int memoryAllocated_ = 0;

        int bidirectionalScale_ = 0;
        int inputTensorSize_ = 0;
        int outputTensorSize_ = 0;
        int hiddenTensorSize_ = 0;

        int numLinearLayers_ = 0;

        std::vector<int> seqLengthArray_;

        /// <summary>
        /// RNN Input dataset x
        /// </summary>
        RnnDataSet x;

        /// <summary>
        /// RNN Output dataset y
        /// </summary>
        RnnDataSet y;

        /// <summary>
        /// Hidden state tensor
        /// </summary>
        Tensor h;

        /// <summary>
        /// Cell state tensor. For LSTM networks only.
        /// </summary>
        Tensor c;

        /// <summary>
        /// Dropout operation required for RNN operation.
        /// </summary>
        Dropout dropout_;

        /// <summary>
        /// RNN operation.
        /// </summary>
        RnnOperation rnnOp_;

        RnnLayerCollection layers_;
    };

    export template class RnnModel<float>;
    export template class RnnModel<double>;
}