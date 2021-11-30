/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef MILA_DNN_RNNDATASET_H_
#define MILA_DNN_RNNDATASET_H_

#include <cudnn.h>

#include "Cuda/CudaError.h"

#include "CuDNN/CudnnContext.h"
#include "CuDNN/Descriptor.h"
#include "CuDNN/Error.h"
#include "CuDNN/Utils.h"

using namespace Mila::Dnn::CuDNN;
using namespace Mila::Dnn::Cuda;

namespace Mila::Dnn
{
    /// <summary>
    /// The data descriptor of an RNN operation.
    /// Use RNNDataDescriptorBuilder to build an RNNDataDescriptor instance.
    /// </summary>
    class RnnDataSet : public CuDNN::Descriptor
    {
        friend class DnnModelBuilder;

    public:

        RnnDataSet() = default;

        /// <summary>
        /// Constructor on anBuilds a instance of a RNN DataSet tensor.
        /// </summary>
        /// <returns>A managed RNN data tensor instance.</returns>
        explicit RnnDataSet( const ManagedCudnnHandle& cudnnHandle )
            : Descriptor( cudnnHandle, CUDNN_RNNDATA_DESCRIPTOR )
        {
        }

        /// <summary>
        /// Sets the datatype of the RNN data tensor.
        /// </summary>
        /// <param name="data_type_">Data type value</param>
        /// <returns>RNN data set builder</returns>
        auto SetDataType( cudnnDataType_t dataType ) -> RnnDataSet&
        {
            CheckFinalizedThrow();
            data_type_ = dataType;
            
            return *this;
        };

        /// <summary>
        /// Sets the memory layout of the RNN data tensor.
        /// </summary>
        /// <param name="layoutType">Data tensor layout value</param>
        /// <returns>RNN data set builder</returns>
        auto SetLayout( cudnnRNNDataLayout_t layoutType ) -> RnnDataSet&
        {
            CheckFinalizedThrow();
            layout_ = layoutType;
            
            return *this;
        };

        /// <summary>
        /// Sets the maximum sequence length within this RNN data tensor.
        /// </summary>
        /// <param name="max_seq_length_"></param>
        /// <returns></returns>
        auto SetMaxSeqLength( int maxSeqLength ) -> RnnDataSet&
        {
            CheckFinalizedThrow();
            max_seq_length_ = maxSeqLength;
            
            return *this;
        };

        /// <summary>
        /// Sets The number of sequences within the mini-batch.
        /// </summary>
        /// <param name="batch_size_"></param>
        /// <returns></returns>
        auto SetBatchSize( int batch_size ) -> RnnDataSet&
        {
            CheckFinalizedThrow();
            batch_size_ = batch_size;
            
            return *this;
        };

        /// <summary>
        /// Sets the vector length (embedding size) of the input or output tensor at each time-step.
        /// </summary>
        /// <param name="vectorSize"></param>
        /// <returns></returns>
        auto SetVectorSize( int vector_size ) -> RnnDataSet&
        {
            CheckFinalizedThrow();
            vector_size_ = vector_size;
            
            return *this;
        };

        /// <summary>
        /// An integer array with batchSize number of elements that describes the length 
        /// (number of time-steps) of each sequence. 
        /// </summary>
        /// <param name="vector_size_"></param>
        /// <returns></returns>
        auto SetSeqLengthArray( const std::vector<int>& seqLengthArray ) -> RnnDataSet&
        {
            CheckFinalizedThrow();
            seq_length_array_.assign( seqLengthArray.begin(), seqLengthArray.end() );
            
            return *this;
        }
        
        auto SetMemorySize( size_t memorySize ) -> RnnDataSet&
        {
            CheckFinalizedThrow();
            descriptor_memory_size_ = memorySize;

            return *this;
        }
        /// <summary>
        /// Sets the symbol for filling the padding position in RNN output.
        /// </summary>
        /// <param name="padding_fill_"></param>
        /// <returns></returns>
        auto SetPaddingFill( double paddingFill ) -> RnnDataSet&
        {
            CheckFinalizedThrow();
            padding_fill_ = paddingFill;
            
            return *this;
        }

        cudnnStatus_t Finalize() override
        {
            if ( IsFinalized() )
            {
                throw std::runtime_error( "RnnDataset is already finalized." );
            }

            InitializeSeqLengthArray();

            descriptor_memory_ = CudaMemory( descriptor_memory_size_ );

            auto status = cudnnSetRNNDataDescriptor(
                static_cast<cudnnRNNDataDescriptor_t>( GetOpaqueDescriptor()),
                data_type_,
                layout_,
                max_seq_length_,
                batch_size_,
                vector_size_,
                seq_length_array_.data(),
                &padding_fill_ );

            if ( status != CUDNN_STATUS_SUCCESS )
            {
                SetErrorAndThrow(
                    this, status, "Finalize RnnDataSet failed." );

                return status;
            }

            return SetFinalized();
        }

        const int* GetSeqLengthArray()
        {
            return (int*)dev_seq_length_array_.GetBuffer();
        }

        std::string ToString() const override
        {
            std::stringstream ss;

            ss << "RnnDataSet::" << std::endl
                << " Datatype: " << CuDNN::to_string( data_type_ ) << std::endl
                << " Layout: " << CuDNN::to_string( layout_ ) << std::endl
                << " MaxSeqLength: " << std::to_string( max_seq_length_ ) << std::endl
                << " BatchSize: " << std::to_string( batch_size_ ) << std::endl
                << " VectorSize: " << std::to_string( vector_size_ ) << std::endl
                << " PaddingFill: " << std::to_string( padding_fill_ ) << std::endl;

            return ss.str();
        }

    private:

        void InitializeSeqLengthArray()
        {
            dev_seq_length_array_ = CudaMemory( batch_size_ * sizeof( int ) );

            cudaCheckStatus( cudaMemcpy(
                dev_seq_length_array_.GetBuffer(),
                seq_length_array_.data(),
                batch_size_ * sizeof( int ),
                cudaMemcpyHostToDevice ) );
        }

        CudaMemory dev_seq_length_array_;
        
        /// <summary>
        /// Data pointer to the GPU memory associated with this RnnDataSet descriptor.
        /// </summary>
        CudaMemory descriptor_memory_;

        // RnnDataSet properties
        cudnnDataType_t data_type_ = CUDNN_DATA_FLOAT;
        cudnnRNNDataLayout_t layout_ = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
        int max_seq_length_ = 0;
        int batch_size_ = 0;
        int vector_size_ = 0;
        int descriptor_memory_size_ = 0;
        double padding_fill_ = 0.0;
        std::vector<int> seq_length_array_;
    };
}
#endif