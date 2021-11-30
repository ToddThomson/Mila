/**
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
  */

#ifndef MILA_DNN_MODEL_BUILDER_H_
#define MILA_DNN_MODEL_BUILDER_H_

#include <string>

#include "CuDNN/CudnnContext.h"
#include "CuDNN/Descriptor.h"
//#include "CuDNN/Error.h"

#include "DnnModel.h"

using namespace Mila::Dnn::CuDNN;

namespace Mila::Dnn
{
    class DnnModelBuilder
    {
    public:

        DnnModelBuilder( const ManagedCudnnContext& cudnnContext )
        {
            std::cout << "DnnModelBuilder() constructor\n";
            cudnn_context_ = cudnnContext;
        }

        /// <summary>
        /// Factory for <seealso ref="Descriptor" /> 
        /// </summary>
        /// <typeparam name="TDesc">The type of Descriptor to create.</typeparam>
        /// <returns>The reference to the descriptor created.</returns>
        template<class TDesc>
        TDesc Create()
        {
            return TDesc( cudnn_context_->GetCudnnHandle() );
        }

        const ManagedCudnnContext& GetCudnnContext()
        {
            return cudnn_context_;
        }

        DnnModelBuilder() = default;

    private:
        
        ManagedCudnnContext cudnn_context_ = nullptr;
    };
}
#endif