/**
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef MILA_DNN_MODEL_H_
#define MILA_DNN_MODEL_H_

#include <string>
#include <sstream>

#include <cudnn.h>

#include "CuDNN/CudnnContext.h"
#include "DnnModelOptions.h"
#include "DnnModelBuilder.h"
#include "NeuralNetType.h"

using namespace Mila::Dnn::CuDNN;

namespace Mila::Dnn
{
    class DnnModel
    {
    public:

        DnnModel() : DnnModel( DnnModelOptions() )
        {}

        DnnModel( const DnnModelOptions& options )
        {
            options_ = options;
            context_ = std::make_unique<CudnnContext>();
            builder_ = DnnModelBuilder( context_ );
        }

        const DnnModelBuilder& GetModelBuilder()
        {
            return builder_;
        }

        virtual void BuildModel() final
        {
            OnModelBuilding( builder_ );
        }

    protected:

        /// <summary>
        /// Override to configure the application dnn model.
        /// </summary>
        /// <param name="builder"></param>
        virtual void OnModelBuilding( DnnModelBuilder builder )
        {
        }

    private:

        DnnModel( DnnModel const& ) = delete;
        DnnModel& operator=( DnnModel const& ) = delete;

        DnnModelOptions options_;
        DnnModelBuilder builder_;
        ManagedCudnnContext context_ = nullptr;
       
        neuralNetType_t type_ = RECURRENT_NN_TYPE;
    };
}
#endif