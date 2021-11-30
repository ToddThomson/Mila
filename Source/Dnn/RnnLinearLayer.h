/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef MILA_DNN_RNN_LINEAR_LAYER_H_
#define MILA_DNN_RNN_LINEAR_LAYER_H_

#include <string>
#include <memory>

#include <cudnn.h>

#include "StateTensor.h"

namespace Mila::Dnn
{
    class RnnLinearLayer
    {
    public:

        RnnLinearLayer( 
            int layerId, 
            int linearLayerId, 
            StateTensor& weightMatrix, 
            StateTensor& biasVector )
        {
            layer_id_ = layerId;
            linear_layer_id_ = linearLayerId;

            weight_matrix_ = std::move( weightMatrix );
            bias_vector_ = std::move( biasVector );
        }

        std::string ToString()
        {
            std::stringstream ss;
            char sep = ' ';
            ss << "RnnLinearLayer:: " << std::endl;

            return ss.str();
        }

        bool HasWeightMatrix()
        {
            return false;// (weight_matrix_.GetAddress() != NULL);
        }

        bool HasBiasVector()
        {
            return false;// (bias_vector_.GetAddress() != NULL);
        }

    private:

        int layer_id_ = 0;
        int linear_layer_id_ = 0;

        StateTensor weight_matrix_;
        StateTensor bias_vector_;
    };
}
#endif