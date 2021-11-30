/**
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
  */

#ifndef _MILA_DNN_TYPE_H_
#define _MILA_DNN_TYPE_H_

namespace Mila::Dnn
{
    typedef enum
    {
        CONVOLUTIONAL_NN_TYPE = 0,
        RECURRENT_NN_TYPE = 1
    } neuralNetType_t;

    static inline std::string to_string( neuralNetType_t netType )
    {
        switch ( netType )
        {
        case CONVOLUTIONAL_NN_TYPE:
            return std::string( "CONVOLUTIONAL_NN_TYPE" );
        case RECURRENT_NN_TYPE:
            return std::string( "RECURRENT_NN_TYPE" );

        default:
            return std::string( "Invalid neuralNetType_t" );
        }
    };
}
#endif