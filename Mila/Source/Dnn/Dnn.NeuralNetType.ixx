/**
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
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

export module Dnn.NeuralNetType;

namespace Mila::Dnn
{
    export typedef enum
    {
        CONVOLUTIONAL_NN_TYPE = 0,
        RECURRENT_NN_TYPE = 1
    } neuralNetType_t;

    export static inline std::string to_string( neuralNetType_t netType )
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